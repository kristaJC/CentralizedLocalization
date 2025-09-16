from abc import ABC, abstractmethod
import time
import mlflow
from contextlib import contextmanager
import json
import pandas as pd
import datetime
import re
import openai
import io
from openai import OpenAI
import gspread
import random
import logging
import os
from unidecode import unidecode
from pathlib import Path
import pyspark
from pyspark.sql.functions import *
from functools import reduce
from typing import *
import tiktoken

from general_config import *
from ml_tracker import MLTracker

EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"
CENTRAL_SHEET_URL = ""

### Helper function to update the status
def update_status(gc, sheet_url, tab, row_fingerprint, status, run_id=None, notes=None):
    ws = gc.open_by_url(sheet_url).worksheet(tab)
    data = ws.get_all_values()
    headers = data[0]; rows = data[1:]
    idx = {h:i for i,h in enumerate(headers)}
    target = None
    for r_i, r in enumerate(rows, start=2):
        if r[idx["RowFingerprint"]] == row_fingerprint:
            target = r_i; break
    if not target: return

    def set_cell(col, val):
        if col in idx:
            ws.update_cell(target, idx[col]+1, val)

    set_cell("Status", status)
    set_cell("RunID", run_id or "")
    set_cell("Notes", (notes or "")[:500])
    set_cell("LastUpdated", datetime.datetime.utcnow().isoformat())


class LocalizationRun(ABC):
    def __init__(self, 
                 request, 
                 gsheet_client=None, 
                 gpt_client=None, 
                 cfg=None,
                 tracker: MLTracker | None = None):
        """
        request: dict with fields like RequestID, LocType, Game, TargetLanguages, etc.
        gsheet_client: injected dependency for Google Sheets
        gpt_client: injected dependency for GPT API
        cfg: loaded YAML/JSON config for this LocType
        """
        self.request = request
        self.gc = gsheet_client
        self.gpt = gpt_client
        self.cfg = cfg or {}
        self.artifacts = {}
        self.tracker =  MLTracker(request, language=None)
        self.lang_trackers = {}


    def _parse_model_json_block(self, raw_output: str):
        """
        Clean and parse a JSON-like string from a model output wrapped in markdown code block.
        """
        try:
            # Strip markdown-style code block markers like ```json ... ```
            cleaned = re.sub(r"^```json|```$", "", raw_output.strip(), flags=re.IGNORECASE).strip()
            # Remove escaped newlines
            cleaned = cleaned.replace("\\n", "").replace("\n", "").strip()
            loaded = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Could not parse JSON: {e}")

        if isinstance(loaded, str):
            try:
                return json.loads(loaded)
            except Exception as e:
                raise ValueError(f"Could not parse inner JSON: {e}")
        return loaded
    
    def _format_results(self, final_rows=None):
        """Subclasses may override. Default: no-op."""
        return final_rows, None

    def _to_dataframe(self, lang: str, raw_output: str) -> pd.DataFrame:
        """
        Convert raw model output for one language into a DataFrame,
        adding a 'lang' column if not already present.
        """
        parsed = self._parse_model_json_block(raw_output)
        df = pd.DataFrame(parsed)
        if "lang" not in df.columns:
            df.insert(0, "lang", lang)
        return df

    # ---- QC hooks (no-op by default) ----
    def qc_checks(self, final_rows: Any) -> Dict[str, Any]:
        """
        Return a dict report. Expected shape (convention):
        {
          "ok": bool,
          "issues": List[dict],         # each issue has enough info to re-run/repair
          "stats": {"overlimit": int},  # optional
        }
        Default: no QC → ok=True
        """
        return {"ok": True, "issues": [], "stats": {}}

    def qc_repair(self, final_rows: Any, report: Dict[str, Any], attempt: int) -> Any:
        """
        Given a QC report with issues, produce a repaired final_rows (e.g., re-translate bad rows).
        Default: return final_rows unchanged.
        """
        return final_rows

    def run(self):
        parent_run_id = self.tracker.start()  # parent request-level run
        try:
            with self.tracker.step("validate_inputs"):
                self.validate_inputs()

            with self.tracker.step("load_inputs"):
                data = self.load_inputs()
                #self.tracker.dict({"preview": str(data)[:2000]}, "snapshots/input_preview.json")

            with self.tracker.step("preprocess"):
                prepped = self.preprocess(data)
                self.tracker.metrics({"rows.prepped": len(prepped)})

            with self.tracker.step("build_prompts"):
                prompts = self.build_prompts(prepped)
                self.tracker.metrics({"prompts.count": len(prompts)})

            # Build per-language trackers (child runs) using request languages
            for lang in self.languages:
                self.lang_trackers[lang] = MLTracker(
                    request=self.request,
                    language=lang,
                    experiment_name=self.tracker.experiment_name  # same experiment
                )

            outputs = self.translate(prompts)  # tracked per language below

            with self.tracker.step("postprocess"):
                final_rows = self.postprocess(outputs)
                self.tracker.metrics({"rows.final": len(final_rows)})
                

            # ---- NEW: formatting + artifact logging hook ----
            with self.tracker.step("format_results_and_log"):
                formatted, artifacts = self._format_results(final_rows)
                # artifacts is an optional dict like {"aso_wide": df, "aso_long": df, "notes": "str", ...}
                self._log_declared_artifacts(artifacts)

            # Strict policies: enforce placeholders BEFORE char-limit QC
            if (self.cfg or {}).get("char_limit_policy", "").lower() == "strict":
                with self.tracker.step("qc_placeholders_enforce"):
                    long_df = _get_long_df(formatted) #TODO - add helper
                    fixed_long, added = _enforce_placeholders_on_df(long_df) #TODO: Add helper
                    formatted = _set_long_df(formatted, fixed_long) #TODO: Add helper
                    # metrics so you can see impact per run
                    self.tracker.metrics({"qc.placeholders_added": float(added)})

            # ---- QC loop (config-driven) ----
            qc_cfg = (self.cfg or {}).get("qc", {})
            enabled = bool(qc_cfg.get("enabled", False))
            max_retries = int(qc_cfg.get("max_retries", 0))

            if enabled:
                for attempt in range(1, max_retries + 1):
                    with self.tracker.step(f"qc_attempt_{attempt}"):
                        report = self.qc_checks(formatted)
                        # Log a few metrics
                        stats = report.get("stats", {})
                        if stats:
                            self.tracker.metrics({f"qc.{k}": v for k, v in stats.items()})
                        self.tracker.event(f"QC attempt {attempt}: ok={report.get('ok')} issues={len(report.get('issues', []))}")

                        if report.get("ok", True):
                            break  # all good
                        # try to repair
                        formatted = self.qc_repair(formatted, report, attempt)

                # (optional) final QC assert
                final_report = self.qc_checks(formatted)
                if not final_report.get("ok", True):
                    self.tracker.event("QC final: still failing after retries")
                    # You can decide to raise or proceed. Here we proceed but tag:
                    self.tracker.params({"qc_final_ok": False})
                else:
                    self.tracker.params({"qc_final_ok": True})


            with self.tracker.step("write_outputs"):
                self.write_outputs(final_rows)

            self.tracker.end(succeeded=True)
            
            # Update succesful status in central sheet
            #optional:Notes
            #update_status(self.gc, CENTRAL_SHEET_URL, "Requests", self.request   ["RowFingerprint"], "SUCCEEDED", parent_run_id,)

            return {"status": "SUCCEEDED", "run_id": parent_run_id}
        except Exception as e:
            self.tracker.end(False, str(e))
            #try:
                # Update Failed status in sheet
                #update_status(self.gc, CENTRAL_SHEET_URL, "Requests",
                #            self.request["RowFingerprint"], "FAILED", run_id, str(e))
            #except Exception:
            #    pass
            raise

    # Shared, tracked translate that spins one child run per language
    def translate(self, groups):

        parent = self.tracker

        total_prompt_tokens = 0
        total_completion_tokens = 0
        results_tracker = {}
        results_list = []

        with parent.step("translate"):
            parent.metrics({"translate.groups": len(groups)})

            for lang, prompts in groups.items():

                with self.tracker.child(lang) as t:
                    with t.step("api_call"):
                        raw_out, usage = self._call_model_batch(prompts)
                    p, c = MLTracker.extract_usage_tokens(usage)
                    results_list.append(raw_out)

                    t.metrics({"items.total": len(prompts) if hasattr(prompts,"__len__") else 1,
                            "tokens.prompt.total": p,
                            "tokens.completion.total": c})
                    # per-language DF artifact
                    df_lang = self._to_dataframe(lang, raw_out)
                    results_tracker[lang]=df_lang
                    t.log_artifact_df(df_lang, f"outputs/{lang}.csv")
                    # roll up to parent
                    self.tracker.add_child_summary(lang, {"rows": len(df_lang), "tokens.prompt.total": p, "tokens.completion.total": c})

                    # accumulate into parent
                    total_prompt_tokens += p
                    total_completion_tokens += c
 
            #parent rollup
            parent.metrics({
                "tokens.prompt.total": total_prompt_tokens,
                "tokens.completion.total": total_completion_tokens,
            })
        self.results_list = results_list
        self.results_tracker = results_tracker
        return results_list

     # ---------- helper to log arbitrary artifacts declared by a subclass ----------
    def _log_declared_artifacts(self, artifacts):
        if not artifacts:
            return
        for name, obj in artifacts.items():
            try:
                if hasattr(obj, "to_csv"):  # e.g., pandas DataFrame
                    self.tracker.log_artifact_df(obj, f"custom/{name}.csv")
                elif isinstance(obj, (dict, list)):
                    self.tracker.log_artifact_json(obj, f"custom/{name}.json")
                elif isinstance(obj, str):
                    # Heuristic: JSON-ish strings to .json, else .txt
                    path = f"custom/{name}.json" if obj.strip().startswith(("{","[")) else f"custom/{name}.txt"
                    self.tracker.log_artifact_text(obj, path)
                else:
                    # Fallback: stringify
                    self.tracker.log_artifact_text(str(obj), f"custom/{name}.txt")
            except Exception as e:
                # don't fail the run just because logging an artifact failed
                self.tracker.event(f"Artifact '{name}' logging failed: {e}")

    @abstractmethod
    def _call_model_batch(self, prompt):
        pass

    @abstractmethod
    def validate_inputs(self): 
        pass
    
    @abstractmethod
    def load_inputs(self):
        pass
    
    @abstractmethod
    def preprocess(self, data): 
        pass

    @abstractmethod
    def build_prompts(self, prepped): 
        pass

    @abstractmethod
    def postprocess(self, outputs): 
        pass

    @abstractmethod
    def write_outputs(self, post: Optional[pd.DataFrame] = None): 
        pass
