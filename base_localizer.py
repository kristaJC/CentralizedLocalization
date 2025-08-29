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
                #TODO: here is where we want to update with results
                # If your postprocess returns a list[DataFrame]
                try:
                    merged_preview = pd.concat([df.head(200) for df in final_rows], ignore_index=True)
                    self.tracker.log_artifact_df(merged_preview, "snapshots/postprocess_preview.csv")
                except Exception:
                    pass


            with self.tracker.step("write_outputs"):
                self.write_outputs(final_rows)
                #TODO: here is where we want to update with results
                # If your postprocess returns a list[DataFrame]
                try:
                    merged_preview = pd.concat([df.head(200) for df in final_rows], ignore_index=True)
                    self.tracker.log_artifact_df(merged_preview, "snapshots/postprocess_preview.csv")
                except Exception:
                    pass

            self.tracker.end(succeeded=True)
            
            # Update succesful status in central sheet
            #optional:Notes
            update_status(self.gc, CENTRAL_SHEET_URL, "Requests", self.request   ["RowFingerprint"], "SUCCEEDED", parent_run_id,)

            return {"status": "SUCCEEDED", "run_id": run_id}
        except Exception as e:
            self.tracker.end(False, str(e))
            try:
                # Update Failed status in sheet
                update_status(self.gc, CENTRAL_SHEET_URL, "Requests",
                            self.request["RowFingerprint"], "FAILED", run_id, str(e))
            except Exception:
                pass
            raise

    # Shared, tracked translate that spins one child run per language
    def translate(self, groups):

        parent = self.tracker

        total_prompt_tokens = 0
        total_completion_tokens = 0
        results = []
        results_dict = {}

        with parent.step("translate"):
            parent.metrics({"translate.groups": len(groups)})

            for group_name, prompts in groups.items():
                # pick the language tracker matching group_name; fallback to a generic child tracker
                lang_tracker = self.lang_trackers.get(group_name) or MLTracker(
                    request=self.request,
                    language=group_name,
                    experiment_name=parent.experiment_name
                )

                # Start the child run as nested=True
                lang_run_id = lang_tracker.start(nested=True)
                try:
                    lang_total_p = 0
                    lang_total_c = 0

                    with lang_tracker.step("api_batch"):
                        out, usage = self._call_model_batch(prompts)  # subclass hook
                        results.append(out)

                         results_dict[group_name] = raw_out
                
                        #lang_tracker.dict()

                        lang_total_p = usage.prompt_tokens
                        lang_total_c = usage.completion_tokens


                    # per-language metric
                    lang_tracker.metrics({
                        "items.total": len(prompts),
                        "tokens.prompt.total": lang_total_p,
                        "tokens.completion.total": lang_total_c,
                    })

                    # accumulate into parent
                    total_prompt_tokens += lang_total_p
                    total_completion_tokens += lang_total_c

                    lang_tracker.end(succeeded=True)

                except Exception as e:
                    lang_tracker.end(succeeded=False, err_text=str(e))
                    raise

            # parent rollup
            parent.metrics({
                "tokens.prompt.total": total_prompt_tokens,
                "tokens.completion.total": total_completion_tokens,
            })

        self.results_dict = results_dict
        return results

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
    def write_outputs(self, post): 
        pass
