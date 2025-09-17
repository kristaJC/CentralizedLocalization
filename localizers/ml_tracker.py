from abc import ABC, abstractmethod
import time
import mlflow
from contextlib import contextmanager
import json
import pandas as pd
import datetime
import re
import openai
from openai import OpenAI
import gspread
import random
import logging
import os
import io
from unidecode import unidecode
from pathlib import Path
import pyspark
from pyspark.sql.functions import *
from functools import reduce
from typing import *
import tiktoken

from general_config import *

# sheet_status.py (MVP)

EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


def _to_str(v, maxlen=255): 
    s = str(v)
    return s if len(s) <= maxlen else s[:maxlen]

class MLTracker:
    def __init__(self, 
                 request, 
                 language=None, 
                 experiment_name = EXPERIMENT_NAME):
        
        self.request = request
        self.language = language
        self.experiment_name = experiment_name
        self.run = None
        self._events = []
        self._child_summaries = {}

    @staticmethod
    def _parse_langs(raw_langs: Optional[str]):
        if not raw_langs: return []
        parts = [x.strip() for x in raw_langs.replace(";", ",").split(",")]
        return [x for x in parts if x]

    @staticmethod
    def extract_usage_tokens(usage) -> tuple[int, int]:
        if usage is None:
            return 0, 0
        # nested attr (resp.usage.prompt_tokens)
        if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
            return int(getattr(usage, "prompt_tokens", 0)), int(getattr(usage, "completion_tokens", 0))
        # object with .usage field
        if hasattr(usage, "usage"):
            inner = getattr(usage, "usage")
            if hasattr(inner, "prompt_tokens"):
                return int(getattr(inner, "prompt_tokens", 0)), int(getattr(inner, "completion_tokens", 0))
        # dict
        if isinstance(usage, dict):
            return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        return 0, 0

    # --------- artifact helpers ----------
    def log_artifact_df(self, df: pd.DataFrame, path: str):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        mlflow.log_text(buf.getvalue(), path)

    def log_artifact_json(self, obj, path: str, pretty=True):
        txt = json.dumps(obj, ensure_ascii=False, indent=(2 if pretty else None))
        mlflow.log_text(txt, path)

    def log_artifact_text(self, text: str, path: str):
        mlflow.log_text(text, path)
    
    # --------- rollup per-language summaries to parent ----------

    def add_child_summary(self, language: str, summary: dict):
        # parent only keeps this; child calls this via parent reference
        self._child_summaries[language] = summary

     # ---------- lifecycle ----------

    def start(self, nested: bool = False):
        mlflow.set_experiment(self.experiment_name)

        # Run name: request row id + optional language suffix
        loc_type = self.request.get("LocType")
        game = self.request.get("Game")
        url = self.request.get("URL")
        row_fingerprint = self.request.get("RowFingerprint") or self.request.get("RequestID", "run")
        base_name = f"{loc_type}:{game}:{row_fingerprint}"
        run_name = f"{base_name}:{self.language}" if self.language else base_name

        self.run = mlflow.start_run(run_name=run_name, nested=nested)

        # Tags (request-level + language-level)
        tags = {
            "RowFingerprint": _to_str(row_fingerprint),
            "LocType": _to_str(loc_type),
            "Game": _to_str(game),
            "URL": _to_str(url),
            "status": "RUNNING",
        }
        if self.language:
            tags["Language"] = self.language
            run_type = "child"
        else:
            run_type = "parent"

        tags["RunType"] = run_type

        mlflow.set_tags(tags)

        # Params
        if not self.language:
            # parent: log the list of languages once
            langs = self._parse_langs(self.request.get("TargetLanguages"))
            mlflow.log_params({
                "TargetLanguages": ",".join(langs),
                "RunType":run_type,
                "QAFlag": str(self.request.get("QAFlag", False)),
            })
        else:
            # child run: log just this language
            mlflow.log_params({"Language":_to_str(self.language),"RunType":run_type})

        return self.run.info.run_id

    def end(self, succeeded: bool, err_text: Optional[str]= None):
        mlflow.set_tag("status", "SUCCEEDED" if succeeded else "FAILED")
        if self._events:
            mlflow.log_text("\n".join(self._events), "logs/events.txt")
        if err_text:
            mlflow.log_text(err_text, "logs/error.txt")

        if not self.language and self._child_summaries:
            self.log_artifact_json(self._child_summaries, "summaries/per_language.json")

        mlflow.end_run()

    def event(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tag("last_event", msg)
        self._events.append(f"[{ts}] {msg}")

    
    def dict(self, obj, path: str):
        mlflow.log_dict(obj, path)

    def text(self, txt: str, path: str):
        mlflow.log_text(txt, path)

    def metrics(self, d: dict, step: Optional[int]= None):
        mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)

    def params(self, d: dict):
        mlflow.log_params({k: str(v) for k, v in d.items()})

    @contextmanager
    def step(self, name: str):
        start = time.time()
        self.event(f"Step start: {name}")
        try:
            yield
        finally:
            dur = time.time() - start
            self.metrics({f"duration_sec.{name}": dur})

    @contextmanager
    def nested(self, 
               name: str, 
               tags: Optional[Dict[str, Any]] = None,
               params: Optional[Dict[str, Any]] = None,):
        # You can still do sub-steps inside a language if you want (e.g., per-batch)
        with mlflow.start_run(run_name=name, nested=True):
            if tags: mlflow.set_tags(tags)
            if params: self.params(params)
            yield
            
    # Nice sugar for per-language runs
    @contextmanager
    def child(self, language: str):
        child = MLTracker(self.request, language=language, experiment_name=self.experiment_name)
        child.start(nested=True)
        try:
            yield child
        except Exception as e:
            child.end(False, str(e))
            raise
        else:
            child.end(True)