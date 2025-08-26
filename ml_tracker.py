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
from unidecode import unidecode
from pathlib import Path
import pyspark
from pyspark.sql.functions import *
from functools import reduce
from typing import *
import tiktoken

from general_config import *
EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


#TODO: Add artifact logging!! Like the results dataframe

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

    @staticmethod
    def _parse_langs(raw_langs: Optional[str]):
        if not raw_langs:
            return []
        parts = [x.strip() for x in raw_langs.replace(";", ",").split(",")]
        return [x for x in parts if x]

    def start(self, nested: bool = False):
        mlflow.set_experiment(self.experiment_name)

        # Run name: request row id + optional language suffix
        base_name = self.request.get("Game")
        run_id = self.request.get("RowFingerprint") or self.request.get("RequestID", "run")
        base_name = f"{base_name}:{run_id}"
        run_name = f"{base_name}:{self.language}" if self.language else base_name

        self.run = mlflow.start_run(run_name=run_name, nested=nested)

        # Tags (request-level + language-level)
        tags = {
            **self.request,
            "status": "RUNNING",
        }
        if self.language:
            tags["Language"] = self.language
        mlflow.set_tags(tags)

        # Params
        if not self.language:
            # parent: log the list of languages once
            langs = self._parse_langs(self.request.get("TargetLanguages"))
            mlflow.log_params({
                "TargetLanguages": ",".join(langs),
                #"QAFlag": str(self.request.get("QAFlag", False)),
            })
        else:
            # child run: log just this language
            mlflow.log_param("Language", self.language)

        return self.run.info.run_id

    def end(self, succeeded: bool, err_text: Optional[str]= None):
        mlflow.set_tag("status", "SUCCEEDED" if succeeded else "FAILED")
        if self._events:
            mlflow.log_text("\n".join(self._events), "logs/events.txt")
        if err_text:
            mlflow.log_text(err_text, "logs/error.txt")
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
