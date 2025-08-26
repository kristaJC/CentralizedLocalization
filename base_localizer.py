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
from ml_tracker import MLTracker

EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


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
                self.tracker.dict({"preview": str(data)[:2000]}, "snapshots/input_preview.json")

            with self.tracker.step("preprocess"):
                prepped = self.preprocess(data)
                self.tracker.metrics({"rows.prepped": len(prepped)})

            with self.tracker.step("build_prompts"):
                prompts = self.build_prompts(prepped)
                self.tracker.metrics({"prompts.count": len(prompts)})

            # Build per-language trackers (child runs) using request languages
            langs = MLTracker._parse_langs(self.request.get("TargetLanguages"))
            # TODO:
            # This could be remedied by just using self.languages instad of self.request.get("TargetLanguages") and parsing
            for lang in langs:
                self.lang_trackers[lang] = MLTracker(
                    request=self.request,
                    language=lang,
                    experiment_name=self.tracker.experiment_name  # same experiment
                )

            outputs = self.translate(prompts)  # tracked per language below

            with self.tracker.step("postprocess"):
                final_rows = self.postprocess(outputs)
                self.tracker.metrics({"rows.final": len(final_rows)})

            with self.tracker.step("write_outputs"):
                self.write_outputs(final_rows)

            self.tracker.end(succeeded=True)
            return {"status": "SUCCEEDED", "run_id": parent_run_id}
        except Exception as e:
            self.tracker.end(succeeded=False, err_text=str(e))
            raise

    # Shared, tracked translate that spins one child run per language
    def translate(self, groups):
        parent = self.tracker
        #batch_size = int(batch_size or self.cfg.get("batch_size", 50))

        total_prompt_tokens = 0
        total_completion_tokens = 0
        results = []

        with parent.step("translate"):
            # TODO: we actually already have them batched in a preprocessing step
            #groups = self._group_prompts_for_translation(prompt_batch) # remove!!!
            #groups = self.groups 
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

                    """
                    ##TODO: We dont need to loop by batch, only by language
                    for i in range(0, len(prompts), batch_size):
                        batch = prompts[i:i+batch_size]
                        with lang_tracker.step("api_batch"):
                            out, usage = self._call_model_batch(batch)  # subclass hook
                            results.extend(out)

                            p = (usage or {}).get("prompt_tokens", 0)
                            c = (usage or {}).get("completion_tokens", 0)
                            lang_total_p += p
                            lang_total_c += c

                            lang_tracker.metrics({
                                "items.batch": len(batch),
                                "tokens.prompt.batch": p,
                                "tokens.completion.batch": c,
                            })

                    # per-language rollup
                    lang_tracker.metrics({
                        "items.total": len(prompts),
                        "tokens.prompt.total": lang_total_p,
                        "tokens.completion.total": lang_total_c,
                    })
                    # accumulate into parent
                    total_prompt_tokens += lang_total_p
                    total_completion_tokens += lang_total_c

                    lang_tracker.end(succeeded=True)"""

                except Exception as e:
                    lang_tracker.end(succeeded=False, err_text=str(e))
                    raise

            # parent rollup
            parent.metrics({
                "tokens.prompt.total": total_prompt_tokens,
                "tokens.completion.total": total_completion_tokens,
            })

        return results
    
    """
    # Actually TODO: Lets alter this
    # Default grouping: by 'lang' key if present
    #def _group_prompts_for_translation(self, prompts):

        ###REMOVE THIS! It's already prepared before now!!
    #    groups = {}
    #    for p in prompts:
    #        key = p.get("lang", "default")
    #        groups.setdefault(key, []).append(p)
    """

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
