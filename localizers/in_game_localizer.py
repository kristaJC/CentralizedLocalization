import time
import mlflow
import json
import pandas as pd
import re
import openai
from openai import OpenAI
import gspread
import os
from pathlib import Path
import pyspark
from pyspark.sql.functions import *
from functools import reduce
from typing import *
import tiktoken

from base_localizer import LocalizationRun, ensure_ids
from ml_tracker import MLTracker
from in_game_config import *
from general_config import *


class InGameLocalizer(LocalizationRun):

    def __init__(self,
                 request,
                 gsheet_client=None,
                 gpt_client=None,
                 cfg=None,
                 tracker: MLTracker | None = None):

        super().__init__(request, gsheet_client, gpt_client, cfg, tracker)

        self.required_tabs = self.cfg.get("input", {}).get("required_tabs", [])
        self.char_limit_policy = self.cfg.get("char_limit_policy", "")

        self._get_game_context()

    def validate_inputs(self):

        try:
            self.sh = self.gc.open_by_url(self.request.get("URL"))
        except Exception as e:
            raise Exception(f"Error opening google sheet: {e}")

        try:
            self.wksht = self.sh.worksheet("input")
        except Exception as e:
            raise Exception(f"Error opening input tab: {e}")

        wkshts = self.sh.worksheets()
        for tab in self.required_tabs:
            if tab not in [wksht.title for wksht in wkshts]:
                self.sh.add_worksheet(tab, rows=200, cols=50)

    def load_inputs(self):

        data = self.wksht.get_all_values()
        self.input_headers = data.pop(0)
        self.data = data
        self.df = pd.DataFrame(data, columns=self.input_headers)
        self.df = ensure_ids(self.df)

        return self.data

    def preprocess(self, data: List[str]) -> str:
        PH_RE = re.compile(r"<[^>]+>|\{[^}]+\}")

        n = len(self.df)
        self.df["char_limit"] = pd.to_numeric(self.df["char_limit"], errors="coerce")

        other_cols = self.df.columns.tolist()
        for col_to_remove in ("row_idx", "en_US", "char_limit", "src_hash8"):
            try:
                other_cols.remove(col_to_remove)
            except ValueError:
                pass

        payload = []
        for _, r in self.df.iterrows():
            en = r.get("en_US", "") or ""
            item = {"row_idx": r["row_idx"], "en_US": en}

            limit = r.get("char_limit", "") or ""
            if pd.notna(limit):
                item["char_limit"] = int(limit)

            for col in other_cols:
                val = r.get(col)
                if pd.notna(val):
                    item[col] = str(val)

            ph = PH_RE.findall(en)
            if ph:
                item["placeholders"] = ph

            if "src_hash8" in self.df.columns and r.get("src_hash8"):
                item["src_hash8"] = r["src_hash8"]

            payload.append(item)

        self.other_cols = other_cols
        self.prepped = json.dumps(payload)
        return self.prepped

    def _get_game_context(self):
        game = self.request.get('Game')
        self.game = game
        self.lang_specific_guidelines = GENERAL_LANG_SPECIFIC_GUIDELINES
        self.general_game_specific_guidelines = GENERAL_GAME_SPECIFIC_GUIDELINES

        if game not in ["Panda Pop", "Cookie Jam Blast", "Genies & Gems"]:
            raise Exception(f"Game '{game}' not supported for InGame localization. Supported: Panda Pop, Cookie Jam Blast, Genies & Gems")

        if game == "Panda Pop":
            self.game_description = self.general_game_specific_guidelines[game]
            self.lang_map = PP_LANG_MAP
            self.languages = list(self.lang_map.keys())
            self.lang_cds = list(self.lang_map.values())
            self.ex_input = PP_EX_INPUT
            self.context_infer = PP_CONTEXT_INFER
            self.token_infer = PP_TOKEN_INFER

        if game == "Cookie Jam Blast":
            self.game_description = self.general_game_specific_guidelines[game]
            self.lang_map = CJB_LANG_MAP
            self.languages = list(self.lang_map.keys())
            self.lang_cds = list(self.lang_map.values())
            self.ex_input = CJB_EX_INPUT
            self.context_infer = CJB_CONTEXT_INFER
            self.token_infer = CJB_TOKEN_INFER

        if game == "Genies & Gems":
            self.game_description = self.general_game_specific_guidelines[game]
            self.lang_map = GG_LANG_MAP
            self.languages = list(self.lang_map.keys())
            self.lang_cds = list(self.lang_map.values())
            self.ex_input = GG_EX_INPUT
            self.context_infer = GG_CONTEXT_INFER
            self.token_infer = GG_TOKEN_INFER

    def _generate_prompt_helper(self, language: str, game: str, prepped: str) -> List[Dict[str, Any]]:

        base = f"""
            You are a professional game localizer translating for a popular mobile puzzle game called {self.game} by Jam City which is described as:
            {self.game_description}
            Please translate the in-game phrases provided below from English into {language}.
                •   Keep the translations natural, playful, and appropriate for a casual mobile gaming tone.
                •   Avoid overly formal or mechanical language.
                •   There is no strict character limit, but translations should not be egregiously longer than the original English text.
                •   {self.token_infer}
                •   {self.context_infer}
            If present, use the context to guide tone, word choice, or brevity — especially when the English phrase is vague or could be interpreted multiple ways.
            Example Inputs as a json string:
            json
                {self.ex_input}
            """
        base += f"""
            You MUST follow these language specific guidelines:
            {self.lang_specific_guidelines[language]}
            """

        lang_cd = self.lang_map[language]

        base += f"""
            Respond in **JSON format**, one object per row:
            json
            [
            {{ "en_US": "original phrase", "row_idx": row_idx, "{lang_cd}": "translated phrase 1" }},
            {{ "en_US": "original phrase", "row_idx": row_idx, "{lang_cd}": "translated phrase 2" }},
            ...
            ]]\n\n
            """

        return [
            {"role": "system", "content": base},
            {"role": "user",   "content": prepped}
        ]

    def build_prompts(self, prepped: str) -> Dict[str, List[Dict[str, Any]]]:
        prompts = []
        self.prepped = prepped

        for lang in self.languages:
            prompt = self._generate_prompt_helper(lang, self.game, self.prepped)
            prompts.append(prompt)

        self.prompts = prompts
        self.groups = dict(zip(self.languages, prompts))
        return self.groups

    def _call_model_batch(self, prompt: str):
        response = self.gpt.chat.completions.create(
            model=MODEL,
            messages=prompt,
            temperature=TEMP
        )
        return response.choices[0].message.content, response.usage

    def postprocess(self, outputs: str) -> List[pd.DataFrame]:
        postprocessed_dfs = []
        for output in outputs:
            parsed = self._parse_model_json_block(output)
            returned_df = pd.DataFrame(parsed)
            postprocessed_dfs.append(returned_df)
        self.postprocessed_dfs = postprocessed_dfs
        return self.postprocessed_dfs

    def _merge_outputs_by_language(self, post: List[pd.DataFrame]) -> pd.DataFrame:
        processed_groups = dict(zip(self.lang_cds, post))

        results = self.df.copy()
        for lang, df in processed_groups.items():
            results = results.merge(df[['row_idx', lang]], on=["row_idx"], how='left')

        select_cols = ['en_US', *self.other_cols, *self.lang_cds]
        if self.char_limit_policy == 'strict':
            select_cols = ['en_US', *self.other_cols, 'char_limit', *self.lang_cds]

        self.results = results[select_cols].fillna({'char_limit': ""})
        return self.results

    def write_outputs(self, post: List[pd.DataFrame]) -> str:
        results = self._merge_outputs_by_language(post)
        self.results = results

        wksht = self.sh.worksheet("output")
        headers = results.columns.tolist()
        out_data = results.values.tolist()

        letter_range = col_letter(len(headers))
        headers_range = f"A1:{letter_range}1"
        data_range = f"A2:{letter_range}{len(out_data)+1}"

        wksht.clear()
        wksht.batch_update([
            {'range': headers_range, 'values': [headers]},
            {'range': data_range, 'values': out_data}
        ])
        return "Done!"
