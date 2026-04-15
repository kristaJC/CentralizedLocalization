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
from general_config import *


class GenericLocalizer(LocalizationRun):

    def __init__(self,
                 request,
                 gsheet_client=None,
                 gpt_client=None,
                 cfg=None,
                 tracker: MLTracker | None = None):

        super().__init__(request, gsheet_client, gpt_client, cfg, tracker)

        self.required_tabs = self.cfg.get("input", {}).get("required_tabs", [])
        self.char_limit_policy = self.cfg.get("char_limit_policy", "")
        self.char_limit_column = "char_limit"

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
        df = pd.DataFrame(data, columns=self.input_headers)

        if "char_limit" in self.input_headers:
            self.char_limit_policy = "strict"
        else:
            df['char_limit'] = ""

        self.df = ensure_ids(df)
        self.data = self.df.values

        return self.df.values

    def _get_languages(self):
        target_langs_str = self.request.get("TargetLanguages")
        self.languages = [lang.strip() for lang in target_langs_str.split(",")]

        missing = [lang for lang in self.languages if lang not in ALL_LANGUAGES]
        if missing:
            raise ValueError(f"Unknown language(s): {missing}. Available: {list(ALL_LANGUAGES.keys())}")

        self.lang_map = {lang: ALL_LANGUAGES[lang] for lang in self.languages}
        self.lang_cds = list(self.lang_map.values())

    def _get_examples(self):
        ex = self.df.iloc[0].to_dict()
        self.ex_input = json.dumps(ex)
        self.context_infer = "The english phrase to translate is indicated by 'en_US'. All other keys in the dict for each row are context and should be used to help inform the translation."

        if self.char_limit_policy == 'strict':
            self.context_infer += " Be careful not to exceed the character limit for the translated sentence. The character limit is denoted in the 'char_limit' field and MUST be respected."

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

        if game not in self.general_game_specific_guidelines:
            raise ValueError(f"Game '{game}' not found in GENERAL_GAME_SPECIFIC_GUIDELINES. Available: {list(self.general_game_specific_guidelines.keys())}")

        self.game_description = self.general_game_specific_guidelines[game]

    def _generate_prompt_helper(self, language: str, game: str, prepped: str) -> List[Dict[str, Any]]:
        self._get_examples()
        self._get_languages()

        n = len(self.df)
        lang_cd = self.lang_map[language]

        base = f"""
            You are a professional game localizer translating for a popular mobile puzzle game called {self.game} by Jam City which is described as:
            {self.game_description}

            Translate English 'en_US' to {language}. Keep tone natural, playful, and mobile-friendly.
            Avoid overly formal or robotic phrasing.

            Follow these guidance notes:
            • {self.context_infer}

            HARD RULES:
            1) Output EXACTLY {n} items, in the SAME ORDER as input.
            2) Copy 'row_idx' unchanged for each item. Never invent, skip, or reorder items.
            3) Write the translation in field '{lang_cd}'.
            4) If 'char_limit' is present, final translation MUST be ≤ char_limit characters (spaces/punctuation count).
            5) Preserve placeholders EXACTLY: any substring wrapped in <...> or {{...}} in en_US must appear unchanged in the translation, with the SAME count.

            If present, use the additional context columns to guide tone, word choice, or brevity — especially when the English phrase is vague or could be interpreted multiple ways.
            Example Inputs as a json string:
            json
                {self.ex_input}

            You MUST follow these language specific guidelines:
            {self.lang_specific_guidelines[language]}

            Return JSON ONLY in this exact shape:
            [
            {{"row_idx":"...","en_US":"...","{lang_cd}":"...","len":123,"ok":true}},
            ...
            ]

            Notes:
            • 'len' is the character count of '{lang_cd}'.
            • 'ok' must be true iff all HARD RULES are satisfied for that item.
            • Do not add commentary, explanations, or extra keys.
            """.strip()

        return [
            {"role": "system", "content": base},
            {"role": "user",   "content": prepped}
        ]

    def build_prompts(self, prepped: str) -> Dict[str, List[Dict[str, Any]]]:
        self._get_languages()
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

    def qc_checks(self, formatted: Any) -> Dict[str, Any]:
        if not isinstance(formatted, pd.DataFrame):
            return {"ok": True, "issues": [], "stats": {}}

        policy = (self.cfg or {}).get("char_limit_policy", "").lower()
        if policy != "strict":
            return {"ok": True, "issues": [], "stats": {}}

        df = formatted.copy()
        if "translation" not in df.columns or "target_char_limit" not in df.columns:
            return {"ok": True, "issues": [], "stats": {}}

        df["translation_len"] = df["translation"].fillna("").map(lambda s: len(str(s)))
        over = df[df["translation_len"] > df["target_char_limit"].astype(int)]
        missing = df[df["translation_len"] == 0]

        issues: List[Dict[str, Any]] = []
        for _, r in over.iterrows():
            issues.append(r.to_dict())
        for _, r in missing.iterrows():
            issues.append(r.to_dict())

        return {
            "ok": (len(issues) == 0),
            "issues": issues,
            "stats": {"overlimit": int(len(issues))},
        }

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
        self.results = results.fillna("")

        wksht = self.sh.worksheet("output")
        headers = self.results.columns.tolist()
        out_data = self.results.values.tolist()

        letter_range = col_letter(len(headers))
        data_range = f"A2:{letter_range}{len(out_data)+1}"

        wksht.batch_update([
            {'range': f"A1:{letter_range}1", 'values': [headers]},
            {'range': data_range, 'values': out_data}
        ])
        return "Done!"
