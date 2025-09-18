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

from base_localizer import LocalizationRun
from ml_tracker import MLTracker

#from marketing_config import * 
from general_config import *


### TODO: MAKE SURE ANY JOINS FOR OUTPUT JOIN ON 'row_idx' ONLY. Then select only the out columns you want to write. 

EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"



import hashlib
import pandas as pd

def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "row_idx" not in out.columns:
        # stable, compact id
        out["row_idx"] = [f"r{i}" for i in range(len(out))]
    # optional: hash of the source string to guard against drift
    out["src_hash8"] = out["en_US"].fillna("").map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:8])
    return out


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
        self.char_limit_column = None

        #Get game relevant context set up
        self._get_game_context()

    def validate_inputs(self):
        
        # Open Sheet
        try:
            self.sh = self.gc.open_by_url(self.request.get("URL"))
        except Exception as e:
            raise Exception(f"Error opening google sheet: {e}")
        
        #Open Tab
        try:
            self.wksht = self.sh.worksheet("input")
        except Exception as e:
            raise Exception(f"Error opening input tab: {e}")
        
        # Check if all required tabs are present
        wkshts = self.sh.worksheets()
        for tab in self.required_tabs:
            if tab not in [wksht.title for wksht in wkshts]:
                self.sh.add_worksheet(tab,rows=200, cols = 50)
                # with expected header row
    
        return
    
    def load_inputs(self):

        self.validate_inputs()

        data = self.wksht.get_all_values()
        self.input_headers = data.pop(0)
        #self.data = data
        df = pd.DataFrame(data, columns=self.input_headers) #pandas DF

        if "char_limit" in self.input_headers:
            self.char_limit_policy = "strict"

        self.df = ensure_ids(df)
        self.data = self.df.values 

        return self.df.values
    
    def _get_languages(self):
        target_langs_str = self.request.get("TargetLanguages")
        self.languages = [lang.strip() for lang in target_langs_str.split(",")]

        lang_map = {}
        for lang in self.languages:
            lang_map[lang] = ALL_LANGUAGES[lang]
        
        self.lang_map = lang_map
        self.lang_cds = list(self.lang_map.values())

        return
  

    def _get_examples(self):
        ex = self.df.iloc[0].to_dict()
        self.ex_input = json.dumps(ex)
        self.context_infer = """The english phrase to translate is indicated by 'en_US'. All other keys in the dict for each row are context and should be used to help inform the translation."""

        if self.char_limit_policy=='strict':
            self.context_infer += """Be careful not to exceed the character limit for the translated sentence. The character limit is denoted in the 'char_limit' field MUST be respected."""

        return

    #TODO: Note, the data as input is redundant 
    def preprocess(self, data:List[str])->str: 
        PH_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
        """
        rows_df must have columns:
            - row_idx (stable id)
            - en_US (source)
            - optional: char_limit (int), context, any other hints
        """
        n = len(self.df)
    
        self.df["char_limit"] = pd.to_numeric(self.df["char_limit"], errors="coerce")
     
        # build compact payload the model needs (ordered!)
        payload = []
        for _, r in self.df.iterrows():
            en = r.get("en_US", "") or ""
            item = {
                "row_idx": r["row_idx"],
                "en_US": en,
                # keep payload tiny; include only relevant hints
            }
            # include char_limit only if it’s a finite number
            limit = r.get("char_limit")
            if pd.notna(limit):
                item["char_limit"] = int(limit)
            # include char_limit if present
            #if "char_limit" in self.df.columns and pd.notna(r["char_limit"]):
            #    item["char_limit"] = int(r["char_limit"])
            # optional per-row context
            if "context" in self.df.columns and r.get("context"):
                item["context"] = str(r["context"])
            # placeholders list (helps the model self-check)
            ph = PH_RE.findall(en)
            if ph:
                item["placeholders"] = ph
            # tiny checksum for alignment debugging
            if "src_hash8" in self.df.columns and r.get("src_hash8"):
                item["src_hash8"] = r["src_hash8"]

            payload.append(item)

        self.prepped = json.dumps(payload)

        return self.prepped

    def _get_game_context(self):
        """ Helper function to get relevant context for in game localization for particular game """
        game = self.request.get('Game')
        self.game = game
        self.lang_specific_guidelines = GENERAL_LANG_SPECIFIC_GUIDELINES
        self.general_game_specific_guidelines = GENERAL_GAME_SPECIFIC_GUIDELINES
        self.game_description = self.general_game_specific_guidelines[game]
        # All the other stuff was done in 


    def _generate_prompt_helper(self, 
                                language:str, 
                                game:str, 
                                prepped:str)->List[Dict[str, Any]]:
        
        
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


        '''
        base_old = f""" 
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
            {{ "en_US": "english phrase 1", "{lang_cd}": "translated phrase 1" }},
            {{ "en_US": "english phrase 2", "{lang_cd}": "translated phrase 2" }},
            ...
            ]\n\n
            """
        '''
        return [
            {"role": "system", "content": base},
            {"role": "user",   "content": prepped}
        ]

    def build_prompts(self, prepped:str)->Dict[str,List[Dict[str, Any]]]: 
        #self._get_game_contex()

        self._get_languages()
        prompts = []
        self.prepped = prepped

        # TODO: Make sure "langauges" is appropriately passed here
        for lang in self.languages:
           prompt = self._generate_prompt_helper(lang, self.game, self.prepped)
           prompts.append(prompt)
        
        self.prompts = prompts
        self.groups = dict(zip(self.languages, prompts))

        return self.groups

    #return ->Tuple[str, Dict[str, int]]
    def _call_model_batch(self, prompt:str):
        """
        Must return (outputs, usage_dict) where usage_dict includes:
          {'prompt_tokens': int, 'completion_tokens': int}
        """
        MODEL = "gpt-4o"
        temperature = 0.05

        response = self.gpt.chat.completions.create(
                model=MODEL, 
                messages=prompt,
                temperature=0.00  # adjust for creativity vs. stability
        )

        output = response.choices[0].message.content
        usage = response.usage
        return output, usage

    '''
    def _parse_model_json_block(self, raw_output:str)->Dict[str,Any]:
        """
        Cleans and parses a JSON-like string from a model output wrapped in markdown code block.
        
        Args:
            raw_output (str): The raw output string, e.g., from GPT, wrapped with ```json ... ```
        
        Returns:
            list[dict]: Parsed JSON content as Python list of dictionaries.
            
        Raises:
            ValueError: If the cleaned string cannot be parsed as valid JSON.
        """
        try:
            # Strip markdown-style code block markers and leading/trailing whitespace
            cleaned = re.sub(r"^```json|```$", "", raw_output.strip(), flags=re.IGNORECASE).strip()

            # Replace escaped newlines (if necessary) and extra leading/trailing junk
            cleaned = cleaned.replace("\\n", "").replace("\n", "").strip()

            # Now parse
            loaded = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON: {e}")

        if isinstance(loaded, str):
            try:
                return json.loads(loaded)
            except:
                raise ValueError(f"Could not parse JSON: {e}")
        else:
            return loaded'''


    ### TODO 
    def qc_checks(self, formatted: Any) -> Dict[str, Any]:
        """
        formatted: expected to be a pandas DataFrame in long format with:
          ['row_idx','language','language_cd','platform','translation','target_char_limit', ...]
        """
        if not isinstance(formatted, pd.DataFrame):
            return {"ok": True, "issues": [], "stats": {}}

        policy = (self.cfg or {}).get("char_limit_policy", "").lower()
        strict = (policy == "strict")

        if not strict:
            # You could add other policies later. For now, only strict is meaningful.
            return {"ok": True, "issues": [], "stats": {}}

        # compute lengths
        df = formatted.copy()
        if "translation" not in df.columns or "target_char_limit" not in df.columns:
            # If absent, just pass
            return {"ok": True, "issues": [], "stats": {}}

        ## Check if over the char limit
        df["translation_len"] = df["translation"].fillna("").map(lambda s: len(str(s)))
        over = df[df["translation_len"] > df["target_char_limit"].astype(int)]

        ### Check if miissing 
        missing = df[df["translation_len"]==0]


        issues: List[Dict[str, Any]] = []
        for _, r in over.iterrows():
            issues.append(r.to_dict())
                
        #### THIS FORMATTING IS LIKELY WRONG
        """
        #"char_limit": int(r["char_limit"]),
                #"current_len": int(r["translation_len"]),
                #"en_text": r.get("en_US"),  # if present in your long DF
                #"current_translation": r["translation"],
        #   })
      
        issues.append({
                "char_limit": int(r["char_limit"]),
                "current_len": int(r["translation_len"]),
                "en_text": r.get("en_US"),  # if present in your long DF
                "current_translation": r["translation"],
        """
        for _, r in missing.iterrows():
            issues.append(r.to_dict())

        return {
            "ok": (len(issues) == 0),
            "issues": issues,
            "stats": {"overlimit": int(len(issues))},
        }
    ###TODO: QC_REPAIR


    def postprocess(self, 
                    outputs:str)->List[pd.DataFrame]: 
        
        ##TODO: Remove extraneous columns... we only want row_idx, <language_cd>

        postprocessed_dfs = []
        for output in outputs:
            parsed = self._parse_model_json_block(output)
            returned_df = pd.DataFrame(parsed)
            postprocessed_dfs.append(returned_df)

        self.postprocessed_dfs = postprocessed_dfs 

        return self.postprocessed_dfs

    #Helper function for write_outputs
    def _merge_outputs_by_language(self, 
                                   post: List[pd.DataFrame])->pd.DataFrame:
        
        #Consider doing this for different cases
        # eg.drop_cols set at init : ['context'] for panda pop
        # eg.join_cols set at init: ['token']

        ### join cols should by by index at this point... or change the output

        processed_groups = dict(zip(self.lang_cds, post))

        ### Make sure this drops off any extra fields.... we want this all clean like, en_US + languages 
        results = self.df.copy()
        for lang,df in processed_groups.items():
            results = results.merge(df[['row_idx',lang]],on = ["row_idx"], how='left')
        
        select_cols = ['en_US',*self.lang_cds]
        if self.char_limit_policy=='strict':
            select_cols = ['en_US','char_limit',*self.lang_cds]
            
        self.results = results[select_cols].fillna({'char_limit':""})

        return self.results
    
    #def _finalize_status_tracking(self):
        #self.tracker.overall_status = "Succeeded"
        ##update tracking sheet 
        #sh = self.gc.open_by_url(TRACKING_SHEET_URL)
        #data = wksht.worksheet("Tracking").get_all_values()
        #header,values = data[0],data[1:]
        #df = pd.DataFrame(values,columns=header)

        ## Find the row, and update the status, based on row fingerprint, and status
        ###TODO
        ## row= sh.find()
        ## col = #should be established
        #self.sh.update("",self.tracker.overall_status)

        ## Send email
        ## self.gc.email??

        ## Send Slack message?

        ## 

    def write_outputs(self, post:List[pd.DataFrame])->str: 

        results = self._merge_outputs_by_language(post)
        self.results = results.fillna("")

        ### flag for recheck for any missing translations!!!
        
        wksht = self.sh.worksheet("output")

        headers = self.results.columns.tolist()
        out_data = self.results.values.tolist()
        
        
        number_to_letter = {
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
            "5": "E",
            "6": "F",
            "7": "G",
            "8": "H",
            "9": "I",
            "10": "J",
            "11": "K",
            "12": "L",
            "13": "M",
            "14": "N",
            "15": "O",
            "16": "P",
            "17": "Q",
            "18": "R",
            "19": "S",
            "20": "T",
            "21": "U",
            "22": "V",
            "23": "W",
            "24": "X",
            "25": "Y",
            "26": "Z"
        }
        letter_range = number_to_letter[str(len(headers))]
        data_range = f"A2:{letter_range}{len(out_data)+1}"


        wksht.batch_update([{
                        'range':f"A1:{letter_range}1", 'values':[headers]
                        },
                        {
                        'range':data_range, 'values':out_data
                        }])

        return "Done!"

