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
from in_game_config import * 

from general_config import *

import hashlib
import pandas as pd

EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "row_idx" not in out.columns:
        # stable, compact id
        out["row_idx"] = [f"r{i}" for i in range(len(out))]
    # optional: hash of the source string to guard against drift
    out["src_hash8"] = out["en_US"].fillna("").map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:8])
    return out



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

        data = self.wksht.get_all_values()
        self.input_headers = data.pop(0)
        self.data = data
        self.df = pd.DataFrame(data, columns=self.input_headers) 
        self.df = ensure_ids(self.df)
        

        #pandas DF

        return self.data
    
    #TODO: Note, the data as input is redundant 
    #### Maybe make this more flexible like marketing... add a character limit
    def preprocess(self, data:List[str])->str: 

        PH_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
        """
        rows_df must have columns:
            - row_idx (stable id)
            - en_US (source)
            - char_limit (int)
            - optional: additional columns
        """
        n = len(self.df)
    
        self.df["char_limit"] = pd.to_numeric(self.df["char_limit"], errors="coerce")


        # Find the cols 
        other_cols = self.df.columns.tolist()
        other_cols.remove("row_idx")
        other_cols.remove("en_US")
        try:
            other_cols.remove('char_limit')
        except:
            pass
        try:
            other_cols.remove("src_hash8")
        except:
            pass
        
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
            limit = r.get("char_limit","") or ""
            if pd.notna(limit):
                item["char_limit"] = int(limit)
            
            ## Grab the other cols and convert to string
            for col in other_cols:
                val = r.get(col)
                if pd.notna(val):
                    item[col] = str(val)
                
                
            # placeholders list (helps the model self-check)
            ph = PH_RE.findall(en)
            if ph:
                item["placeholders"] = ph
            # tiny checksum for alignment debugging
            if "src_hash8" in self.df.columns and r.get("src_hash8"):
                item["src_hash8"] = r["src_hash8"]

            payload.append(item)

        self.other_cols = other_cols
        self.prepped = json.dumps(payload)


        ## Convert data to slug....
        #prepped = json.dumps(self.df.to_dict(orient='records'))

        return self.prepped

    
    def _get_game_context(self):

        """ Helper function to get relevant context for in game localization for particular game """
        game = self.request.get('Game')
        self.game = game
        self.lang_specific_guidelines = GENERAL_LANG_SPECIFIC_GUIDELINES
        self.general_game_specific_guidelines = GENERAL_GAME_SPECIFIC_GUIDELINES

        if game not in ["Panda Pop","Cookie Jam Blast","Genies & Gems"]:
            raise Exception(f"Game {game} not supported")

        # Specifics for games
        if game == "Panda Pop":
            self.game_description = self.general_game_specific_guidelines[game]
            self.lang_map = PP_LANG_MAP
            self.languages = list(self.lang_map.keys())
            self.lang_cds = list(self.lang_map.values())
        
            # game specific prompt inputs 
            self.ex_input = PP_EX_INPUT
            self.context_infer = PP_CONTEXT_INFER
            self.token_infer = PP_TOKEN_INFER
        
        if game == "Cookie Jam Blast":
            self.game_description = self.general_game_specific_guidelines[game]
            self.lang_map = CJB_LANG_MAP
            self.languages = list(self.lang_map.keys())
            self.lang_cds = list(self.lang_map.values())

            # game specific prompt inputs 
            self.ex_input = CJB_EX_INPUT
            self.context_infer = CJB_CONTEXT_INFER
            self.token_infer = CJB_TOKEN_INFER
            
        if game == "Genies & Gems":
            self.game_description = self.general_game_specific_guidelines[game]
            self.lang_map = GG_LANG_MAP
            self.languages = list(self.lang_map.keys())
            self.lang_cds = list(self.lang_map.values())

            # game specific prompt inputs 
            self.ex_input = GG_EX_INPUT
            self.context_infer = GG_CONTEXT_INFER
            self.token_infer = GG_TOKEN_INFER

        #self.languages = list(self.lang_map.keys())
        #self.lang_cds = list(self.lang_map.values())

    def _generate_prompt_helper(self, 
                                language:str, 
                                game:str, 
                                prepped:str)->List[Dict[str, Any]]:

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
            {{ "en_US": "original phrase","row_idx" : row_idx, "{lang_cd}": "translated phrase 2" }},
            ...
            ]\n\n
            """
    
        return [
            {"role": "system", "content": base},
            {"role": "user",   "content": prepped}
        ]

    def build_prompts(self, prepped:str)->Dict[str,List[Dict[str, Any]]]: 
        #self._get_game_contex()
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
                temperature=0.05  # adjust for creativity vs. stability
        )
    
        ### call GPT model for translation
        #GPT chat completions prompt
        #raw_results = self.gpt_model.

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
            return loaded
    '''

    def postprocess(self, 
                    outputs:str)->List[pd.DataFrame]: 
        
        postprocessed_dfs = []
        for output in outputs:
            parsed = self._parse_model_json_block(output)
            returned_df = pd.DataFrame(parsed)
            postprocessed_dfs.append(returned_df)

        self.postprocessed_dfs = postprocessed_dfs 

        return self.postprocessed_dfs

    def _merge_outputs_by_language(self, 
                                   post: List[pd.DataFrame])->pd.DataFrame:
        

        processed_groups = dict(zip(self.lang_cds, post))

        ### Make sure this drops off any extra fields.... we want this all clean like, en_US + languages 
        results = self.df.copy()
        for lang,df in processed_groups.items():
            results = results.merge(df[['row_idx',lang]],on = ["row_idx"], how='left')
        
        select_cols = ['en_US',*self.other_cols, *self.lang_cds]
        if self.char_limit_policy=='strict':
            select_cols = ['en_US',*self.other_cols,'char_limit',*self.lang_cds]
            
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
        self.results = results
        
        wksht = self.sh.worksheet("output")

        number_to_letter = {
            "1": "A",  "2": "B",  "3": "C",  "4": "D",  "5": "E",
            "6": "F",  "7": "G",  "8": "H",  "9": "I", "10": "J",
            "11": "K", "12": "L", "13": "M", "14": "N", "15": "O",
            "16": "P", "17": "Q", "18": "R", "19": "S", "20": "T",
            "21": "U", "22": "V", "23": "W", "24": "X", "25": "Y",
            "26": "Z", "27": "AA", "28": "AB", "29": "AC", "30": "AD",
            "31": "AE", "32": "AF", "33": "AG", "34": "AH", "35": "AI",
            "36": "AJ", "37": "AK", "38": "AL", "39": "AM", "40": "AN",
            "41": "AO", "42": "AP", "43": "AQ", "44": "AR", "45": "AS",
            "46": "AT", "47": "AU", "48": "AV", "49": "AW", "50": "AX",
            "51": "AY", "52": "AZ", "53": "BA", "54": "BB", "55": "BC",
            "56": "BD", "57": "BE", "58": "BF", "59": "BG", "60": "BH",
        }
        headers = results.columns.tolist()
        out_data = results.values.tolist()

        letter_range = number_to_letter[str(len(headers))]
        
        headers_range = f"A1:{letter_range}1"
        data_range = f"A2:{letter_range}{len(out_data)+1}"
        wksht.clear()

        wksht.batch_update([{'range':headers_range, 'values':[headers]},
                            {'range':data_range, 'values':out_data}])

        return "Done!"

