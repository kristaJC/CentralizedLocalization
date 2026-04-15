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
from dmm_humourous_guidelines import *

from general_config import *

import hashlib
import pandas as pd

from context_enrichment import *
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "row_idx" not in out.columns:
        # stable, compact id
        out["row_idx"] = [f"r{i}" for i in range(len(out))]
    # optional: hash of the source string to guard against drift
    out["src_hash8"] = out["en_US"].fillna("").map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:8])
    return out



class DMMInGameLocalizer(LocalizationRun):

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
        #self.df = pd.DataFrame(data, columns=self.input_headers) 
        #self.df = ensure_ids(self.df)
        

        ## Filter for context enrichment
        df = spark.createDataFrame(data, schema = self.input_headers)
        self.item_desc = df.where("KEY like '%ITEM_DESC_%'").toPandas()
        self.item_desc = ensure_ids(self.item_desc)
        
        self.df = df.where("KEY not like '%ITEM_DESC_%'").toPandas()
        self.df = ensure_ids(self.df)

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

        payload_item_desc = []
        for _, r in self.item_desc.iterrows():
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
            if "src_hash8" in self.item_desc.columns and r.get("src_hash8"):
                item["src_hash8"] = r["src_hash8"]

            payload_item_desc.append(item)

        self.other_cols = other_cols
        self.prepped = json.dumps(payload)
        

        self.prepped_item_desc = json.dumps(payload_item_desc)
        #self.preprocessed_enriched = self._preprocess_for_humorous(self.item_desc)

        ## Convert data to slug....
        #prepped = json.dumps(self.df.to_dict(orient='records'))

        return self.prepped, self.prepped_item_desc


    def _preprocess_for_humorous(self, df):

        prompt = preprocess_item_desc_DMM(df)
        enriched, token = self._call_model_batch(prompt)
        parsed_enriched = self._parse_json_model_block(enriched)
        self.preprocessed_enriched = write_preprocessed(parsed_enriched)

        payload = []
        for _, r in self.df.iterrows():
            en = r.get("en_US", "") or ""

            item = {
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
            if "src_hash8" in self.item_desc.columns and r.get("src_hash8"):
                item["src_hash8"] = r["src_hash8"]

            payload_item_desc.append(item)

        return preprocessed_enriched

    def _get_game_context(self, game = "Disney Magic Match"):

        """ Helper function to get relevant context for in game localization for particular game """
        game = self.request.get('Game')
        self.game = game
        self.lang_specific_guidelines = GENERAL_LANG_SPECIFIC_GUIDELINES
        self.general_game_specific_guidelines = GENERAL_GAME_SPECIFIC_GUIDELINES

        #if game not in ["Panda Pop","Cookie Jam Blast","Genies & Gems"]:
        #    raise Exception(f"Game {game} not supported")

        # Specifics for games
     
        self.game_description = self.general_game_specific_guidelines[game]
        self.lang_map = DMM_LANG_MAP
        self.languages = list(self.lang_map.keys())
        self.lang_cds = list(self.lang_map.values())

        #self.humorous_lang_guide = HUMOROUS_DMM_LANG_GUIDE

    def _generate_payload_humourous(self, prerocessed):
        payload = []
        for _, r in preprocessed.iterrows():
            #en = r.get("en_US", "") or ""
            key = r.get("key")
            item = {
                'key':key,
                'object': r.get("object",""),
                'descriptor': r.get("descriptor",""),
                'additional_context': r.get("additional_context",""),
                'simple_description_en': r.get("simple_description_en",""),
                # keep payload tiny; include only relevant hints
            }

            payload.append(item)


        prepped_payload = json.dumps(payload)
        return prepped_payload

        
    def _generate_humorous_prompt_helper(self,language:str, 
                                game:str, 
                                prepped:str, 
                                humor_mode:str,
                                )->List[Dict[str, Any]]:
        preprocessed_enriched = self._preprocess_for_humorous(prepped)

        if humor_mode == "object_quip":
            base = QUIP_PROMPT
            base += QUIP_LANGUAGE_GUIDE[language] #TODO, check how this is keeyed
            base += OUTPUT_FORMAT_QUIP
        elif humor_mode == "fun_translation":
            base = FUN_DESCRIPTION_PROMPT
            base += FUN_DESCRIPTIONS_LANGUAGE_GUIDE[language]
            base += OUTPUT_FORMAT_FUN_DESCRIPTION
        else:
            raise Exception(f"Mode {humor_mode} not supported")

        ### prepped needs to be preprocesed_enriched
        preprocessed_payload = self._generate_payload_humourous(preprocessed_enriched)
        return [
            {"role": "system", "content": base},
            {"role": "user",   "content": preprocessed_payload}
        ]



    def _generate_prompt_helper(self, 
                                language:str, 
                                game:str, 
                                prepped:str, 
                                )->List[Dict[str, Any]]:

        base = f""" 
            You are a professional game localizer translating for a popular mobile puzzle game called {self.game} by Jam City which is described as:
            {self.game_description}
            Please translate the in-game phrases provided below from English into {language}.
                •   Keep the translations natural, playful, and appropriate for a casual mobile gaming tone.
                •   Avoid overly formal or mechanical language.
                •   There is no strict character limit, but translations should not be egregiously longer than the original English text.
            If present, use the context to guide tone, word choice, or brevity — especially when the English phrase is vague or could be interpreted multiple ways.
            
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

        prompts = []
        self.prepped = prepped

        # TODO: Make sure "langauges" is appropriately passed here
        for lang in self.languages:
           prompt = self._generate_prompt_helper(lang, self.game, self.prepped)
           prompts.append(prompt)
        
        self.prompts = prompts
        self.groups = dict(zip(self.languages, prompts))

        return self.groups
    
    def build_prompts_humorous(self, prepped:str, humor_mode:str)->Dict[str,List[Dict[str, Any]]]: 

        prompts = []
        self.prepped = prepped

        # TODO: Make sure "langauges" is appropriately passed here
        for lang in self.languages:
           prompt = self._generate_humorous_prompt_helper(lang, self.game, self.preprocessed_enriched, humor_mode = humor_mode)
           prompts.append(prompt)
        
        self.prompts_humourous = prompts
        self.groups_humorous = dict(zip(self.languages, prompts))

        return self.groups

    #TODO: call model batch separately for diff humor modes
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

        return results
    
    def _merge_outputs_by_language_humorous(self, 
                                   post: List[pd.DataFrame])->pd.DataFrame:
        

        processed_groups = dict(zip(self.lang_cds, post))

        ### Make sure this drops off any extra fields.... we want this all clean like, en_US + languages 
        results = self.prepped_item_desc.copy()
        for lang,df in processed_groups.items():
            results = results.merge(df[['row_idx',lang]],on = ["row_idx"], how='left')
        
        select_cols = ['en_US',*self.other_cols, *self.lang_cds]
        if self.char_limit_policy=='strict':
            select_cols = ['en_US',*self.other_cols,*self.lang_cds]
            
        results = results[select_cols]#.fillna({'char_limit':""})

        return results



    def write_outputs(self, post:List[pd.DataFrame], tab_name='output')->str: 
        
        if tab_name == "output":
            results = self._merge_outputs_by_language(post)
            self.results = results
        else:
            results = self._merge_outputs_by_language_humorous(post)
            #self.results = results
    
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

        try:
            self.sh.worksheet(tab_name, rows =len(out_data)+1, cols = len(headers))
        except:
            pass

        wksht = self.sh.worksheet(tab_name)
        wksht.clear()

        wksht.batch_update([{'range':headers_range, 'values':[headers]},
                            {'range':data_range, 'values':out_data}])

        return "Done!"

