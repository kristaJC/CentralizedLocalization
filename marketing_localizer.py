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
#from InGame_Config import *
from marketing_config import * 

from general_config import *


EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


class MarketingLocalizer(LocalizationRun):

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
        self.df = pd.DataFrame(data, columns=self.input_headers) #pandas DF

        return self.data
    
    def _get_languages(self):
        target_langs_str = self.request.get("TargetLanguages")
        self.languages = [lang.strip() for lang in target_langs_str.split(",")]

        lang_map = {}
        for lang in self.languages:
            lang_map[lang] = ALL_LANGUAGES[lang]
        

        self.lang_map = lang_map
        self.lang_cds = list(self.lang_map.values())
  


    def _get_examples(self):
        self.ex_input = json.dumps(self.df.iloc[0].to_dict())
        self.token_infer = ""
        self.context_infer = "The english phrase to translate is indicated by the value for key 'en_US'. All other keys in the dict for each row are context and should be used to help inform the translation."


        #self.ex_output = 
    

    #TODO: Note, the data as input is redundant 
    def preprocess(self, data:List[str])->str: 

        ## Convert data to slug....
        prepped = json.dumps(self.df.to_dict(orient='records'))

        return prepped

    def _get_game_context(self):
        """ Helper function to get relevant context for in game localization for particular game """
        game = self.request.get('Game')
        self.game = game
        self.lang_specific_guidelines = GENERAL_LANG_SPECIFIC_GUIDELINES
        self.general_game_specific_guidelines = GENERAL_GAME_SPECIFIC_GUIDELINES
        self.game_description = self.general_game_specific_guidelines[game]

        ### still need to set self.lang_map, self.languages, self.lang_cds


    def _generate_prompt_helper(self, 
                                language:str, 
                                game:str, 
                                prepped:str)->List[Dict[str, Any]]:
        
        
        self._get_examples()

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
            {{ "en_US": "english phrase 1", "{lang_cd}": "translated phrase 1" }},
            {{ "en_US": "english phrase 2", "{lang_cd}": "translated phrase 2" }},
            ...
            ]\n\n
            """

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
                temperature=0.05  # adjust for creativity vs. stability
        )
    
        ### call GPT model for translation
        #GPT chat completions prompt
        #raw_results = self.gpt_model.

        output = response.choices[0].message.content
        usage = response.usage
        return output, usage

    
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


    def postprocess(self, 
                    outputs:str)->List[pd.DataFrame]: 
        
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


        results = self.df
        for i in post:
            results = results.merge(i, on=['en_US'],how='left')
        
        self.results = results[['en_US',*self.lang_cds]]

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

