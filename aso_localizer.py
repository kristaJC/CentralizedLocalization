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

from general_config import *
#from ASO_Config import ASO_TARGET_LANGUAGE_MAPPING, ASO_HP_TARGET_LANGUAGE_MAPPING
from aso_config import *


EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"

class ASOLocalizer(LocalizationRun):

    def __init__(self, 
                 request, 
                 gsheet_client=None, 
                 gpt_client=None, 
                 cfg=None, 
                 tracker: MLTracker | None = None):
        super().__init__(request, gsheet_client, gpt_client, cfg, tracker)

        # Now subclass-specific initialization
        self.required_tabs = self.cfg.get("input", {}).get("required_tabs", [])
        self.char_limit_policy = self.cfg.get("char_limit_policy", "strict")
        # e.g., store header row counts for ios/android
        self.ios_header_rows = self.cfg.get("input", {}).get("ios_header_rows", 3)
        self.android_header_rows = self.cfg.get("input", {}).get("android_header_rows", 3)
        self.required_output_tabs = self.cgf.get("output sheets",[])
        self.sh = None
        self.ios_wksht = None
        self.android_wksht = None
        self.ios_long_df = None
        self.ios_wide_df = None
        self.android_long_df = None
        self.android_wide_df = None

        self._get_game_context()

    def _get_game_context(self):

        """ Helper function to get relevant context for in game localization for particular game """
        game = self.request.get('Game')
        self.game = game
        self.lang_specific_guidelines = GENERAL_LANG_SPECIFIC_GUIDELINES
        self.general_game_specific_guidelines = GENERAL_GAME_SPECIFIC_GUIDELINES

        if game not in ["Panda Pop","Cookie Jam","Harry Potter: Hogwarts Mystery","Disney Emoji Blitz","Disney Magic Match"]:
            raise Exception(f"Game {game} not supported")

        # Specifics for games
        if game == "Harry Potter: Hogwarts Mystery":
            self.lang_map = ASO_HP_TARGET_LANGUAGE_MAPPING        
        else:
            self.lang_map = ASO_TARGET_LANGUAGE_MAPPING

        self.game_description = self.general_game_specific_guidelines[game]
        self.languages = list(self.lang_map.keys())
        self.lang_cds = list(self.lang_map.values())

    def validate_inputs(self): 

        # Open url
        try:
            sh = self.gsheet_client.open_by_url(self.request.get("url"))
        except Exception as e:
            raise Exception(f"Invalid spreadsheet URL: {e}")

        self.sh = sh

        try:
            ios_wksht = self.sh.worksheet('ios')
        except:
            raise Exception(f"ios worksheet not found in spreadsheet '{self.request.get('url')}'")
        try:
            android_wksht = self.sh.worksheet('android')
        except:
            raise Exception(f"android worksheet not found in spreadsheet '{self.request.get('url')}'")

        
        # Check other tabs exist, if not, add them
        wkshts = self.sh.worksheets()
        for tab in self.required_output_tabs:
            if tab not in [w.title for w in wkshts]:
                wksht = self.sh.add_worksheet(tab, rows=200, cols = 50)
                ## add output appropriate output columns here
                # update header row and formatting if needed
        

        # Validate IOS formatting - if no data, leave the worksheet object as None
        ios_data =  ios_wksht.get_all_records()
        ios_rows = len(ios_data)
        if len(ios_data) >=4:
            self.ios_wksht = sh.worksheet('ios')
        
        ### TODO: Do some data valiation for the header rows for ios and such later
        #ios_headers = ios_rows[0:3] # make sure this is right later (exclusive end slice)
        #if 
        

        #Validate Android Formatting
        android_data =  android_wksht.get_all_records()
        android_rows = len(android_data)
        if len(android_data)>=4:
            self.android_wksht = self.sh.worksheet('android')

        ### TODO: Do some data valiation for the header rows for android and such later
        #ios_headers = ios_rows[0:3] # make sure this is right later (exclusive end slice)
        #if 
        
    
        return
    
    #Helper method to put input data into appropriate wide format
    def _get_wide_input_format(self, values, platform):

        headers, data = values[0:3],values[3:]
        if platform=='ios':
            df = pd.DataFrame(data,columns = ['en_US_30','en_US_50','en_US_120'])
        if platform=='android':
            df = pd.DataFrame(data, columns = ['en_US_80','en_US_500'])

        df['row_id'] = df.index
        df['RowFingerprint'] = self.get('RowFingerprint')
        df['platform'] = platform
        df['game'] = self.game  

        return df
    

    def _helper_by_platform_language(self, df, language:str, platform:str)->pd.DataFrame:
        lang_cd = self.lang_map[language]
        if lang_cd in ['ja_JP','ko_KR','zh_CN','zh_TW'] and platform=='android':
            altered = df.withColumn("target_char_limit", 
                                    (col('en_char_limit')/2).cast(IntegerType()))
        else:
            altered = df.withColumn("target_char_limit", col('en_char_limit'))
            
            
        altered = altered.withColumn("language_cd", 
                                    lit(lang_cd))\
                        .withColumn('language',lit(language))\
                        .withColumn(
                                "row_idx",
                                concat_ws(
                                    "::",
                                    concat_ws("_", lit("row"), col("row_id").cast("string")),
                                    col("game"),
                                    col("platform"),
                                    col("language_cd"),
                                    col("en_char_limit").cast("string")
                                )
                            )
        return altered.toPandas() # a spark dataframe

    #Helper method to convert wide inputs to long using sql query
    def _convert_wide_to_long_inputs(self, df, platform):
        
   
        #TODO make sure to start spark session
        df = spark.createDataFrame(df)
        if platform == "ios":
            return spark.sql(Q_IOS)#.toPandas()
        if platform == "android":
            return spark.sql(Q_ANDROID)#.toPandas()
        
        return

    def load_inputs(self):
        if self.ios_wksht:
            ios_data = self.ios_wksht.get_all_records()
            #TODO: Test this
            self.ios_wide_df = self._get_wide_input_format(ios_data,'ios')
    
            #TODO:  Test this
            self.ios_long_df = self._convert_wide_to_long_inputs(self.ios_wide_df,'ios')

        if self.android_wksht:
            android_data = self.android_wksht.get_all_records()
            #TODO: Test this
            self.android_wide_df = self._get_wide_input_format(android_data,'android')
            #TODO: Test this
            self.android_long_df = self._convert_wide_to_long_inputs(self.android_wide_df,'android')

        return None
    
    #def _group_prompts_for_translation(self, prompts):
    #    #[{'lang':"","prompts":[]}]
    #    return groups
 
    def preprocess(self, data:None)->Dict[str,str]: 
        
        slug_by_lang = {} # holder for each slug for inputs by language
        prepped_holder = [] # holder for the dataframes 
        for lang in self.languages:
            if self.ios_wksht:
                ios_altered = self._helper_by_platform_language(self.ios_long_df, lang, "ios")
            if self.android_wksht:
                android_altered = self._helper_by_platform_language(self.android_long_df, lang, "android")
            altered = pd.concat([ios_altered, android_altered],axis=0)
            prepped_holder.append(altered)

            vals_write = altered[['row_idx','target_char_limit','en_US']].values.tolist()
            slug = json.dumps(data_values)
            slug_by_lang[lang] = slug

        self.prepped = slug_by_lang # dictionary {'French':<json slug>,...}
        self.prepped_holder = prepped_holder
        return self.prepped
    
    def _generate_prompt_helper(self,
                                language:str, 
                                game:str, 
                                prepped:str):
        
        """ Build the prompt by language... prepped should basically be the slug for all of the inputs"""

        #TODO - likely need to adjust some information about the example inputs and outputs
        ASO_EX_INPUT = """
            [
                {"row_idx": "row_idx_1", "target_char_limit": 30, "en_US": "Super Rainbow Power"}, 
                {"row_idx": "row_idx_2", "target_char_limit": 50, "en_US": "Match pieces to collect super rainbow cake!"}, 
                {"row_idx": "row_idx_3", "target_char_limit": 120, "en_US": "Celebrate a new month with the power of Super Rainbow Cake! Match pieces to unleash it's power!"}
            ]"""


        # ASO Guidelines
        base = f""""

            You are a professional game copy localizer.

            Translate the following English phrases into {language} for a mobile game. Use the context and tone described below. Ensure each translation is idiomatic, natural, and within the specified character limit.

            --- Context ---
            Game: {game}
            Game Description: {self.game_description}

            --- Guidelines ---
            The tone should be:
            - Fun, playful, and energetic
            - Casual and approachable
            - Clear, concise, and engaging for players

            You must:
            - Use natural, idiomatic language for the target audience
            - Prioritize clarity and emotional appeal over literal translation
            - Maintain consistent tone and phrasing across all content
            - Stay within any provided character limits

            If the English text includes puns, idioms, or culturally specific phrases:
            - Adapt them to something that feels native and engaging in the target language
            - It's okay to rephrase for clarity or punch — fun and fluid is better than literal

            Avoid:
            - Overly formal or technical phrasing
            - Translating idioms or jokes literally if they don’t work in the target language
            - DO NOT EVER provide translations that are over the character limit denoted by "target_char_limit"

            
            --- Language Guidelines for Translation ---
            {self.lang_specific_guidelines[language]}

            --- Instructions ---
            - Use the game context and tone described above.  
            - Each phrase must be translated idiomatically and playfully.  
            - Do **NOT** exceed the provided character limit.  
            - Return results in JSON format.

            --- Input Description and Examples ---
            Each row includes a 'row_idx' which is a unique identifier for the row. The 'target_char_limit' is the maximum number of characters allowed for the translated phrase and must ALWAYS be respected in the translated text. The 'en_US' field is the English phrase to translate.

            Example Inputs as a json string:
            json
                [
                {"row_idx": "row_idx_1", "target_char_limit": 30, "en_US": "Super Rainbow Power"}, 
                {"row_idx": "row_idx_2", "target_char_limit": 50, "en_US": "Match pieces to collect super rainbow cake!"}, 
                {"row_idx": "row_idx_3", "target_char_limit": 120, "en_US": "Celebrate a new month with the power of Super Rainbow Cake! Match pieces to unleash it's power!"}
            ]

            """
        lang_cd = self.lang_map[language]

        # Output example
        base += f"""
            --- Output Format ---
            Respond in **JSON format**, one object per row:
            json
            [
            {{ "row_idx": "row_idx_1", "{lang_cd}" : "translated phrase 1" }},
            {{ "row_idx": "row_idx_2", "{lang_cd}": "translated phrase 2" }},
            ...
            ]\n\n
            """
        ### COULD ALTER OUTPUT LATER
    
        return [
            {"role": "system", "content": base},
            {"role": "user",   "content": prepped}
        ]


    def build_prompts(self, prepped:Dict[str,str])->Dict[str, str]: 

        prompts = []

        for lang in self.languages:
           prompt = self._generate_prompt_helper(lang, self.game, self.prepped[lang])
           prompts.append(prompt)

        self.prompts = prompts
        self.groups = dict(zip(self.languages, prompts))

        return self.groups
    

    def _call_model_batch(self, prompt:str)->Tuple[str,Dict[str,int]]:
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


    def postprocess(self, outputs)->List[pd.DataFrame]: 
        
        postprocessed_dfs = []
        for output in outputs:
            parsed = self._parse_model_json_block(output)
            returned_df = pd.DataFrame(parsed)
            postprocessed_dfs.append(returned_df)

        self.postprocessed_dfs = postprocessed_dfs 

        return self.postprocessed_dfs

    #TODO: THIS MAY BE DIFFERNT FOR THIS CONTEXT Helper function for write_outputs
    def _merge_outputs_by_language(self, 
                                   post: List[pd.DataFrame])->pd.DataFrame:
        
        #Consider doing this for different cases
        # eg.drop_cols set at init : ['context'] for panda pop
        # eg.join_cols set at init: ['token']
        aso_join_columns = []
        aso_drop_columns = []

        self.df = self.df.drop(columns= drop_columns)
        for i in post:
            self.joined_long_inputs = self.joined_long_inputs.merge(i, on= aso_join_columns,how='left')
        
        return self.df 
    
    def _finalize_status_tracking(self):
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
            return

    def _write_long_results(self):
        
        wksht = self.sh.worksheet("long_results")

        out_data = self.long_results.values.tolist()
        #data_range = f"A2:Q{len(out_data)+1}" # Figure out the right number of columns... 
        ## Likely will be 
        data_range = ""
        try:
            wksht.batch_update([{'range':data_range, 'values':out_data}])
        except Exception as e:
            print(e)
            print("Couldn't write to sheet")

        return
    
    #def _helper_group_row_id_language,align in character(self)

    def _write_formatted_results(self):

        if self.ios_wksht:
            ios_results = self.long_results.loc[(self.long_results['platform']=='ios')]
            ## Format 



            ### write to sheet
            #data_range=
            #out_data
            #self.ios_wksht.batch_update([{'range':data_range, 'values':out_data}])


        if self.android_wksht:
            android_results = self.long_results.loc[(self.long_results['platform']=='android')]



            ## Format 


            ### write to sheet
            #data_range=
            #out_data
            #self.android_wksht.batch_update([{'range':data_range, 'values':out_data}])

        return
    
    def write_outputs(self, post:List[pd.DataFrame])->str: 

        self.long_results = self._merge_outputs_by_language(post) # maybe return some tracking data for postprocessing and writing steps
       
        self._write_long_results() # maybe return some tracking data for postprocessing and writing steps
        
        self._write_formatted_results() # maybe return some tracking data for postprocessing and writing steps

        #self._finalize_status_tracking()

        return "Done!"