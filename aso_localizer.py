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
from pyspark.sql.functions import *
from pyspark.sql.types import *

from base_localizer import LocalizationRun
from ml_tracker import MLTracker

from general_config import *
from aso_config import *


EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


spark = pyspark.sql.SparkSession.builder.getOrCreate()

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
        self.required_output_tabs = self.cfg.get("output sheets",[])
        self.sh = None
        self.ios_wksht = None
        self.android_wksht = None
        self.ios_long_df = None
        self.ios_wide_df = None
        self.android_long_df = None
        self.android_wide_df = None

        #self.long_wksht = None
        #self.wide_wksht = None

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
            self.sh = self.gc.open_by_url(self.request.get("URL"))
        except Exception as e:
            raise Exception(f"Invalid spreadsheet URL: {e}")

        wkshts = self.sh.worksheets()
        for tab in self.required_tabs:
            if tab not in [w.title for w in wkshts]:
               raise Exception("necessary input tabs are missing!")

        try:
            self.ios_wksht = self.sh.worksheet('ios')
        except:
            raise Exception(f"ios worksheet not found in spreadsheet '{self.request.get('URL')}'")
        try:
            self.android_wksht = self.sh.worksheet('android')
        except:
            raise Exception(f"android worksheet not found in spreadsheet '{self.request.get('URL')}'")

        
        # Check other tabs exist, if not, add them
        wkshts = self.sh.worksheets()
        for tab in self.required_output_tabs:
            if tab not in [w.title for w in wkshts]:
                print(f"Adding output sheet {tab}")
                wksht = self.sh.add_worksheet(tab, rows=400, cols = 50)

        # Validate IOS formatting - if no data, leave the worksheet object as None
        ios_data =  self.ios_wksht.get_all_records()
        ios_rows = len(ios_data)
        if len(ios_data) < 4:
            self.ios_wksht = None
            self.formatted_ios_wksht = None

        #Validate Android Formatting
        android_data =  self.android_wksht.get_all_records()
        android_rows = len(android_data)
        if len(android_data)<4:
            self.android_wksht = None
            self.formatted_android_wksht = None

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
        df['RowFingerprint'] = self.request.get('RowFingerprint')
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
        df.createOrReplaceTempView('df')
        if platform == "ios":
            return spark.sql(Q_IOS)#.toPandas()
        if platform == "android":
            return spark.sql(Q_ANDROID)#.toPandas()
        
        return

    def load_inputs(self):
        if self.ios_wksht:
            ios_data = self.ios_wksht.get_all_values()
            self.ios_wide_df = self._get_wide_input_format(ios_data,'ios')
            self.ios_long_df = self._convert_wide_to_long_inputs(self.ios_wide_df,'ios')

        if self.android_wksht:
            android_data = self.android_wksht.get_all_values()
            self.android_wide_df = self._get_wide_input_format(android_data,'android')
            self.android_long_df = self._convert_wide_to_long_inputs(self.android_wide_df,'android')

        return
    
    #def _group_prompts_for_translation(self, prompts):
    #    #[{'lang':"","prompts":[]}]
    #    return groups
 
    def preprocess(self, data=None)->Dict[str,str]: 

        slug_by_lang = {} # holder for each slug for inputs by language
        prepped_holder = [] # holder for the dataframes 
        ## Add language and character limits by language
        for lang in self.languages:
            holder = []
            if self.ios_wksht:
                ios_altered = self._helper_by_platform_language(self.ios_long_df, lang, "ios")
                holder.append(ios_altered)
            if self.android_wksht:
                android_altered = self._helper_by_platform_language(self.android_long_df, lang, "android")
                holder.append(android_altered)
            if len(holder)>1:
                altered = pd.concat(holder,axis=0)
            else:
                altered=holder[0]
            prepped_holder.append(altered)

            vals_write = altered[['row_idx','target_char_limit','en_US']].to_dict(orient='records')
            slug = json.dumps(vals_write)
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

            """

        base+= f"""
        
            Example Inputs as a json string:
            json
                [
                {{"row_idx": "row_idx_1", "target_char_limit": 30, "en_US": "Super Rainbow Power"}}, 
                {{"row_idx": "row_idx_2", "target_char_limit": 50, "en_US": "Match pieces to collect super rainbow cake!"}}, 
                {{"row_idx": "row_idx_3", "target_char_limit": 120, "en_US": "Celebrate a new month with the power of Super Rainbow Cake! Match pieces to unleash it's power!"}},
                ...
                ]\n\n"""
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


    def build_prompts(self, prepped:Dict[str,str]=None)->Dict[str, str]: 

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
        

    def _parse_model_json_block_old_ignore(self, raw_output:str)->Dict[str,Any]:
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
    
    def _helper_long_inputs_for_merge(self):

        ### Get init data
        if self.android_long_df:
            unioned_inputs = pd.concat([self.ios_long_df.toPandas(), self.android_long_df.toPandas()])
        else:
            unioned_inputs = self.ios_long_df.toPandas()

        #unioned_inputs['row_id'] = unioned_inputs['row_id'].astype(int)

        self.unioned_inputs = unioned_inputs
        return self.unioned_inputs
    
    def _row_idx_split(self,df):

        df[['row_id', 'game', 'platform', 'language_cd', 'en_char_limit']] = (
            df['row_idx'].str.split('::', expand=True)
        )
        # Clean up row_id (strip off 'row_')
        df['row_id'] = df['row_id'].str.replace('row_', '', regex=False)
        df['row_id'] = df['row_id'].astype(int)
        df['en_char_limit'] = df['en_char_limit'].astype(int)

        return df
    
    def _rename_columns(self,df):
        lang_cd = df['language_cd'].iloc[0]
        return df.rename(columns = {lang_cd: 'translation'},inplace=False)


    def _helper_parse_row_idx(self, post):
        """ Helper to get the outputs formatted for long and wide result format"""

        split_holder_wide = []
        split_holder_long = []
        for df in post:
            
            # split to wide format 
            #[row_idx,<lang_cd>,row_id,game,platform,language_cd,en_char_limit]
            df = self._row_idx_split(df)
            split_holder_wide.append(df)

            #[row_idx,translation,row_id,game,platform,language_cd,en_char_limit]
            df_long = self._rename_columns(df)
            split_holder_long.append(df_long)

        self.parsed_post_wide = split_holder_wide
        self.parsed_post_long = split_holder_long

        return self.parsed_post_wide, self.parsed_post_long

    def _merge_outputs_by_language_wide(self, 
                                   post: List[pd.DataFrame])->pd.DataFrame:
        
        parsed_postprocessed_wide, parsed_postprocessed_long = self._helper_parse_row_idx(post)

        #Start with the initial English input df, merge with each language df
        unioned_wide = self._helper_long_inputs_for_merge()
        for df in parsed_postprocessed_wide:
            unioned_wide = unioned_wide.merge(df.drop(columns = ['row_idx','language_cd']), on= ['row_id','platform','game','en_char_limit'],how='left')
        

        # Merge post processed dfs with init prepped dfs, add to a holder
        long_holder = []
        for i in range(0,len(self.prepped_holder)):
            joined = self.prepped_holder[i].merge(parsed_postprocessed_long[i], on=['game','platform','row_id','row_idx','en_char_limit','language_cd'],how='left')
            long_holder.append(joined)

        # Concat all for long format
        unioned_long = pd.concat(long_holder)
        
        return unioned_wide, unioned_long
    
    def _format_results(self, post: list[pd.DataFrame]):
        """
        Produce both ASO shapes. Return them in a dict so the base will hand
        them to write_outputs(formatted_rows).
        """
        wide, long_ = self._merge_outputs_by_language_wide(post)
        self.unioned_wide = wide
        self.unioned_long = long_

        artifacts = {
            "aso_outputs_wide_preview": wide.head(1000),
            "aso_outputs_long_preview": long_.head(1000),
            "aso_output_schema": {
                "wide_columns": list(wide.columns),
                "long_columns": list(long_.columns),
            },
        }
        if self.cfg.get("log_full_artifacts", False):
            artifacts["aso_outputs_wide"] = wide
            artifacts["aso_outputs_long"] = long_

        # Return BOTH for the writer
        return {"wide": wide, "long": long_}, artifacts

    ## DEPRECATED
    #def _format_results_helper_old(self, post: List[pd.DataFrame]):

    #    unioned_wide, unioned_long = self._merge_outputs_by_language_wide(post)
    #    self.unioned_wide = unioned_wide
    #    self.unioned_long = unioned_long
        
    #    return unioned_wide, unioned_long
    
    #def _finalize_status_tracking(self):
        #self.tracker.overall_status = "Succeeded"
        ##update tracking sheet 
        #sh = self.gc.open_by_url(TRACKING_SHEET_URL)
        #data = wksht.worksheet("Tracking").get_all_values()
        #header,values = data[0],data[1:]
        #df = pd.DataFrame(values,columns=header)

        ## Find the row, and update the status, based on row fingerprint, and status
        ###TODO
        ## row = sh.find()
        ## col = #should be established
        #self.sh.update("",self.tracker.overall_status)

        ## Send email
        ## self.gc.email??

        ## Send Slack message?

        ## 
    #    return

    def _write_long_results(self, long_df: pd.DataFrame = None):
        
        print("TRYING LONG RESULTS")
        try:
            wksht = self.sh.worksheet("long results")
            print("opening long results tab")
        except Exception as e:
            print(e)
            print("Couldn't open long results tab")

        long_order = ["RowFingerprint","row_idx",'row_id','en_char_limit','game','platform','type_desc','en_US','language','language_cd','target_char_limit','translation']

        if long_df:
            ordered_long = long_df[long_order]
        else:
            ordered_long = self.unioned_wide[long_order]
        #ordered_long = self.unioned_long[long_order]

        data_long_range = f"A2:L{len(ordered_long)+1}"
        long_data = ordered_long.values.tolist()
        try:
            wksht.batch_update([{'range':data_long_range, 'values':long_data}])
            print("Writing long results...DONE!")
        except Exception as e:
            print(e)
            print("Couldn't write long results to tab")

        return
    
    
    def _write_wide_results(self, wide_df: pd.DataFrame = None):
        
        print("TRYING WIDE RESULTS")
        #if "wide results" in self.required_output_tabs:
        wide_order = ['row_id','en_char_limit','game','platform','type_desc','en_US',*self.lang_cds]

        if wide_df:
            ordered_wide = wide_df[wide_order]
        else:
            ordered_wide = self.unioned_wide[wide_order]

        data_wide_range = f"A2:P{len(ordered_wide)+1}"
        wide_data = ordered_wide.values.tolist()

        wksht = self.sh.worksheet("wide results")
        try:
            wksht.batch_update([{'range':data_wide_range, 'values':wide_data}])
            print("Writing wide results...DONE!")
        except Exception as e:
            print(e)
            print("Couldn't write wide results to tab")

        return
    
    
    """
    # Probably better to do with the app script after writing everything to the sheet
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
    """

    def write_outputs(self, formatted_rows) -> str:
        """
        formatted_rows comes from _format_results(). For ASO it's {"wide": df, "long": df}.
        We still use the class vars set in _format_results for simplicity.
        """
        # Be tolerant to being called with either dict or None (future-proof).
        if isinstance(formatted_rows, dict):
            self.unioned_wide = formatted_rows.get("wide", self.unioned_wide)
            self.unioned_long = formatted_rows.get("long", self.unioned_long)

        # Write both tabs if configured
        self._write_long_results(self.unioned_long)
        self._write_wide_results(self.unioned_wide)
        return f"Done writing results to URL {self.sh.url}!"
    
    """
    def write_outputs_old(self, post:List[pd.DataFrame])->str: 

        #wide_results, long_results = self.format_results(post) 
        print("Writing outputs....")
        print("length long_results", len(long_results))
       
        #self.wide_results = wide_results
        #self.long_results = long_results
        self._write_long_results(self.unioned_long) 
        self._write_wide_results(self.unioned_wide)
        
        #self._write_formatted_results() 
        #self._finalize_status_tracking()

        return "Done!"

    """