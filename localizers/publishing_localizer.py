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
import numpy as np

from base_localizer import LocalizationRun
from ml_tracker import MLTracker

from general_config import *
from publishing_config import *


EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"


spark = pyspark.sql.SparkSession.builder.getOrCreate()

class PublishingLocalizer(LocalizationRun):

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
            self.lang_map = pub_HP_TARGET_LANGUAGE_MAPPING        
        else:
            self.lang_map = pub_TARGET_LANGUAGE_MAPPING

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
        ios_data =  self.ios_wksht.get_all_values()
        ios_rows = len(ios_data)
        if len(ios_data) < 4:
            self.ios_wksht = None
            self.formatted_ios_wksht = None

        #Validate Android Formatting
        android_data =  self.android_wksht.get_all_values()
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
 

    #### Issue here... 
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
                altered = pd.concat(holder)
            else:
                altered=holder[0]
            prepped_holder.extend(altered)

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
        pub_EX_INPUT = """
            [
                {"row_idx": "row_idx_1", "target_char_limit": 30, "en_US": "Super Rainbow Power"}, 
                {"row_idx": "row_idx_2", "target_char_limit": 50, "en_US": "Match pieces to collect super rainbow cake!"}, 
                {"row_idx": "row_idx_3", "target_char_limit": 120, "en_US": "Celebrate a new month with the power of Super Rainbow Cake! Match pieces to unleash it's power!"}
            ]"""


        # pub Guidelines
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

            ---- Handling Placeholders  ---
            Important: Keep placeholders like {ITEM} or <NAME> unchanged in the translation. Copy them exactly as in English.

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
                temperature=0.001  # adjust for creativity vs. stability
        )

        output = response.choices[0].message.content
        usage = response.usage

        return output, usage
        

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
        Produce both pub shapes. Return them in a dict so the base will hand
        them to write_outputs(formatted_rows).
        """
        wide, long_ = self._merge_outputs_by_language_wide(post)
        self.unioned_wide = wide
        self.unioned_long = long_

        artifacts = {
            "pub_outputs_wide_preview": wide.head(1000),
            "pub_outputs_long_preview": long_.head(1000),
            "pub_output_schema": {
                "wide_columns": list(wide.columns),
                "long_columns": list(long_.columns),
            },
        }
        if self.cfg.get("log_full_artifacts", False):
            artifacts["pub_outputs_wide"] = wide
            artifacts["pub_outputs_long"] = long_

        # Return BOTH for the writer
        ##TODO: we actually want to rebuild the long later
        #return {"wide": wide, "long": long_}, artifacts
        return self.unioned_long.copy(), artifacts
    
    # ---------- QC: strict char limit ----------
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

        df["translation_len"] = df["translation"].fillna("").map(lambda s: len(str(s)))
        over = df[df["translation_len"] > df["target_char_limit"].astype(int)]

        issues: List[Dict[str, Any]] = []
        for _, r in over.iterrows():
            issues.append({
                "row_idx": r["row_idx"],
                "language": r.get("language"),
                "language_cd": r.get("language_cd"),
                "platform": r.get("platform"),
                "target_char_limit": int(r["target_char_limit"]),
                "current_len": int(r["translation_len"]),
                "en_text": r.get("en_US"),  # if present in your long DF
                "current_translation": r["translation"],
            })

        return {
            "ok": (len(issues) == 0),
            "issues": issues,
            "stats": {"overlimit": int(len(issues))},
        }

    # ---------- QC Repair: re-translate only failing rows ----------
    def qc_repair(self, formatted: Any, report: Dict[str, Any], attempt: int) -> Any:
        """
        Re-translate only rows in report['issues'], per language.
        Returns a new long DataFrame with updated translations.
        """
        if not isinstance(formatted, pd.DataFrame):
            return formatted

        issues = report.get("issues", [])
        if not issues:
            return formatted

        self.tracker.event(f"QC repair (attempt {attempt}): fixing {len(issues)} overlimit rows")

        # Group issues by language so we can call the model per language with a compact JSON payload
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for it in issues:
            lang = it.get("language") or it.get("language_cd") or "unknown"
            by_lang.setdefault(lang, []).append(it)

        # Build and call prompts per language with a *hard* constraint instruction.
        new_rows: List[Dict[str, Any]] = []
        for lang, items in by_lang.items():
            # Build a tiny JSON input of just the broken rows
            # Using the *same output schema* you expect: [{row_idx, lang_cd: "new text"}]
            payload = []
            for it in items:
                payload.append({
                    "row_idx": it["row_idx"],
                    "target_char_limit": it["target_char_limit"],
                    "en_US": it.get("en_text", ""),  # if you stored English
                })
            slug = json.dumps(payload, ensure_ascii=False)

            # Build a stricter prompt: “DO NOT EXCEED N CHARS—hard requirement”
            # You can reuse your prompt builder with extra constraint text.
            strict_prompt = self._generate_qc_prompt(lang, slug)

            with self.tracker.child(f"qc_repair:{lang}") as t:
                with t.step("api_call"):
                    out_str, usage = self._call_model_batch(strict_prompt)
                p, c = MLTracker.extract_usage_tokens(usage)
                t.metrics({"qc.tokens.prompt": p, "qc.tokens.completion": c})

            # Parse the model JSON (same parser you already have)
            try:
                fixed_list = self._parse_model_json_block(out_str)
            except Exception as e:
                self.tracker.event(f"QC parse failed for {lang}: {e}")
                continue

            # fixed_list like: [{"row_idx": "...", "<lang_cd>": "new text"}, ...]
            # Merge back into `formatted` by row_idx + language_cd
            lang_cd = self.lang_map[lang]
            fix_df = pd.DataFrame(fixed_list)
            if "row_idx" in fix_df.columns and lang_cd in fix_df.columns:
                fix_df = fix_df[["row_idx", lang_cd]].rename(columns={lang_cd: "translation"})
                fix_df["language"] = lang
                fix_df["language_cd"] = lang_cd
                new_rows.append(fix_df)

        if not new_rows:
            return formatted

        fixes = pd.concat(new_rows, axis=0, ignore_index=True)

        updated = self.apply_translation_fixes(formatted, fixes)
        
        # (Optional) Rebuild the wide view for logging again
        try:
            wide_again, long_again = self._merge_outputs_by_language_wide([df for _, df in updated.groupby("language_cd")])
            self.unioned_wide = wide_again
            self.unioned_long = long_again
            # You may want the QC loop to continue working with the long df:
            formatted = self.unioned_long.copy()
            # Re-log artifacts for this attempt:
            self.tracker.log_artifact_df(self.unioned_long, f"qc/attempt_{attempt}_long.csv")
            self.tracker.log_artifact_df(self.unioned_wide, f"qc/attempt_{attempt}_wide.csv")
        except Exception:
            # If your helper expects a different shape, just continue with the updated long df
            self.tracker.log_artifact_df(updated, f"qc/attempt_{attempt}_long.csv")
            formatted = updated

        return formatted
    
    def _normalize_lang_col(self,df: pd.DataFrame) -> pd.DataFrame:
        # tolerate either language_cd or target_lang_cd
        if "language_cd" in df.columns:
            return df
        if "target_lang_cd" in df.columns:
            return df.rename(columns={"target_lang_cd": "language_cd"})
        return df

    # formatted: long DF with ['row_idx','language_cd','translation',...]
    # fixes: DF with new translations for subset rows; must end up as
    #        ['row_idx','language_cd','translation_fix']
    def apply_translation_fixes(self, formatted: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
        formatted = self._normalize_lang_col(formatted.copy())
        fixes = fixes.copy()

        # Ensure fixes has language_cd and a translation_fix column
        if "language_cd" not in fixes.columns and "target_lang_cd" in fixes.columns:
            fixes = fixes.rename(columns={"target_lang_cd": "language_cd"})

        # If fixes still lacks language_cd but has a single language, you can inject it:
        # if "language_cd" not in fixes.columns and "language" in fixes.columns:
        #     fixes = fixes.rename(columns={"language": "language_cd"})

        # Normalize fix col name
        if "translation_fix" not in fixes.columns:
            # typical case: fixes has 'translation' (from parsed model output)
            if "translation" in fixes.columns:
                fixes = fixes.rename(columns={"translation": "translation_fix"})
            elif "translation_fixed" in fixes.columns:   # earlier variant
                fixes = fixes.rename(columns={"translation_fixed": "translation_fix"})
            else:
                # nothing to apply
                return formatted

        key_cols = ["row_idx", "language_cd"]

        # Sanity: ensure keys exist
        for k in key_cols:
            if k not in formatted.columns or k not in fixes.columns:
                # Nothing to merge if keys missing
                return formatted

        # DO NOT drop 'translation' before merge; merge fixes to the right
        updated = formatted.merge(fixes[key_cols + ["translation_fix"]],
                                on=key_cols, how="left")

        # Prefer fixed where present
        if "translation_fix" in updated.columns:
            updated["translation"] = np.where(
                updated["translation_fix"].notna(),
                updated["translation_fix"],
                updated["translation"]
            )
            updated = updated.drop(columns=["translation_fix"])

        return updated

    # --- helper: stricter prompt for QC repair ---
    def _generate_qc_prompt(self, language: str, slug_json: str):
        """
        Reuse your style but *enforce* hard char cap.
        Expect JSON input: [{row_idx, target_char_limit, en_US}]
        Output JSON: [{row_idx, <lang_cd>: "fixed translation <= limit"}]
        """
        lang_cd = self.lang_map[language]
        base = f"""
            You are a professional localizer. FIX translations that exceed character limits.
            HARD REQUIREMENT: The translation for each row MUST be <= target_char_limit characters (count spaces/punctuation).
            If needed, shorten by rephrasing while keeping meaning & tone. Do NOT omit the meaning.

            Important: Keep placeholders like {ITEM} or <NAME> unchanged in the translation. Copy them exactly as in English.

            Output JSON only, with this schema:
            [
              {{ "row_idx": "row_...", "{lang_cd}": "<final translation <= target_char_limit>" }},
              ...
            ]
        """
        return [
            {"role": "system", "content": base},
            {"role": "user", "content": slug_json}
        ]

    def _write_long_results(self, long_df: pd.DataFrame = None):
        
        print("TRYING LONG RESULTS")
        try:
            wksht = self.sh.worksheet("long results")
            print("opening long results tab")
        except Exception as e:
            print(e)
            print("Couldn't open long results tab")

        # This could be in configs
        long_order = ["RowFingerprint","row_idx",'row_id','en_char_limit','game','platform','type_desc','en_US','language','language_cd','target_char_limit','translation']

        if long_df is not None:
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
        try:
            wksht = self.sh.worksheet("wide results")
            print("opening wide results tab")
        except Exception as e:
            print(e)
            print("Couldn't open wide results tab")

        #if "wide results" in self.required_output_tabs:
        wide_order = ['row_id','en_char_limit','game','platform','type_desc','en_US',*self.lang_cds]

        if wide_df is not None:
            ordered_wide = wide_df[wide_order]
        else:
            ordered_wide = self.unioned_wide[wide_order]

        data_wide_range = f"A2:P{len(ordered_wide)+1}"
        wide_data = ordered_wide.values.tolist()

        #wksht = self.sh.worksheet("wide results")
        try:
            wksht.batch_update([{'range':data_wide_range, 'values':wide_data}])
            print("Writing wide results...DONE!")
        except Exception as e:
            print(e)
            print("Couldn't write wide results to tab")

        return
    

    def write_outputs(self, formatted_rows=None) -> str:
        """
        formatted_rows comes from _format_results(). For pub it's {"wide": df, "long": df}.
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
    