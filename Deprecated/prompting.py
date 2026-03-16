import mlflow
import json
from datetime import datetime
from config import MODEL
from language import *
import re
import pandas as pd


def generate_prompt(language, job_dict):
    lang_map = job_dict['all_language_map']
    lang_cd = lang_map['language']
    lang_cd = ALL_LANGUAGE_MAP[language]
    #lang_specific_guidelines = LANG_SPECIFIC_GUIDELINES[lang_cd]

    ##
    return 

def generate_PP_prompt(language, slug):
    
    token_infer = """Each row includes a token which may contain clues about the theme or in-game context (e.g. “detectives”, “event”, “deluxe”).
    If possible, infer the theme from the token and apply that understanding to improve the translation — especially when the English phrase is short or ambiguous."""

    context_infer = """Some rows may include a context field. This may contain additional information such as:\n - The theme or character referenced (e.g. 'Detective Panda', 'Mama Panda') \n -UI usage hints (e.g. 'banner title', 'button label') \n - Occasional formatting or character length tips \n \n
    If present, use the context to guide tone, word choice, or brevity — especially when the English phrase is vague or could be interpreted multiple ways. """


    ex_input = '[{"token": "detectives_card_4", "context": "Detective Panda event", "en_US": "Treasure Map"}, {"token": "pause_button", "context": "UI label, keep < 10 characters", "en_US": "Pause"}]'


    base = f""" 
    You are a professional game localizer translating for a popular mobile puzzle game called Panda Pop by Jam City.
    Please translate the in-game phrases provided below from English into {language}.
        •   Keep the translations natural, playful, and appropriate for a casual mobile gaming tone.
        •   Avoid overly formal or mechanical language.
        •   There is no strict character limit, but translations should not be egregiously longer than the original English text.
        •   Each row includes a token which may contain clues about the theme or in-game context (e.g. “detectives”, “event”, “deluxe”). If possible, infer the theme from the token and apply that understanding to improve the translation — especially when the English phrase is short or ambiguous.
        •   Some rows may include a context field. This may contain additional information such as:\n - The theme or character referenced (e.g. 'Detective Panda', 'Mama Panda') \n -UI usage hints (e.g. 'banner title', 'button label') \n - Occasional formatting or character length tips \n \n
    If present, use the context to guide tone, word choice, or brevity — especially when the English phrase is vague or could be interpreted multiple ways.
    Example Inputs as a json string:
    json
        {ex_input}
    
    """
    #TODO: add language specific guidelines
    base += f"""You MUST follow these language specific guidelines:{LANG_SPECIFIC_GUIDELINES[language]} """
    #TODO: add token limit check

    lang_cd = LANG_MAP[language]
 
    base += f"""Respond in **JSON format**, one object per row:
    json
    [
    {{ "token": "token_name_1", "{lang_cd}": "translated phrase 1" }},
    {{ "token": "token_name_2", "{lang_cd}": "translated phrase 2" }},
    ...
    ]\n\n"""
 
    #user_message = base + ex_output
 
    return [
        {"role": "system", "content": base},
        {"role": "user",   "content": slug}
    ]

def parse_model_json_block(raw_output):
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