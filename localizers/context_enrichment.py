### Context Enrichment/Preprocessing step
import re
import json
import pandas as pd

def build_context_enrichment_prompt():

  # peform preprocessing
  PREPROCESSING_PROMPT = f""" You are helping preprocess item-description data for a mobile game called Disney Magic Match.

  Each input row contains:
  - KEY: an internal item key
  - en_US: an English flavor text line

  The English flavor text may include puns, wordplay, humor, or figurative language. Your task is NOT to explain the joke. Your task is to identify the underlying object being described and generate structured preprocessing fields.

  For each row, produce the following new columns:

  1. object
  - Identify the core object being described.
  - Infer it primarily from the KEY, using the English phrase only as supporting context.
  - Use a short singular noun phrase.
  - Normalize the object into a clean, simple label.
  - Examples:
    - "dumbbell"
    - "wide-brim hat"
    - "bowler hat"
    - "surfboard"
    - "music note"
    - "treasure chest"

  2. descriptor
  - Identify the descriptor attached to the object, if present.
  - Usually this comes from the KEY and may refer to color, material, style, shape, tone, size, or other modifier.
  - Examples:
    - "blue"
    - "purple"
    - "silver"
    - "light gold"
    - "round"
    - "wide-brim"
    - "murky"
  - If no useful descriptor is present, output an empty string.

  3. additional_context
  - Add short helpful inferred context ABOUT the object itself.
  - This should describe what the object is, what it is used for, or what kind of thing it is.
  - Do NOT explain the joke or pun.
  - Do NOT repeat the object or descriptor unless necessary.
  - This field is optional. If there is no useful extra context, output an empty string.
  - Good example:
    - For "wide-brim hat": "A style of hat with a large brim, often used for shade or sun protection."
  - Other examples:
    - "A handheld controller used to play video games."
    - "A tool with two blades used for cutting paper or other materials."
    - "A musical symbol used to represent pitch or rhythm."

  4. simple_description_en
  - Write a plain, literal English description of the item using the object, descriptor, and any relevant context.
  - This should be simple and clear, not funny.
  - Do NOT preserve the pun or wordplay.
  - Keep it concise, usually one sentence or sentence fragment.
  - Examples:
    - "A blue dumbbell used for strength training."
    - "A purple wide-brim hat that provides shade from the sun."
    - "A silver drinking glass for serving beverages."

  General rules:
  - Prioritize accuracy and consistency over creativity.
  - Base the answer mainly on the KEY, then refine with the English flavor text if helpful.
  - Ignore the humorous wording when identifying the object.
  - Normalize object names so similar items use the same naming style across rows.
  - Use sentence case for descriptions.
  - Do not invent lore, story context, or visual details that are not reasonably inferable.
  - If the KEY implies a stylized or unusual color for a real-world object, keep that descriptor.
  - If a gemstone or item name is specific (for example, "emerald"), preserve that specificity in the object when useful.
  - If the descriptor is actually a material or style rather than a color, record it as written in natural English.
  - If the object name in the KEY is camelCase, convert it into normal readable English.
    - Example: "hatWideBrim" -> "wide-brim hat"
    - Example: "tableRound" -> "round table"
    - Example: "musicNote2" -> "music note"

  Do not include any extra commentary.
  Do not include markdown fences.
  Do not include the original input fields unless explicitly requested. 


  Output format:
  """

  PREPROCESSING_PROMPT += f"""
              Respond in **JSON format**, one object per row:
              json
              [
              {{ "key":"key 1",
                  "object": "object 1",
                  "descriptor": "descriptor 1",
                  "additional_context":  "additional context 1",
                  "simple_description_en":"simple description 1" }},
              {{ "key":"key 2",
                  "object": "object 2",
                  "descriptor": "descriptor 2",
                  "additional_context":  "additional context 2",
                  "simple_description_en":"simple description 2" }},,
              ...
              ]\n\n
              """
  return PREPROCESSING_PROMPT


def prepare_data_json_for_enrichment(item_desc):
    PH_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
    payload = []
    for _, r in item_desc.iterrows():
        en = r.get("en_US", "") or ""
        key = r.get("KEY")
        item = {
            "en_US": en,
            'key':key
            # keep payload tiny; include only relevant hints
        }
            
        # placeholders list (helps the model self-check)
        ph = PH_RE.findall(en)
        if ph:
            item["placeholders"] = ph
        payload.append(item)


    prepped = json.dumps(payload)
    return prepped


def preprocess_item_desc_DMM(df):


    PREPROCESSING_PROMPT = build_context_enrichment_prompt()
    data_json = prepare_data_json_for_enrichment(df)
    return [
            {"role": "system", "content": PREPROCESSING_PROMPT},
            {"role": "user",   "content":data_json}
    ]


def write_preprocessed(parsed_df):
    num_cols= len(list(parsed_df.columns))
    num_rows = len(parsed_df.values.tolist())+1
    try:
        sh.add_worksheet('preprocessed', rows=num_rows, cols=num_cols)
    except Exception as e:
        pass
    preprocessed = sh.worksheet('preprocessed')

    headers = parsed_df.columns.tolist()
    out_data = parsed_df.values.tolist()


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
    letter_range = number_to_letter[str(len(headers))]
    data_range = f"A2:{letter_range}{len(out_data)+1}"


    preprocessed.batch_update([{
                    'range':f"A1:{letter_range}1", 'values':[headers]
                    },
                    {
                    'range':data_range, 'values':out_data
                    }])
    
    return "Done writing preprocessed!"