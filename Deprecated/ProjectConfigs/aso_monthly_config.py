SOURCE_CD = "ASO Monthly"
ALL_TARGET_LANGUAGES =  ['Latin American Spanish','Brazilian Portuguese','Italian','Japanese','French','German','Simplified Chinese', 'Traditional Chinese','Korean','Russian']
ALL_LANG_CDS = ['es_LA','pt_BR','it_IT','ja_JP','fr_FR','de_DE','zh_CN','zh_TW','ko_KR','ru_RU']

## Mapping for language --> lang_cd
LANG_MAP = dict(zip(ALL_TARGET_LANGUAGES, ALL_LANG_CDS))
ALL_LANG_MAP = LANG_MAP

# Mapping for lang_cd -> language 
#LANG_MAP_REV = dict(zip(ALL_LANG_CDS,ALL_TARGET_LANGUAGES))

### TODO: When splitting the languages from the entry, trim off the leading white space...
"""
TARGET_LANGUAGE_MAPPING = {
    'Spanish (Latin America)':'es_LA',
    ' French (France)':'fr_FR',
    ' German':'de_DE',
    ' Russian':'ru_RU',
    ' Korean':'ko_KR',
    ' Italian':'it_IT',
    ' Japanese': 'ja_JP',
    ' Simplified Chinese': 'zh_CN',
    ' Traditional Chinese (Taiwan)':'zh_TW',
    ' Portuguese (Brazil)':'pt_BR',
}"""

#trimmed_list = [s.lstrip() for s in original_list]

#TODO: Fix this for future 
#all_language_map = ALL_LANG_MAP
#all_language_cds = ALL_LANG_CDS
#all_languages = ALL_TARGET_LANGUAGES

"""
import pandas as pd
import gspread


def get_processing_rows_from_sheet_aso_monthly():

    #WORKBOOK_ID = "1WnCBGk1V1nEUeU7aPWBqrQXczmaMop6AbGmkDKkAQcE"
    ASO_MONTHLY_SS_URL = "https://docs.google.com/spreadsheets/d/1WnCBGk1V1nEUeU7aPWBqrQXczmaMop6AbGmkDKkAQcE/edit?gid=1033119277#gid=1033119277"

    WKSHT_NAME = "Validation and Url Tracking"

    source_ss = gc.open_by_url(ASO_MONTHLY_SS_URL)

    source_wksht = source_ss.worksheet(WKSHT_NAME)
    vals = source_wksht.get_all_values()

    headers, data = vals[0],vals[1:]
    df = pd.DataFrame(data, columns=headers)

    to_process_df = df[df.JobStatus == 'NotStarted']


    return to_process_df


def get_row_dicts(to_process_df:pd.DataFrame, 
                  source_cd:str = 'ASO Monthly')->list:

    list_slugs = []
    for i,row in to_process_df.iterrows():
        row_dict = row.to_dict()
        row_dict['source_cd'] = source_cd
        json_job_slug = json.dumps(row_dict)
        list_slugs.append(json_job_slug)

    return list_slugs
"""



