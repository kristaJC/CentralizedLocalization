ios_input_headers = [["Title", "Short Description","Long Description"],["English", "English","English"],[30,50,120]]
ios_input_header_range = "A1:C3"

android_input_headers = [["Short Description","Long Description"],["English","English"],[80,500]]
android_input_header_range = "A1:B2"


lang_order = ['es_LA','de_DE','it_IT','fr_FR','ja_JP','zh_CN','zh_TW','ko_KR','pt_BR','ru_RU']
hp_lang_order = ['es_LA','de_DE','it_IT','fr_FR','ja_JP','zh_CN','zh_TW','ko_KR','pt_BR']


long_result_header = ["RowFingerprint","row_idx",'row_id','en_char_limit','game','platform','type_desc','en_US','language','target_lang_cd','target_char_limit','translation',"","","CHAR_COUNT","OVERLIMIT_CHECK"]


aso_cfg_example = {
    "input": 
        {
            "required_tabs": ["ios","android"],
            "ios_header_rows": 3, 
            "android_header_rows": 3
        },
    ##add more formatting data for header rows
    "char_limit_policy": "strict",
    "output_sheets":
        ["formatted_ios", "formatted_android", "long_results"],
    ##add more formatting info for output sheets
    }

Q_IOS = f"""
        select 
            game,
            platform,
            RowFingerprint,
            row_id,
            'title' as type_desc,
            en_US_30 as en_US,
            30 as en_char_limit
        from df
        union all
        select 
            game,
            platform,
            RowFingerprint,
            row_id,
            'short_description' as type_desc,
            en_US_50 as en_US,
            50 as en_char_limit
        from df
        union all
        select 
            game,
            platform,
            RowFingerprint,
            row_id,
            'long_description' as type_desc,
            en_US_120 as en_US,
            120 as en_char_limit
        from df
"""


#TODO: TEST THIS
Q_ANDROID= f""" select 
            game,
            platform,
            RowFingerprint,
            row_id,
            'short_description' as type_desc,
            en_US_80 as en_US,
            80 as en_char_limit
        from df
        union all
        select 
            game,
            platform,
            RowFingerprint,
            row_id,
            'long_description' as type_desc,
            en_US_500 as en_US,
            500 as en_char_limit
        from df
    """


ASO_TARGET_LANGUAGE_MAPPING = {
    'Spanish (Latin America)':'es_LA',
    'French (France)':'fr_FR',
    'German':'de_DE',
    'Russian':'ru_RU',
    'Korean':'ko_KR',
    'Italian':'it_IT',
    'Japanese': 'ja_JP',
    'Simplified Chinese': 'zh_CN',
    'Traditional Chinese (Taiwan)':'zh_TW',
    'Portuguese (Brazil)':'pt_BR',
}

ASO_HP_TARGET_LANGUAGE_MAPPING = {
    'Spanish (Latin America)':'es_LA',
    'French (France)':'fr_FR',
    'German':'de_DE',
    'Korean':'ko_KR',
    'Italian':'it_IT',
    'Japanese': 'ja_JP',
    'Simplified Chinese': 'zh_CN',
    'Traditional Chinese (Taiwan)':'zh_TW',
    'Portuguese (Brazil)':'pt_BR',
}
