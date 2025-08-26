ios_input_headers = [["Title", "Short Description","Long Description"],["English", "English","English"],[30,50,120]]
ios_input_header_range = "A1:C3"

android_input_headers = [["Short Description","Long Description"],["English","English"],[80,500]]
android_input_header_range = "A1:B2"


lang_order = ['es_LA','de_DE','it_IT','fr_FR','ja_JP','zh_CN','zh_TW','ko_KR','pt_BR','ru_RU']


long_result_header = ["RowFingerprint","row_idx",'row_id','en_char_limit','game','platform','type_desc','en_US','language','target_lang_cd','target_char_limit','translation',"","","CHAR_COUNT","OVERLIMIT_CHECK"]


aso_cfg_example = {
    "input": {"required_tabs": ["ios","android"], "ios_header_rows": 3, "android_header_rows": 3},
    "char_limit_policy": "strict",
    "output_sheets":["formatted_ios", "formatted_android", "long_results"]}