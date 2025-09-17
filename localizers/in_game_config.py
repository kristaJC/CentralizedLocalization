### GENERALIZE LATER
# This should include all guidelines for all languages

##########################################


PP_LANGS = ['Japanese', 'French', 'Italian', 'German', 'Spanish (Spain)', 'Korean', 'Danish', 'Swedish', 'Norwegian', 'Traditional Chinese (Taiwan)', 'Simplified Chinese', 'Portuguese (Brazil)', 'Russian', 'Arabic', 'Turkish']

PP_LANG_CDS = ['ja', 'fr', 'it', 'de', 'es', 'ko', 'da', 'sv', 'nb', 'zh-TW', 'zh-CN', 'pt-BR', 'ru', 'ar', 'tr']

PP_LANG_MAP = dict(zip(PP_LANGS, PP_LANG_CDS))

# game specific prompt inputs 

### may rewrite this later...
PP_EX_INPUT = '[{"token": "detectives_card_4", "context": "Detective Panda event", "en_US": "Treasure Map"}, {"token": "pause_button", "context": "UI label, keep < 10 characters", "en_US": "Pause"}]'

PP_CONTEXT_INFER = """Some rows may include a context field. This may contain additional information such as:\n - The theme or character referenced (e.g. 'Detective Panda', 'Mama Panda') \n -UI usage hints (e.g. 'banner title', 'button label') \n - Occasional formatting or character length tips \n \n If present, use the context to guide tone, word choice, or brevity — especially when the English phrase is vague or could be interpreted multiple ways. """

PP_TOKEN_INFER = """Each row includes a token which may contain clues about the theme or in-game context (e.g. “detectives”, “event”, “deluxe”). If possible, infer the theme from the token and apply that understanding to improve the translation — especially when the English phrase is short or ambiguous."""



##########################################


CJB_LANGS = ["Japanese", "French", "Italian", "German", "Korean", 
             "Traditional Chinese (Taiwan)", "Simplified Chinese", 
             "Portuguese (Brazil)", "Russian", "Spanish (Latin America)", 
             "Spanish (Spain)", "Norwegian", "Swedish"]

CJB_LANG_CDS = ['ja','fr','it','de','ko','zh-TW','zh-CN','pt-BR','ru','es-419','es-ES','nb','sv']

CJB_LANG_MAP = dict(zip(CJB_LANGS, CJB_LANG_CDS))

### CJB_EX_INPUT= ""


## CJB_CONTEXT_INFER = """ """

## CJB_TOKEN_INFER = """ """


##########################################

GG_LANGS = ["German", "French", "Korean", "Spanish (Spain)", "Spanish (Colombia)",
            "Italian", "Simplified Chinese", "Portuguese (Brazil)", 'Russian',
            'Norwegian', 'Swedish', 'Danish', 'Finnish', 'Dutch (Netherlands)', 
            'Thai', 'Vietnamese', 'Indonesian', 'Malay', 'Turkish', 'Japanese']


GG_LANG_CDS = ['de','fr','ko','es-ES','es-CO','it','zh-CN','pt-BR','ru','nb','sv','da','fi','nl','th','vi','id','ms','tr','ja']

GG_LANG_MAP = dict(zip(GG_LANGS, GG_LANG_CDS))

### GG_EX_INPUT= ""


## GG_CONTEXT_INFER = """ """

## GG_TOKEN_INFER = """ """



