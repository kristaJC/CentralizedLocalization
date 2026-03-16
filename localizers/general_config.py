# Model and experiment cases

EXPERIMENT_NAME = "/Users/pandreoni@jamcity.com/centralized_loc_translation_run"
MODEL = 'gpt-4o'
TEMP = 0.05


# Tracking spreadsheet for all requests
CENTRALIZED_SHEET_URL = "https://docs.google.com/spreadsheets/d/13IqPWBFqoZwALTbYLSeEn_pGYLWOXIkNNN2Y2yyzj48/edit?gid=0#gid=0"

# Directory where notebooks are stored
DIR = "/Workspace/Users/krista@jamcity.com/CentralizedLocalizationWorkflow/" 

GENERIC_LOC_CFG = {
    "input": {"required_tabs": ["input","output"]},
    "char_limit_policy": "strict",
}

INGAME_LOC_CFG = {
    "input": {"required_tabs": ["input","output"], 
              "input_headers": ['token', 'context', 'en_US'], 
              "output_headers": ['token','en_US','ja','fr',
                'it','de','es','ko','da','sv','no','zh-TW',
                'zh-CN','pt-BR','ru','ar','tr']
    },
    "char_limit_policy": "strict",
    "tracking_sheet_url": "",
}

PUBLISHING_LOC_CFG = {
    "input": 
        {
            "required_tabs": ["ios","android"],
            "ios_header_rows": 3, 
            "android_header_rows": 3
        },
    ##add more formatting data for header rows
    "char_limit_policy": "strict",
    "output_sheets":
        ["formatted ios", "formatted android", "long results",'wide results'],
    ##add more formatting info for output sheets
    "qc": 
        {
            "enabled": True, 
            "max_retries": 5,
        }
}

MARKETING_LOC_CFG = {
    "input": {"required_tabs": ["input","output"]},
    "char_limit_policy": "strict",
}


# General guidelines for languages
GENERAL_LANG_SPECIFIC_GUIDELINES = {

    "English (Great Britain)": """
        Tone:
            Friendly, natural, and upbeat.
        Style:
            Clear, concise, and conversational — use British spelling (e.g., “favourite,” “colour,” “centre”).
        Notes:
            Avoid Americanisms and overly casual slang; keep phrasing natural for UK players.
            Use simple action-oriented CTAs like “Play now!”, “Join the event!”, “Collect rewards!”
        Examples:
            - "Start now!" → "Start now!"
            - "Collect rewards!" → "Collect rewards!"
            - "Win big prizes!" → "Win big prizes!"
    """,

    "French": """
        Tone:
            Informal (tu-form).
        Style:
            Smooth and playful, but avoid lengthy constructions.
        Notes:
            Avoid starting phrases with “Les X ont commencé !” unless space allows.
            Prefer direct imperatives like “Joue !”, “Commence !”, “Découvre !”.
        Examples:
            - "Play now!" → "Joue maintenant !"
            - "Start the event!" → "Commence l’événement !"
    """,

    "French (Canada)": """
        Tone:
            Informal (tu), casual, and fun.
        Style:
            Playful and colloquial, with Canadian French idioms.
        Notes:
            Avoid European French phrasing; use Quebec-specific terms (e.g., “cellulaire,” not “portable”).
    """,

    "Spanish (Spain)": """
        Tone:
            Informal (tú).
        Style:
            Natural, short, and punchy.
        Notes:
            Avoid expanded forms like “¡El evento ha comenzado!” — favor direct verbs.
        Examples:
            - "Start now!" → "¡Empieza ahora!"
            - "Complete Card Sets!" → "¡Completa los conjuntos!"
    """,

    "Spanish (Latin America)": """
        Tone:
            Informal (tú or neutral imperatives), playful, and lively.
        Style:
            Engaging, warm, and upbeat.
        Notes:
            Use neutral LatAm vocabulary (avoid Spain-specific terms like “ordenador”).
        Examples:
            - "Start now!" → "¡Empieza ahora!"
            - "Complete Card Sets!" → "¡Completa los conjuntos!"
    """,

    "Spanish (Colombia)": """
        Tone:
            Informal (tú), warm, and inviting.
        Style:
            Light, fun, and lively.
        Notes:
            Use regional phrasing like “celular” instead of “móvil.”
            Stay aligned with es-LA consistency, but allow slight flavoring for Colombian users.
    """,

    "Italian": """
        Tone:
            Informal and motivational.
        Style:
            Use imperatives like Gioca, Scopri, Vinci.
            Avoid long clauses and unnecessary articles.
        Notes:
            Keep phrasing gamey and energetic.
        Examples:
            - "Worlds of Wonder has started!" → "Inizia Mondi Meravigliosi!"
            - "Get big rewards!" → "Ricevi grandi ricompense!"
    """,

    "Portuguese (Brazil)": """
        Tone:
            Informal (você-form).
        Style:
            Expressive but concise — short, motivating sentences work best.
        Notes:
            Prefer "Comece já!" and "Jogue agora!" over longer passive forms.
        Examples:
            - "The event has started!" → "Comece o evento!"
            - "Collect cards!" → "Colete cartas!"
    """,

    "German": """
        Tone:
            Informal (du-form).
        Style:
            Direct and motivational; favor imperatives like Spiele, Entdecke, Gewinne.
        Notes:
            Avoid overly long noun compounds or trailing verbs where possible.
        Examples:
            - "Start now!" → "Jetzt starten!"
            - "Collect cards!" → "Sammle Karten!"
    """,

    "Polish": """
        Tone:
            Informal (ty), energetic, and engaging.
        Style:
            Short, punchy, and game-like.
        Notes:
            Use clear imperatives (Graj, Zbieraj, Wygraj) and natural Polish word order.
            Avoid overly long sentences or stiff expressions.
        Examples:
            - "Start now!" → "Zacznij teraz!"
            - "Collect rewards!" → "Zbieraj nagrody!"
            - "Win big prizes!" → "Wygraj wielkie nagrody!"
    """,

    "Russian": """
        Tone:
            Formal 2nd person plural (вы), uncapitalized.
        Style:
            Polite and clear, not bureaucratic.
        Notes:
            Mix imperative and noun-based phrasing for balance.
            Avoid long descriptive constructions.
        Examples:
            - "Start now!" → "Начните сейчас!"
            - "Collect Card Sets!" → "Соберите наборы карт!"
    """,

    "Turkish": """
        Tone:
            Informal (sen), casual, and friendly.
        Style:
            Short, motivational, and visually punchy.
        Notes:
            Favor imperative verbs like Başla, Tamamla, Topla.
            Avoid repetitive phrasing like “tamamlamaya devam et.”
        Examples:
            - “Complete Card Sets!” → “Kart Setlerini Tamamla!”
            - “Win big rewards!” → “Büyük Ödülleri Kazan!”
    """,

    "Japanese": """
        Tone:
            Casual-polite (〜しよう, 〜が登場).
        Style:
            Dynamic, concise, and gacha-friendly.
        Notes:
            Avoid chaining too many clauses; favor short, high-energy verbs like ゲット, 集めよう.
        Examples:
            - “Collect cards!” → “カードを集めよう！”
            - “Win big rewards!” → “豪華報酬をゲット！”
    """,

    "Korean": """
        Tone:
            Casual or semi-formal (-요 form).
        Style:
            Encouraging and direct.
        Notes:
            Use game terms like 보상 (rewards), 모으다 (collect), 시작하다 (start).
        Examples:
            - “Start now!” → “지금 시작하세요!”
            - “Complete card sets!” → “카드를 모아 보상 받으세요!”
    """,

    "Danish": """
        Tone:
            Informal and instructive.
        Style:
            Short imperatives like Fuldfør, Spil, Saml.
        Notes:
            Trim long constructions where possible.
        Examples:
            - “Collect rewards!” → “Saml belønninger!”
            - “Start now!” → “Start nu!”
    """,

    "Dutch (Netherlands)": """
        Tone:
            Informal, approachable, and friendly (je/jij).
        Style:
            Casual, upbeat, and slightly playful, but not childish.
        Notes:
            Keep tone lighthearted and use natural Dutch phrasing.
            Avoid over-translating gaming terms like “combo” or “level.”
        Examples:
            - “Pop all the bubbles to win!” → “Knal alle bubbels om te winnen!”
    """,

    "Finnish": """
        Tone:
            Informal (sinä), friendly, and playful.
        Style:
            Concise and upbeat.
        Notes:
            Avoid stiff literal translations; use natural spoken Finnish.
    """,

    "Swedish": """
        Tone:
            Informal and motivational.
        Style:
            Snappy and direct.
        Notes:
            Trim unnecessary articles and helpers.
        Examples:
            - “Start the event!” → “Starta evenemanget!”
            - “Get rewards!” → “Få belöningar!”
    """,

    "Norwegian": """
        Tone:
            Informal and energetic.
        Style:
            Use imperatives like Spill, Fullfør, Samle.
        Notes:
            Avoid overly long verb chains.
        Examples:
            - “Collect cards!” → “Samle kort!”
            - “Start now!” → “Start nå!”
    """,

    "Traditional Chinese (Taiwan)": """
        Tone:
            Clear and punchy Mandarin.
        Style:
            Direct and compact for mobile UI.
        Notes:
            Use phrases like 限時, 組合, 獎勵; avoid excess explanations.
        Examples:
            - “Complete card sets!” → “完成卡牌組合！”
            - “Win rewards!” → “獲得獎勵！”
    """,

    "Traditional Chinese (Hong Kong)": """
        Tone:
            Friendly, casual, slightly cheeky.
        Style:
            Playful and fun.
        Notes:
            Use Hong Kong-specific traditional characters (not Taiwan forms).
            Add light Cantonese flavor without heavy slang.
    """,

    "Simplified Chinese": """
        Tone:
            Direct and engaging.
        Style:
            Use short, action-driven phrasing.
        Notes:
            Include key terms like 限时, 收集, 获得, 卡组.
        Examples:
            - “Start now!” → “马上开始！”
            - “Collect cards for rewards!” → “收集卡牌赢奖励！”
    """,

    "Arabic": """
        Tone:
            Polite yet energetic (Modern Standard Arabic).
        Style:
            Compact and direct with imperative CTAs.
        Notes:
            Avoid verbose or repetitive phrasing.
        Examples:
            - “Start now!” → “ابدأ الآن!”
            - “Complete sets for rewards!” → “أكمل المجموعات لتحصل على المكافآت!”
    """,

    "Hebrew": """
        Tone:
            Informal and engaging.
        Style:
            Energetic, direct, and natural for mobile.
        Notes:
            Use masculine/feminine forms as contextually appropriate.
            Maintain correct RTL direction and spacing.
        Examples:
            - “Start now!” → “התחילו עכשיו!”
            - “Collect cards!” → “אספו קלפים!”
            - “Win rewards!” → “זכו בפרסים!”
    """,

    "Thai": """
        Tone:
            Polite but casual.
        Style:
            Playful, light, and friendly.
        Notes:
            Use ครับ/ค่ะ endings where appropriate; avoid stiff formal language.
    """,

    "Vietnamese": """
        Tone:
            Friendly, natural, and conversational.
        Style:
            Concise, energetic, and engaging.
        Notes:
            Use bạn for player address; avoid overly formal phrasing.
            Use active voice; avoid long or literal translations.
        Examples:
            - “Let’s blast some bubbles!” → “Cùng bắn bong bóng nào!”
    """,

    "Filipino": """
        Tone:
            Informal, friendly, and fun.
        Style:
            Casual and lively; Taglish (English + Filipino) acceptable.
        Notes:
            Favor modern, playful phrasing that feels natural.
    """,

    "Indonesian": """
        Tone:
            Informal (kamu), warm, and casual.
        Style:
            Fun, lively, and conversational.
        Notes:
            Avoid heavy formality (e.g., Anda); keep phrasing spoken-like.
    """,

    "Malay": """
        Tone:
            Informal, casual, and direct.
        Style:
            Light and friendly.
        Notes:
            Avoid overly formal or literary Malay; prefer modern phrasing.
    """,

    "Hindi": """
        Tone:
            Friendly, energetic, and motivational.
        Style:
            Short, playful, and natural.
        Notes:
            Use direct imperatives (शुरू करो!, खेलो!, जीत लो!) and avoid passive constructions.
            Retain common English game terms like “level” or “combo.”
        Examples:
            - “Start now!” → “अभी शुरू करो!”
            - “Play now!” → “अभी खेलो!”
            - “Win rewards!” → “इनाम जीत लो!”
    """,
}


### General guides for games
GENERAL_GAME_SPECIFIC_GUIDELINES = {
    'Cookie Jam': """ A colorful match-3 puzzle game set in a cheerful bakery world. Players help Chef Panda and his assistants bake desserts by matching cookies, candies, and pastries. The vocabulary should feel light, sweet, and food-related, often tied to baking (e.g., “cookie,” “cake,” “oven,” “recipe”). Tone is playful and friendly.""",

    'Disney Emoji Blitz':""" A match-3 puzzle game featuring emoji versions of Disney and Pixar characters. Players collect emoji-style characters and power-ups, each with special abilities. Vocabulary should reflect both Disney’s official character names and the emoji-style playful format (e.g., “emoji,” “power-up,” “collection”). Tone is light, fun, energetic, and playful, reflecting both mobile chat culture and Disney magic.""",

    'Panda Pop':""" A bubble-shooter puzzle game where Mama Panda rescues her baby pandas trapped in colorful bubbles. Players shoot bubbles from the bottom of the screen to make matches and free the babies. Vocabulary is centered on pandas, colors, bubbles, and rescue themes. Tone is warm, caring, but still playful and lighthearted.""",

    'Harry Potter: Hogwarts Mystery':""" A story-driven role-playing game set in the official Harry Potter universe. 
    
    Players create their own Hogwarts student and progress through school years, attending classes, learning spells, making friends and rivals, and uncovering mysteries. All names, locations, spells, creatures, and items must follow the official Harry Potter franchise terminology (e.g., “Hogwarts,” “Potion,” “Quidditch,” “Alohomora”). Tone is magical, immersive, and aligned with the wizarding world.

    Proper nouns & spells: Preserve every franchise term exactly as it appears in the official localized Wizarding World canon for your language. Examples: Hufflepuff → Poufsouffle (FR), Diagon Alley → Chemin de Traverse (FR). Do not alter house names, character names, spells, or location names.

    What can change: Ordinary adjectives, verbs, and connective text may be freely adapted to fit local style and store character limits.""",

    'Cookie Jam Blast': """ A sequel to Cookie Jam with faster-paced match-3 action. Still starring Chef Panda, but with more emphasis on “blasting” pieces, combos, and quick gameplay. Vocabulary should combine the same sweet bakery terms with action-oriented words like “blast,” “pop,” and “combo.” Tone is energetic, fun, and slightly more dynamic than the original. """,

    "Disney Magic Match": """ A match-3 puzzle game themed around Disney animated movies. Players solve puzzles to unlock characters, scenes, and storylines from various Disney classics (e.g., Frozen, The Little Mermaid, Aladdin). Vocabulary should stay consistent with the Disney universe, using official character names, settings, and items. Tone is whimsical, family-friendly, and tied to the magic of Disney stories. """,

    "Genies & Gems": """ A fantasy-themed match-3 adventure. Players follow Jenni the Genie and her fox companion Trix as they solve puzzles and recover lost treasures from bandits. Vocabulary includes gems, magic, adventure, and treasure-hunting words (e.g., “ruby,” “amulet,” “caravan,” “bandits”). Tone is adventurous, mystical, and slightly whimsical. """,
}



## ALL_LANGUAGES Map
ALL_LANGUAGES = {
    "Japanese": "ja",
    "French": "fr",
    "French (France)":"fr-FR",
    "Italian": "it",
    "German": "de",
    "Spanish (Spain)": "es-ES",
    "Spanish":"es-ES",
    "Korean": "ko",
    "Danish": "da",
    "Swedish": "sv",
    "Norwegian": "nb",
    "Finnish": "fi",
    "Traditional Chinese (Taiwan)": "zh-TW",
    "Simplified Chinese": "zh-CN",
    "Traditional Chinese (Hong Kong)": "zh-HK",
    "Portuguese (Brazil)": "pt-BR",
    "Russian": "ru",
    "Arabic": "ar",
    "Turkish": "tr",
    "Spanish (Latin America)": "es-419",
    "French (Canada)": "fr-CA",
    "Malay": "ms",
    "Filipino": "fil",
    "Thai": "th",
    "Indonesian": "id",
    "Vietnamese": "vi",
    "Dutch (Netherlands)":"nl",
    "Hindi":"hi",
    "Polish":"pl",
    "Hebrew":"he",
    "English (Great Britain)":"en-GB",
}