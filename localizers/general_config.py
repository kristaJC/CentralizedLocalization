# Model and experiment cases

EXPERIMENT_NAME = "/Users/krista@jamcity.com/centralized_loc_translation_run"
MODEL = 'gpt-4o'
TEMP = 0.05



# Tracking spreadsheet for all requests
CENTRALIZED_SHEET_URL = "https://docs.google.com/spreadsheets/d/13IqPWBFqoZwALTbYLSeEn_pGYLWOXIkNNN2Y2yyzj48/edit?gid=0#gid=0"

# Directory where notebooks are stored
DIR = "/Workspace/Users/krista@jamcity.com/CentralizedLocalizationWorkflow/" 



# General guidelines for languages
GENERAL_LANG_SPECIFIC_GUIDELINES = {

    'French': """
        - Use informal tu-form
        - Keep phrasing smooth and playful, but avoid lengthy constructions
        - Avoid starting phrases with “Les X ont commencé !” unless space allows
        - Prefer direct imperatives: “Joue !”, “Commence !”, “Découvre !”

        Examples:
        - "Play now!" → "Joue maintenant !"
        - "Start the event!" → "Commence l’événement !" (✅ shorter than "L’événement a commencé !")""",

    "French (Canada)":""" 
        - Tone: Informal (tu), casual, and fun. 
        - Style: Playful and colloquial, with Canadian French idioms. 
        - Notes: Avoid European French phrasing; keep Quebec-specific terms (e.g., “cellulaire,” not “portable”).""",

    'Spanish (Spain)': """
        - Use informal tú-form
        - Keep copy natural, short, and punchy
        - Avoid expanded forms like “¡El evento ha comenzado!” — favor direct verbs

        Examples:
        - "Start now!" → "¡Empieza ahora!"
        - "Complete Card Sets!" → "¡Completa los conjuntos!" """,

    'Spanish': """
        - Use informal tú-form
        - Keep copy natural, short, and punchy
        - Avoid expanded forms like “¡El evento ha comenzado!” — favor direct verbs

        Examples:
        - "Start now!" → "¡Empieza ahora!"
        - "Complete Card Sets!" → "¡Completa los conjuntos!" """,


    'Spanish (Latin America)':""" 
        - Tone: Informal (tú or neutral imperatives), playful, and lively. 
        - Style: Engaging, warm, and upbeat. 
        - Notes: Use neutral LatAm vocabulary (avoid Spain-specific terms like “ordenador”). """,

    'Spanish (Colombia)': """ 
        - Tone: Informal (tú), warm, and inviting. Marketing and gaming copy in Colombia favors approachable, enthusiastic language.
        - Style: Light, fun, and lively. Regional phrasing makes the text feel more authentic (e.g., “celular” instead of “móvil”). 
        - Notes: If global Latin American Spanish (es-LA) is used elsewhere, keep es-CO consistent but allow slight flavoring for Colombian users. """,

    'Italian': """
        - Use informal tone with imperatives: Gioca, Scopri, Vinci
        - Avoid long clauses and articles unless needed for fluency
        - Aim for gamey, motivational phrases

        Examples:
        - "Worlds of Wonder has started!" → "Inizia Mondi Meravigliosi!"
        - "Get big rewards!" → "Ricevi grandi ricompense!" """,

    'Portuguese (Brazil)': """
        - Use informal você-form
        - Keep tone expressive but concise — short, motivating sentences work best
        - Prefer "Comece já!", "Jogue agora!" over longer passive forms

        Examples:
        - "The event has started!" → "Comece o evento!"
        - "Collect cards!" → "Colete cartas!" """,

    'German': """
        - Use informal du-form
        - Favor imperatives like Spiele, Entdecke, Gewinne
        - Avoid overly long noun compounds or trailing verbs where possible

        Examples:
        - "Start now!" → "Jetzt starten!"
        - "Collect Cards!" → "Sammle Karten!" """,

    'Russian': """
        - Always use formal 2nd person plural (вы), uncapitalized
        - Use polite, clear phrasing — not overly formal or bureaucratic
        - Mix imperative verbs and noun-based phrasing for balance
        - Avoid overly long descriptive constructions unless necessary

        Examples:
        - "Start now!" → "Начните сейчас!"
        - "Collect Card Sets!" → "Соберите наборы карт!"
        - Do NOT capitalize "Вы" — neutral form is standard """,

    'Turkish': """
        - Use informal ‘sen’ form
        - Keep copy short, motivational, and casual
        - Avoid wordy structures — prefer "Hemen oyna!" over passive descriptions

        Examples:
        - "Collect rewards!" → "Ödülleri topla!"
        - "Start now!" → "Hemen başla!" """,

    'Japanese': """
        - Use casual-polite phrasing (〜しよう, 〜が登場) appropriate for gacha/puzzle games
        - Avoid excessive chaining of actions; trim to 1–2 short clauses
        - Prefer dynamic verbs and game lingo like ゲット, 集めよう

        Examples:
        - “Collect cards!” → “カードを集めよう！”
        - “Win big rewards!” → “豪華報酬をゲット！” """,

    'Korean': """
        - Use casual or semi-formal tone (e.g., -요 form)
        - Avoid long sentence chains — use imperative encouragement
        - Use game terms like 보상 (rewards), 모으다 (collect), 시작하다 (start)

        Examples:
        - “Start now!” → “지금 시작하세요!”
        - “Complete card sets!” → “카드를 모아 보상 받으세요!” """,

    'Danish': """
        - Use informal, instructive tone
        - Prefer short imperatives like Fuldfør, Spil, Saml
        - Trim long constructions where possible

        Examples:
        - “Collect rewards!” → “Saml belønninger!”
        - “Start now!” → “Start nu!” """,

    'Dutch (Netherlands)':"""
        -Tone: Informal, approachable, and friendly. Use je/jij rather than formal u. 
        - Style: Casual but not slangy. Dutch players expect direct, enthusiastic language.         
        - Notes: Dutch tends to be more concise than English. Avoid over-translating gaming terms like “combo” or “level” — they are often left in English. """,

    'Finnish': """ 
        - Tone: Informal, use “sinä” casually. 
        - Style: Concise and upbeat. Add a friendly, playful feel. 
        - Notes: Avoid stiff literal translations; use natural spoken-like Finnish. """,

    'Swedish': """
        - Use informal, motivational tone
        - Keep it snappy — trim articles and helpers when unnecessary
        - Favor direct action: Spela, Samla, Få

        Examples:
        - “Start the event!” → “Starta evenemanget!”
        - “Get rewards!” → “Få belöningar!” """,

    'Norwegian': """
        - Use informal imperatives and energetic tone
        - Avoid overly long verb chains
        - Game verbs like Spill, Fullfør, Samle are effective

        Examples:
        - “Collect cards!” → “Samle kort!”
        - “Start now!” → “Start nå!” """,

    'Traditional Chinese (Taiwan)': """
        - Use clear, punchy Mandarin suitable for Taiwan audiences
        - Favor phrases like 限時, 組合, 獎勵 for game-style copy
        - Avoid excessive explanatory text

        Examples:
        - “Complete card sets!” → “完成卡牌組合！”
        - “Win rewards!” → “獲得獎勵！” """,

    'Traditional Chinese (Hong Kong)': """
        - Tone: Friendly, casual, slightly cheeky. Use Cantonese-flavored word choice where appropriate, but not slang-heavy. 
        - Style: Playful and fun. 
        - Notes: Ensure HK-traditional characters, not Taiwan terms. """,
    
    'Simplified Chinese': """
        - Use short, directive action phrases
        - Common terms include 限时, 收集, 获得, 卡组
        - Avoid overly literal or long sentence chaining

        Examples:
        - “Start now!” → “马上开始！”
        - “Collect cards for rewards!” → “收集卡牌赢奖励！” """,

    'Arabic': """
        - Use standard Modern Standard Arabic (MSA) with polite but energetic tone
        - Avoid overly verbose sentence chaining
        - Prefer clear imperatives for CTAs (ابدأ، أكمل، اربح)
        - Keep messages compact and avoid repetition

        Examples:
        - “Start now!” → “ابدأ الآن!”
        - “Complete sets for rewards!” → “أكمل المجموعات لتحصل على المكافآت!” """,

    'Turkish': """
        - Use casual, friendly tone suitable for mobile games
        - Favor imperative mood verbs like Başla, Tamamla, Topla
        - Avoid repetitive phrasing like “tamamlamaya devam et” when shorter forms suffice
        - Keep translations short and visually punchy

        Examples:
        - “Complete Card Sets!” → “Kart Setlerini Tamamla!”
        - “Win big rewards!” → “Büyük Ödülleri Kazan!” """,
    
    'Thai': """ 
        -Tone: Polite but casual. Use ครับ/ค่ะ endings where appropriate, but keep it fun. 
        - Style: Playful, light, and friendly.  
        - Notes: Avoid stiff, overly formal language. """,

    'Vietnamese': """
        -Tone: Polite-casual. Use informal, friendly phrasing but keep pronouns respectful (bạn is the safest default). Avoid overly formal or bureaucratic Vietnamese. 
        - Style: Concise, energetic, and engaging. Overly long sentences should be avoided, as they read awkwardly in mobile UI. 
        - Notes: Vietnamese is non-inflected for politeness like Japanese/Korean, but pronoun choice matters. Stick with bạn as the player address form in games. """,

    'Filipino': """ 
        -Tone: Informal, friendly, and fun. 
        - Style: Casual, lively, and warm. Mixing English with Filipino is common in games (Taglish). 
        - Notes: Favor modern, playful terms.""",

    'Indonesian': """ 
        - Tone: Informal (kamu), warm, and casual. 
        - Style: Fun, lively, and conversational. 
        - Notes: Avoid heavy formality (e.g., Anda); use natural spoken-like Indonesian. """,

    'Malay': """ 
        - Tone: Informal, casual, and direct. 
        - Style: Light and friendly. 
        - Notes: Avoid overly formal or literary Malay; stick to modern, simple phrasing.""",

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
    "Indonesian": "id"
}