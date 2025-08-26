### GENERAL LATER
#TODO:
##GENERAL_LANG_SPECIFIC_GUIDELINES = {"ja":"",""}

PP_LANGS = ['Japanese', 'French', 'Italian', 'German', 'Spanish (Spain)', 'Korean', 'Danish', 'Swedish', 'Norwegian', 'Traditional Chinese (Taiwan)', 'Simplified Chinese', 'Portuguese (Brazil)', 'Russian', 'Arabic', 'Turkish']

PP_LANG_CDS = ['ja', 'fr', 'it', 'de', 'es', 'ko', 'da', 'sv', 'no', 'zh-TW', 'zh-CN', 'pt-BR', 'ru', 'ar', 'tr']

PP_LANG_MAP = dict(zip(PP_LANGS, PP_LANG_CDS))

PP_LANG_SPECIFIC_GUIDELINES = {

    'French': """- Use informal tu-form
- Keep phrasing smooth and playful, but avoid lengthy constructions
- Avoid starting phrases with “Les X ont commencé !” unless space allows
- Prefer direct imperatives: “Joue !”, “Commence !”, “Découvre !”

Examples:
- "Play now!" → "Joue maintenant !"
- "Start the event!" → "Commence l’événement !" (✅ shorter than "L’événement a commencé !")""",

    'Spanish (Spain)': """- Use informal tú-form
- Keep copy natural, short, and punchy
- Avoid expanded forms like “¡El evento ha comenzado!” — favor direct verbs

Examples:
- "Start now!" → "¡Empieza ahora!"
- "Complete Card Sets!" → "¡Completa los conjuntos!" """,

    'Italian': """- Use informal tone with imperatives: Gioca, Scopri, Vinci
- Avoid long clauses and articles unless needed for fluency
- Aim for gamey, motivational phrases

Examples:
- "Worlds of Wonder has started!" → "Inizia Mondi Meravigliosi!"
- "Get big rewards!" → "Ricevi grandi ricompense!" """,

    'Portuguese (Brazil)': """- Use informal você-form
- Keep tone expressive but concise — short, motivating sentences work best
- Prefer "Comece já!", "Jogue agora!" over longer passive forms

Examples:
- "The event has started!" → "Comece o evento!"
- "Collect cards!" → "Colete cartas!" """,

    'German': """- Use informal du-form
- Favor imperatives like Spiele, Entdecke, Gewinne
- Avoid overly long noun compounds or trailing verbs where possible

Examples:
- "Start now!" → "Jetzt starten!"
- "Collect Cards!" → "Sammle Karten!" """,

    'Russian': """- Always use formal 2nd person plural (вы), uncapitalized
- Use polite, clear phrasing — not overly formal or bureaucratic
- Mix imperative verbs and noun-based phrasing for balance
- Avoid overly long descriptive constructions unless necessary

Examples:
- "Start now!" → "Начните сейчас!"
- "Collect Card Sets!" → "Соберите наборы карт!"
- Do NOT capitalize "Вы" — neutral form is standard""",

    'Turkish': """- Use informal ‘sen’ form
- Keep copy short, motivational, and casual
- Avoid wordy structures — prefer "Hemen oyna!" over passive descriptions

Examples:
- "Collect rewards!" → "Ödülleri topla!"
- "Start now!" → "Hemen başla!" """,

'Japanese': """- Use casual-polite phrasing (〜しよう, 〜が登場) appropriate for gacha/puzzle games
- Avoid excessive chaining of actions; trim to 1–2 short clauses
- Prefer dynamic verbs and game lingo like ゲット, 集めよう

Examples:
- “Collect cards!” → “カードを集めよう！”
- “Win big rewards!” → “豪華報酬をゲット！” """,

    'Korean': """- Use casual or semi-formal tone (e.g., -요 form)
- Avoid long sentence chains — use imperative encouragement
- Use game terms like 보상 (rewards), 모으다 (collect), 시작하다 (start)

Examples:
- “Start now!” → “지금 시작하세요!”
- “Complete card sets!” → “카드를 모아 보상 받으세요!” """,

    'Danish': """- Use informal, instructive tone
- Prefer short imperatives like Fuldfør, Spil, Saml
- Trim long constructions where possible

Examples:
- “Collect rewards!” → “Saml belønninger!”
- “Start now!” → “Start nu!” """,

    'Swedish': """- Use informal, motivational tone
- Keep it snappy — trim articles and helpers when unnecessary
- Favor direct action: Spela, Samla, Få

Examples:
- “Start the event!” → “Starta evenemanget!”
- “Get rewards!” → “Få belöningar!” """,

    'Norwegian': """- Use informal imperatives and energetic tone
- Avoid overly long verb chains
- Game verbs like Spill, Fullfør, Samle are effective

Examples:
- “Collect cards!” → “Samle kort!”
- “Start now!” → “Start nå!” """,

    'Traditional Chinese (Taiwan)': """- Use clear, punchy Mandarin suitable for Taiwan audiences
- Favor phrases like 限時, 組合, 獎勵 for game-style copy
- Avoid excessive explanatory text

Examples:
- “Complete card sets!” → “完成卡牌組合！”
- “Win rewards!” → “獲得獎勵！” """,

    'Simplified Chinese': """- Use short, directive action phrases
- Common terms include 限时, 收集, 获得, 卡组
- Avoid overly literal or long sentence chaining

Examples:
- “Start now!” → “马上开始！”
- “Collect cards for rewards!” → “收集卡牌赢奖励！” """,
'Arabic': """- Use standard Modern Standard Arabic (MSA) with polite but energetic tone
- Avoid overly verbose sentence chaining
- Prefer clear imperatives for CTAs (ابدأ، أكمل، اربح)
- Keep messages compact and avoid repetition

Examples:
- “Start now!” → “ابدأ الآن!”
- “Complete sets for rewards!” → “أكمل المجموعات لتحصل على المكافآت!” """,

    'Turkish': """- Use casual, friendly tone suitable for mobile games
- Favor imperative mood verbs like Başla, Tamamla, Topla
- Avoid repetitive phrasing like “tamamlamaya devam et” when shorter forms suffice
- Keep translations short and visually punchy

Examples:
- “Complete Card Sets!” → “Kart Setlerini Tamamla!”
- “Win big rewards!” → “Büyük Ödülleri Kazan!” """
}
##PP_GAME_DESCRIPTION = 
##PP_LANG_MAP =

# game specific prompt inputs 
PP_EX_INPUT = '[{"token": "detectives_card_4", "context": "Detective Panda event", "en_US": "Treasure Map"}, {"token": "pause_button", "context": "UI label, keep < 10 characters", "en_US": "Pause"}]'

PP_CONTEXT_INFER = """Some rows may include a context field. This may contain additional information such as:\n - The theme or character referenced (e.g. 'Detective Panda', 'Mama Panda') \n -UI usage hints (e.g. 'banner title', 'button label') \n - Occasional formatting or character length tips \n \n If present, use the context to guide tone, word choice, or brevity — especially when the English phrase is vague or could be interpreted multiple ways. """

PP_TOKEN_INFER = """Each row includes a token which may contain clues about the theme or in-game context (e.g. “detectives”, “event”, “deluxe”). If possible, infer the theme from the token and apply that understanding to improve the translation — especially when the English phrase is short or ambiguous."""



### CJB_EX_INPUT= ""


## CJB_CONTEXT_INFER = """ """

## CJB_TOKEN_INFER = """ """