FUN_DESCRIPTION_PROMPT = f""" You are localizing item flavor text for a mobile game called Disney Magic Match.

Your task is to write a fun, punchy, Disney-appropriate item description in the TARGET language for each row.

You will receive structured preprocessing fields that describe the literal item:
- KEY
- object
- descriptor
- additional_context
- simple_description_en
- target_language

Your goal is to transform the literal item information into a short, charming, playful in-game description in the TARGET language.

Style goals:
- Cute, whimsical, playful
- Disney-appropriate in tone
- Natural and idiomatic in the TARGET language
- Short and punchy
- Suitable for a casual mobile game
- Focused on the item itself
- Can be lively, warm, magical, cheeky, or charming

Important:
- Do NOT produce a literal translation of the English.
- Do NOT explain the item mechanically unless needed.
- Do NOT sound technical, dry, or encyclopedic.
- The line should feel like item flavor text written for players.
- The object should remain clear.
- Use the structured fields to understand the item, not to rigidly repeat them.
- You may use playful wording, mild humor, charm, or light personality.
- Do not force a pun if it sounds unnatural in the TARGET language.
- Prefer a native-sounding line over preserving any English phrasing.
- Avoid references that are culturally awkward or hard to understand in the TARGET language.
- Keep the line concise.

Output requirements:
- Return exactly one JSON object.
- Include the KEY exactly as provided.
- Include one field:
  - "localized_fun_description"

Additional style constraints:
- Prefer 1 sentence only.
- Avoid exceeding 12 words unless necessary.
- Avoid repetitive openings across rows.
- Vary punctuation and sentence structure.
- Avoid ending every line with an exclamation mark.
- Keep the object as the center of the line.
- Prefer delight over complexity.

Follow these specific language guidelines: 
"""



QUIP_PROMPT = f""" You are localizing humorous item quips for a mobile game called Disney Magic Match.

Your task is to write a short, witty, playful quip in the TARGET language for each item.

You will receive:
- KEY
- object
- descriptor
- additional_context
- target_language

Your goal is to create a humorous line that feels naturally written in the TARGET language.

Style goals:
- Short, punchy, playful
- Funny or witty in a way that feels native to the TARGET language
- Disney-appropriate
- Centered on the object itself
- Can use wordplay, sound play, idioms, double meanings, exaggeration, personification, or playful phrasing
- Should feel like charming in-game flavor text, not stand-up comedy

Important:
- Do NOT translate English jokes directly unless they sound natural in the TARGET language.
- Prefer native humor patterns over preserving source wording.
- The object should be the center of the joke.
- The line may personify the object or give it a playful attitude.
- Use puns or wordplay only when they sound natural.
- If strong wordplay would sound forced, write a witty or charming quip instead.
- Avoid humor that feels sarcastic, mean, rude, or too adult.
- Avoid culture-specific references unless clearly appropriate for the TARGET language audience.
- Keep it concise and game-friendly.
- Vary sentence structure and punctuation naturally.

Output requirements:
- Return exactly one JSON object.
- Include the KEY exactly as provided.
- Include one field:
  - "localized_humorous_quip"


Follow these specific language guidelines: 
"""


FUN_DESCRIPTIONS_LANGUAGE_GUIDE = {
    "French": """
            Tone and style:
            - Light, elegant, slightly poetic
            - Warm and whimsical, but not overly childish
            - Avoid overly literal phrasing

            Writing patterns:
            - Prefer smooth, flowing sentences
            - Sentence fragments are acceptable if stylish
            - Alliteration and rhythm work very well
            - Personification is natural and encouraged

            Humor and charm:
            - Light wordplay is good, but subtle
            - Prefer charm over obvious jokes
            - Idioms can work if simple and widely understood

            Punctuation:
            - Use exclamation marks sparingly
            - Ellipses (…) and commas work well for softness

            Avoid:
            - Direct translations of English puns
            - Overly goofy or slapstick tone
        """,
    "Spanish (Latin America)": """ 
            Tone and style:
            - Friendly, upbeat, playful
            - Slightly energetic and expressive

            Writing patterns:
            - Short, clear sentences
            - Exclamations are more acceptable than in French
            - Personification works well

            Humor and charm:
            - Light humor is good
            - Simple wordplay works; avoid complex puns
            - Use approachable, conversational tone

            Punctuation:
            - Occasional exclamation marks are fine
            - Keep rhythm lively

            Avoid:
            - Overly complex phrasing
            - Regional slang
        """,
    "Portuguese (Brazil)":"""
            Tone and style:
            - Warm, cheerful, slightly playful
            - Friendly and approachable

            Writing patterns:
            - Smooth, natural phrasing
            - Slight rhythm or musicality works well
            - Personification is common and effective

            Humor and charm:
            - Light humor and warmth preferred over heavy puns
            - Wordplay should feel effortless

            Punctuation:
            - Moderate use of exclamation marks
            - Avoid overly flat tone

            Avoid:
            - Overly formal tone
            - Literal English structures
            """,
    "Italian":"""
        Tone and style:
        - Expressive, lively, slightly dramatic
        - Warm and playful

        Writing patterns:
        - Smooth, flowing sentences
        - Musical rhythm is important
        - Personification works very well

        Humor and charm:
        - Light humor and expressive charm preferred
        - Wordplay is good but should feel natural

        Punctuation:
        - Exclamation marks acceptable but not overused
        - Rhythmic phrasing encouraged

        Avoid:
        - Flat or overly neutral tone
        """,
    "German":"""
        Tone and style:
        - Clean, clever, slightly witty
        - Less “cute”, more clever charm

        Writing patterns:
        - Compact, efficient phrasing
        - Compound words can be used creatively
        - Sentence fragments work well

        Humor and charm:
        - Clever wordplay and precision are preferred
        - Subtle humor > silly humor
        - Personification is acceptable but restrained

        Punctuation:
        - Minimal exclamation marks
        - Prefer clean statements

        Avoid:
        - Overly childish tone
        - Overly long sentences
        """,
    "Korean":"""
        Tone and style:
        - Cute, friendly, slightly playful
        - Light and approachable

        Writing patterns:
        - Short, natural phrases
        - Sentence endings carry tone (very important)
        - Fragments are acceptable

        Humor and charm:
        - Avoid heavy puns
        - Prefer tone-based humor, cuteness, or light exaggeration
        - Personification works well

        Punctuation:
        - Minimal punctuation
        - Avoid excessive exclamation marks

        Avoid:
        - Literal translations of wordplay
        - Overly complex humor
        """,
    "Japanese":"""
        Tone and style:
        - Cute, soft, slightly magical
        - Friendly and light

        Writing patterns:
        - Short phrases preferred
        - Sentence fragments are very natural
        - Often omit explicit subject

        Humor and charm:
        - Avoid complex puns
        - Prefer cute phrasing, tone particles, or gentle playfulness
        - Personification works well

        Punctuation:
        - Minimal punctuation
        - Use ellipses or soft endings when needed

        Avoid:
        - Direct translation of Western humor
        - Overly complex sentences
        """,
}


QUIP_LANGUAGE_GUIDE = {
    "French": """
           Native humor patterns:
            - Wordplay based on sound, rhyme, or double meaning
            - Light absurdity or poetic twists

            Puns:
            - Common and effective, but must feel natural

            Best quip shapes:
            - Object-led statement
            - Short witty observation
            - Playful twist on expectation

            Tone:
            - Clever, slightly ironic, elegant

            Avoid:
            - Forced English-style puns
            - Overly childish humor
        """,
    "Spanish (Latin America)": """ 
            Native humor patterns:
            - Light wordplay, repetition, rhythm
            - Friendly exaggeration

            Puns:
            - Moderate — keep simple

            Best quip shapes:
            - Playful statement
            - Mini punchline
            - Slight exaggeration

            Tone:
            - Energetic, cheerful

            Avoid:
            - Complex or hard-to-parse wordplay
        """,
    "Portuguese (Brazil)":"""
            Native humor patterns:
            - Playful tone, rhythm, light exaggeration

            Puns:
            - Moderate — should feel natural

            Best quip shapes:
            - Cheerful observation
            - Light wordplay
            - Friendly joke

            Tone:
            - Warm, lively, slightly cheeky

            Avoid:
            - Dry or overly clever humor
            """,
    "Italian":"""
        Native humor patterns:
        - Expressive phrasing
        - Slight exaggeration

        Puns:
        - Moderate — should feel fluid

        Best quip shapes:
        - Dramatic mini-line
        - Rhythmic phrasing
        - Playful exaggeration

        Tone:
        - Warm, expressive, theatrical

        Avoid:
        - Flat or overly literal humor
        """,
    "German":"""
        Native humor patterns:
        - Clever word construction
        - Compound-based humor

        Puns:
        - Effective but should be precise

        Best quip shapes:
        - Short clever statement
        - Wordplay through compounds
        - Dry humor

        Tone:
        - Smart, understated

        Avoid:
        - Silly or overly whimsical humor
        """,
    "Korean":"""
        Native humor patterns:
        - Tone endings
        - Cute exaggeration
        - Light wordplay

        Puns:
        - Limited; must feel natural

        Best quip shapes:
        - Short playful line
        - Slight exaggeration
        - Friendly tone

        Tone:
        - Cute, approachable

        Avoid:
        - Complex or forced wordplay
        - Sarcasm
        """,
    "Japanese":"""
        Native humor patterns:
        - Cute phrasing
        - Sound-based repetition
        - Tone-based humor

        Puns:
        - Limited use; only if very natural

        Best quip shapes:
        - Short cute line
        - Object personification
        - Soft playful tone

        Tone:
        - Cute, gentle, light

        Avoid:
        - Western-style wordplay
        - Complex jokes
        """,
}



OUTPUT_FORMAT_FUN_DESCRIPTION = f""" 
              Respond in **JSON format**, one object per row:
              json
              [
              {{ "key":"key 1",
                  "localized_fun_description": "localized fun description 1"}},
              {{ "key":"key 2",
                  "localized_fun_description": "localized fun description 2"}},,
              ...
              ]\n\n
              
            
            Do not include any extra commentary or markdown.  
              
            """

OUTPUT_FORMAT_QUIP = f""" 
              Respond in **JSON format**, one object per row:
              json
              [
              {{ "key":"key 1",
                  "localized_humorous_quip": "localized humorous quip 1"}},
              {{ "key":"key 2",
                  "localized_humorous_quip": "localized humorous quip 2"}},,
              ...
              ]\n\n

            Do not include any extra commentary or markdown.
            
            """