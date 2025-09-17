import re
import pandas as pd
from collections import Counter

# non-greedy, supports multiple instances
_RE_ANGLE = re.compile(r"<[^>]*>")
_RE_BRACE = re.compile(r"\{[^}]*\}")

def extract_placeholders(text: str) -> list[str]:
    """Return ALL literal placeholders found in text, in order, duplicates allowed."""
    if not text:
        return []
    return _RE_ANGLE.findall(text) + _RE_BRACE.findall(text)

def count_tokens(tokens: list[str]) -> Counter:
    return Counter(tokens)

def enforce_placeholders_on_df(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, ensure that every placeholder found in en_US appears in translation
    EXACTLY as many times. If missing, append copies (space-separated) at the end.
    (We append rather than attempt to place semantically.)
    """
    df = long_df.copy()
    if "en_US" not in df.columns or "translation" not in df.columns:
        return df

    new_translations = []
    for _, row in df.iterrows():
        en_tokens = extract_placeholders(str(row.get("en_US", "")))
        tr = str(row.get("translation", "") or "")

        if not en_tokens:
            new_translations.append(tr)
            continue

        need = count_tokens(en_tokens)

        # Count *exact* matches in translation
        have = Counter()
        for tok in need.keys():
            # count non-overlapping exact occurrences
            have[tok] = len(re.findall(re.escape(tok), tr))

        # Determine missing copies and append them (literal, unchanged)
        pieces = [tr.rstrip()]
        for tok, n_need in need.items():
            n_have = have.get(tok, 0)
            if n_have < n_need:
                missing = n_need - n_have
                # append space + exact token, repeated
                pieces.append((" " + tok) * missing)

        new_tr = "".join(pieces) if len(pieces) > 1 else tr
        new_translations.append(new_tr)

    df["translation"] = new_translations
    return df