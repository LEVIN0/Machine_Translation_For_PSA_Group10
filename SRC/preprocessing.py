"""Week 2 preprocessing: deep normalization, tokenization, code-switching,
and cultural-term glossary tagging.

This module complements (does not replace) `SRC/cleaning.py`: cleaning
decides which rows survive; preprocessing annotates surviving rows with
model-ready columns (`English_norm`, `Kiswahili_norm`, token counts,
code-switch flags, glossary hits) WITHOUT altering the original columns.
"""

import json
import re
import unicodedata

from . import config

_WORD_RE = re.compile(r"[a-zà-ÿ]+(?:'[a-z]+)?")
# Word tokens plus each punctuation mark as its own token.
_TOKEN_RE = re.compile(r"[a-zà-ÿ]+(?:'[a-z]+)?|[^\w\s]", re.UNICODE)

# --- Stopwords (hand-written, ~40 per language) ------------------------------
# Used for code-switch detection and available for EDA; intentionally small
# and function-word heavy so they generalize across domains.
EN_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "you", "your", "we",
    "our", "they", "their", "he", "she", "his", "her", "not", "no", "as",
    "if", "when", "will", "can", "all", "has", "have", "had",
})

SW_STOPWORDS = frozenset({
    "na", "ya", "wa", "kwa", "katika", "au", "ama", "lakini", "ni", "si",
    "za", "la", "cha", "vya", "huu", "hii", "hilo", "hiyo", "wewe", "yako",
    "sisi", "yetu", "wao", "yao", "yeye", "wake", "kuwa", "kama", "sasa",
    "hapa", "pia", "kabla", "baada", "ndani", "juu", "chini", "kati",
    "hadi", "mpaka", "kila", "yote", "wote", "mtu", "watu", "mno",
})


def normalize_deep(text):
    """Deep unicode/punctuation normalization (beyond cleaning.normalize_text).

    Steps: unicode NFC, curly quotes/apostrophes -> ASCII (' \"),
    en/em dashes -> '-', collapse all whitespace runs, strip.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.replace("\u2018", "'").replace("\u2019", "'")   # ' '
    s = s.replace("\u201c", '"').replace("\u201d", '"')   # " "
    s = s.replace("\u2013", "-").replace("\u2014", "-")   # – —
    s = " ".join(s.split())
    return s.strip()


def tokenize(text, lang="en"):
    """Lowercase regex tokenizer: word tokens + separate punctuation tokens.

    Words match `[a-zà-ÿ]+(?:'[a-z]+)?` (accented Latin letters included,
    contractions kept together). Each punctuation mark is its own token.

    `lang` is accepted for interface stability but the tokenizer is
    currently language-agnostic (works for both EN and SW, which share the
    Latin script). NOTE: model subword tokenization (SentencePiece/BPE) is a
    Week-3 concern and will wrap, not replace, this function.
    """
    return _TOKEN_RE.findall(normalize_deep(text).lower())


def word_tokens(text, lang="en"):
    """Word tokens only (punctuation dropped) — used for counts and stats."""
    return _WORD_RE.findall(normalize_deep(text).lower())


def codeswitch_ratio(text):
    """Fraction of word tokens matching the NON-dominant language's stopwords.

    The dominant language is whichever stopword list (EN or SW) matches more
    tokens; the ratio is non_dominant_hits / total_word_tokens. A monolingual
    text scores ~0; a heavily mixed text approaches 0.5. Empty text -> 0.0.
    """
    words = word_tokens(text)
    if not words:
        return 0.0
    en_hits = sum(1 for w in words if w in EN_STOPWORDS)
    sw_hits = sum(1 for w in words if w in SW_STOPWORDS)
    dominant, other = (en_hits, sw_hits) if en_hits >= sw_hits else (sw_hits, en_hits)
    return other / len(words)


def is_codeswitched(text, threshold=0.15):
    """True when the non-dominant-language stopword share exceeds `threshold`."""
    return codeswitch_ratio(text) > threshold


def load_glossary(path=None):
    """Load the Kenyan cultural/institutional glossary (data/glossary.json).

    Returns a dict: {"term": {"sw": ..., "note": ...}}. Missing file ->
    empty dict with a printed warning (glossary tagging is non-fatal).
    """
    path = path or (config.BASE_DIR / "data" / "glossary.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"[preprocessing] WARNING: glossary not found at {path}; "
              "glossary tagging disabled.")
        return {}


def glossary_hits(text, glossary):
    """Return glossary terms found in `text` (case-insensitive whole-word).

    Whole-word here means not bounded by word characters, so hyphenated
    terms ("M-Pesa") and multi-word terms ("boda boda") also match.
    """
    if not glossary or not text:
        return []
    hits = []
    for term in glossary:
        pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                             re.IGNORECASE)
        if pattern.search(str(text)):
            hits.append(term)
    return hits


def preprocess_dataframe(df, glossary=None):
    """Annotate a schema dataframe with Week 2 preprocessing columns.

    Adds: `English_norm`, `Kiswahili_norm` (normalize_deep),
    `tokens_en`, `tokens_sw` (word-token counts, int),
    `codeswitch` (bool, on the English text),
    `glossary_terms` (comma-joined glossary hits on the English text).

    The original schema columns are NOT altered. `glossary` defaults to
    load_glossary() when not provided.
    """
    if glossary is None:
        glossary = load_glossary()
    df = df.copy()
    df["English_norm"] = df["English"].map(normalize_deep)
    df["Kiswahili_norm"] = df["Kiswahili"].fillna("").map(normalize_deep)
    df["tokens_en"] = df["English_norm"].map(lambda s: len(word_tokens(s))).astype(int)
    df["tokens_sw"] = df["Kiswahili_norm"].map(lambda s: len(word_tokens(s))).astype(int)
    df["codeswitch"] = df["English_norm"].map(is_codeswitched).astype(bool)
    df["glossary_terms"] = df["English_norm"].map(
        lambda s: ", ".join(glossary_hits(s, glossary)))
    return df
