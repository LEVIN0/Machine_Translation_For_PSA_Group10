"""Cleaning pipeline: normalize, length-filter, dedupe, language-check, validate.

clean() steps (each stage counted in the returned stats dict):
  1. drop rows with empty English
  2. word-count filter (MIN_WORDS..MAX_WORDS on English)
  3. dedupe (exact on normalized English, then near-dup on first 8 words)
  4. English rows of >=10 words must langdetect as "en" (else drop)
  5. rows with non-empty Kiswahili must langdetect as "sw"
     (else the Kiswahili cell is cleared and the row is kept)
  6. domain validity (invalid domains are dropped)
"""

import re

import pandas as pd

from .config import DOMAINS, MAX_WORDS, MIN_WORDS

try:
    from langdetect import DetectorFactory
    from langdetect import LangDetectException
    from langdetect import detect as _ld_detect

    # langdetect is non-deterministic unless seeded; pin it so cleaning
    # results (and tests) are reproducible run-to-run.
    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover - dependency listed in requirements
    _ld_detect = None
    LangDetectException = Exception


def normalize_text(s):
    """Collapse all whitespace runs and strip."""
    return " ".join(str(s or "").split()).strip()


def dedupe(df):
    """Remove exact then near duplicates; return (cleaned_df, n_removed).

    Exact key: normalized lowercase English. Near-dup key: first 8 normalized
    lowercase words (catches re-scrapes with trailing edits).
    """
    df = df.copy()
    df["_key_exact"] = df["English"].map(lambda s: normalize_text(s).lower())
    before = len(df)
    df = df.drop_duplicates(subset="_key_exact", keep="first")
    df["_key_near"] = df["_key_exact"].map(lambda s: " ".join(s.split()[:8]))
    df = df.drop_duplicates(subset="_key_near", keep="first")
    df = df.drop(columns=["_key_exact", "_key_near"])
    return df, before - len(df)


def detect_lang(text):
    """Detect language of text via langdetect; None on error or short input."""
    if _ld_detect is None:
        return None
    text = normalize_text(text)
    if len(text) < 10:
        return None
    try:
        return _ld_detect(text)
    except LangDetectException:
        return None
    except Exception:
        return None


# Leading list/bullet markers left over from sentence-splitting scraped HTML.
_BULLET_RE = re.compile(r"^[\s•‣◦\-\u2013\u2014\*\+]+")


def strip_bullet(s):
    """Remove leading bullet/list markers (e.g. '• ', '- ', '– ') from text."""
    return _BULLET_RE.sub("", s or "").strip()


def is_dangling_intro(s):
    """True for fragments that introduce a list that was split away
    (e.g. 'The levy shall be collected as follows: -' or '...as follows:')."""
    t = (s or "").rstrip().rstrip("-–—•*").rstrip()
    return t.endswith(":")


def clean(df):
    """Run the full cleaning pipeline; return (cleaned_df, stats dict)."""
    stats = {"input": len(df)}
    df = df.copy()

    # Normalize English/Kiswahili text up front (incl. bullet stripping).
    df["English"] = df["English"].map(normalize_text).map(strip_bullet)
    df["Kiswahili"] = df["Kiswahili"].fillna("").map(normalize_text).map(strip_bullet)

    # 1. drop empty English
    df = df[df["English"] != ""]
    stats["after_empty"] = len(df)

    # 1b. drop dangling list-intro fragments (e.g. ending ':' or ': -')
    # (astype(bool): on an EMPTY frame, ~ on an object mask is treated as
    #  column selection by pandas and wipes the DataFrame)
    df = df[~df["English"].map(is_dangling_intro).astype(bool)]
    stats["after_fragments"] = len(df)

    # 2. word-count filter
    n_words = df["English"].map(lambda s: len(s.split()))
    df = df[(n_words >= MIN_WORDS) & (n_words <= MAX_WORDS)]
    stats["after_length"] = len(df)

    # 3. dedupe (exact + near)
    df, _ = dedupe(df)
    stats["after_dedupe"] = len(df)

    # 4. long English rows must detect as English
    def _english_ok(row):
        text = row["English"]
        if len(text.split()) < 10:
            return True  # too short for reliable detection; keep
        lang = detect_lang(text)
        return lang is None or lang == "en"
    df = df[df.apply(_english_ok, axis=1)]

    # 5. non-empty Kiswahili must detect as Kiswahili; else clear, keep row
    def _swahili_checked(row):
        sw = row["Kiswahili"]
        if not sw:
            return row
        lang = detect_lang(sw)
        if lang is not None and lang != "sw":
            row["Kiswahili"] = ""  # unreliable translation cell; keep the row
        return row
    df = df.apply(_swahili_checked, axis=1)
    stats["after_lang"] = len(df)

    # 6. domain validity
    df = df[df["Domain"].isin(DOMAINS)]

    df = df.reset_index(drop=True)
    stats["output"] = len(df)
    stats["removed_total"] = stats["input"] - stats["output"]
    return df, stats
