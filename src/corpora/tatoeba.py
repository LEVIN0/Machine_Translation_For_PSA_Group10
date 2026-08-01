"""Tatoeba importer (user-downloaded eng<->swa exports, license CC BY 2.0 FR).

Tatoeba is a general-domain sentence corpus. PSA relevance filtering is done
with a keyword heuristic over the five project domains: rows whose English text
matches no domain keyword are DROPPED (they are not PSA-relevant anyway).
This decision is documented in README.md.

Expected input under EXTERNAL_DIR/tatoeba (download from
https://tatoeba.org/en/downloads):
- sentences_detailed.csv + links.csv  (official exports), or
- any *.tsv with per-pair rows: eng_text <TAB> swa_text (2+ columns).
"""

import csv
from pathlib import Path

from ..config import EXTERNAL_DIR, MAX_WORDS, MIN_WORDS
from ..schema import new_record

_SOURCE = "Tatoeba"
_URL = "https://tatoeba.org/"

# Keyword heuristic for domain assignment (checked against lowercase English).
DOMAIN_KEYWORDS = {
    "Health": [
        "health", "doctor", "hospital", "vaccine", "vaccinat", "disease",
        "medicine", "clinic", "fever", "covid", "cholera", "malaria",
        "wash your hands", "symptom", "patient",
    ],
    "Agriculture": [
        "farm", "crop", "harvest", "seed", "fertilizer", "livestock",
        "soil", "plant", "maize", "irrigation", "pest", "drought",
    ],
    "Security": [
        "police", "security", "emergency", "fire", "accident", "road",
        "traffic", "safety", "flood", "warning", "danger", "evacuate",
        "curfew",
    ],
    "Education": [
        "school", "student", "teacher", "exam", "learn", "education",
        "class", "university", "scholarship", "read", "book",
    ],
    "Governance": [
        "government", "tax", "vote", "election", "citizen", "license",
        "permit", "register", "law", "court", "ministry", "public",
        "id card", "passport",
    ],
}


def _assign_domain(text):
    """Assign a domain by keyword heuristic; return None if no keyword matches."""
    low = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return domain
    return None  # non-matching rows are dropped (relevance filtering)


def _length_ok(text):
    """True when the text is within the project word-count bounds."""
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS


def _pair_from_official_exports(tatoeba_dir):
    """Pair eng<->swa sentences using sentences_detailed.csv + links.csv."""
    sent_path = tatoeba_dir / "sentences_detailed.csv"
    links_path = tatoeba_dir / "links.csv"
    sentences = {}  # id -> (lang, text)
    with open(sent_path, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 3 and row[1] in ("eng", "swa"):
                sentences[row[0]] = (row[1], row[2])
    pairs = []
    with open(links_path, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 2:
                continue
            a, b = sentences.get(row[0]), sentences.get(row[1])
            if not a or not b:
                continue
            if a[0] == "eng" and b[0] == "swa":
                pairs.append((a[1], b[1]))
            elif a[0] == "swa" and b[0] == "eng":
                pairs.append((b[1], a[1]))
    return pairs


def _pair_from_pair_tsv(tatoeba_dir):
    """Pair eng<->swa from any per-pair TSV (eng_text <TAB> swa_text)."""
    pairs = []
    for tsv in sorted(tatoeba_dir.glob("*.tsv")):
        with open(tsv, encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) >= 2:
                    en, sw = row[0].strip(), row[1].strip()
                    if en and sw:
                        pairs.append((en, sw))
    return pairs


def _print_instructions(tatoeba_dir):
    """Print exact download instructions for the manual Tatoeba export."""
    print(
        "[tatoeba] No exports found. To include Tatoeba data:\n"
        "  1. Go to https://tatoeba.org/en/downloads\n"
        "  2. Download 'sentences_detailed.csv' and 'links.csv'\n"
        "     (or export an eng-swa per-pair TSV: eng_text<TAB>swa_text)\n"
        f"  3. Place the files in: {tatoeba_dir}\n"
        "  License: CC BY 2.0 FR — attribution required."
    )


def import_tatoeba(verbose=True):
    """Import eng<->swa pairs from user-downloaded Tatoeba exports.

    Returns [] (with download instructions) when no export files exist.
    Rows are length-filtered and assigned a domain by keyword heuristic;
    rows matching no domain keyword are dropped as not PSA-relevant.
    """
    tatoeba_dir = Path(EXTERNAL_DIR) / "tatoeba"
    sent = tatoeba_dir / "sentences_detailed.csv"
    links = tatoeba_dir / "links.csv"

    if sent.exists() and links.exists():
        pairs = _pair_from_official_exports(tatoeba_dir)
    elif list(tatoeba_dir.glob("*.tsv")):
        pairs = _pair_from_pair_tsv(tatoeba_dir)
    else:
        if verbose:
            _print_instructions(tatoeba_dir)
        return []

    records = []
    dropped = 0
    for en, sw in pairs:
        en, sw = " ".join(en.split()), " ".join(sw.split())
        if not _length_ok(en):
            continue
        domain = _assign_domain(en)
        if domain is None:
            dropped += 1  # no domain keyword -> not PSA-relevant -> drop
            continue
        records.append(new_record(
            domain=domain,
            english=en,
            kiswahili=sw,
            source=_SOURCE,
            url=_URL,
            metadata={"type": "corpus", "corpus": "Tatoeba",
                      "license": "CC BY 2.0 FR"},
        ))
    if verbose:
        print(f"[tatoeba] imported {len(records)} pairs "
              f"({dropped} dropped by domain relevance filter)")
    return records
