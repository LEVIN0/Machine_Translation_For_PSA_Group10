"""Dataset schema — single source of truth for column order and record shape.

Schema (exact order, used everywhere: collectors, corpora importers, CSV, report):
    PSA_ID, Domain, English, Kiswahili, Ekegusii, Source, Date, URL, Metadata, Status

- PSA_ID:   "PSA" + 6 digits, assigned AFTER merge+clean (PSA000001, ...).
- Domain:   one of config.DOMAINS (Health, Education, Agriculture, Security,
            Governance).
- Kiswahili: filled for corpus rows (TICO-19) and bilingual scrapes; else "".
- Ekegusii: always "" in Week 1 (documented placeholder; Week 3 transfer target).
- Date:     ISO YYYY-MM-DD; scrape date if publication date unknown.
- Metadata: JSON string, e.g. {"type": "scraped", "tool": "requests+bs4"}.
- Status:   "Pending" (native-speaker validation happens in Week 2).
"""

import json
from datetime import date

import pandas as pd

from .config import DOMAINS

COLUMNS = [
    "PSA_ID",
    "Domain",
    "English",
    "Kiswahili",
    "Ekegusii",
    "Source",
    "Date",
    "URL",
    "Metadata",
    "Status",
]


def new_record(domain, english, kiswahili="", ekegusii="", source="",
               date=None, url="", metadata=None, status="Pending"):
    """Create one schema-conformant record dict.

    `metadata` (dict) is serialized to a JSON string; `date` defaults to
    today's ISO date when not provided.
    """
    return {
        "PSA_ID": "",  # assigned later by assign_ids(), after merge+clean
        "Domain": domain,
        "English": english,
        "Kiswahili": kiswahili,
        "Ekegusii": ekegusii,
        "Source": source,
        "Date": date or date_today(),
        "URL": url,
        "Metadata": json.dumps(metadata or {}, ensure_ascii=False),
        "Status": status,
    }


def date_today():
    """Return today's date as an ISO YYYY-MM-DD string."""
    return date.today().isoformat()


def assign_ids(df, prefix="PSA", width=6):
    """Assign sequential PSA_IDs (PSA000001, ...) in stable row order."""
    df = df.copy().reset_index(drop=True)
    df["PSA_ID"] = [f"{prefix}{i + 1:0{width}d}" for i in range(len(df))]
    return df[COLUMNS]


def validate_schema(df):
    """Return a list of schema problems; empty list means the frame is valid.

    Checks: exact columns present, domains valid, English non-empty.
    """
    problems = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems  # further checks would be meaningless

    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        problems.append(f"unexpected columns: {extra}")

    bad_domains = sorted(set(df["Domain"]) - set(DOMAINS))
    if bad_domains:
        problems.append(f"invalid domains: {bad_domains} (allowed: {DOMAINS})")

    n_empty_en = int((df["English"].fillna("").str.strip() == "").sum())
    if n_empty_en:
        problems.append(f"{n_empty_en} rows with empty English text")

    return problems
