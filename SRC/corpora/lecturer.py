"""Lecturer gold dataset loader (SPEC_REMEDIATION.md §2).

Reads ``PSA_KE_Final.csv`` — the lecturer-provided gold dataset (all rows
Class=PSA) with columns
``PSA_Id,Domain,Class,English,Kiswahili,Ekegusii,Dholuo,Somali`` and English
text prefixed with a ``[Topic Tag]``.

Each row becomes a schema record with Source "Lecturer dataset
(PSA_KE_Final)", Status "Validated", and Metadata
``{"type": "gold", "license": "lecturer-provided", "psa_class": "PSA",
"lecturer_id": <PSA_Id>, "dholuo": <...>, "somali": <...>, "topic": <tag>}``
(dholuo/somali keys omitted when empty; topic omitted when no [Tag] prefix).

Non-negotiables:
- Gold text is kept VERBATIM (including [Tag] prefixes and any mojibake) —
  the tag is recorded in Metadata.topic, never stripped from the text.
- Dedupe within the file on normalized English (keep first).
- Rows with empty English are skipped with a warning.
"""

import re
from pathlib import Path

import pandas as pd

from ..cleaning import normalize_text
from ..schema import new_record

_SOURCE = "Lecturer dataset (PSA_KE_Final)"
_TAG_RE = re.compile(r"^\s*\[([^\]]+)\]")


def load_lecturer(csv_path: Path, verbose=True) -> list[dict]:
    """Load the lecturer gold CSV as schema records.

    Returns a list of schema records ([] with a printed warning when the
    file is missing — the gold dataset is optional by design).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        if verbose:
            print(f"[lecturer] WARNING: {csv_path} not found; "
                  "no lecturer gold rows imported.")
        return []

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")

    records = []
    seen = set()
    n_dupes = 0
    n_skipped = 0
    for i, row in df.iterrows():
        english = (row.get("English") or "").strip()
        if not english:
            print(f"[lecturer] WARNING: row {i + 2} has empty English; skipped.")
            n_skipped += 1
            continue
        key = normalize_text(english).lower()
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)

        metadata = {
            "type": "gold",
            "license": "lecturer-provided",
            "psa_class": "PSA",
            "lecturer_id": (row.get("PSA_Id") or "").strip(),
        }
        dholuo = (row.get("Dholuo") or "").strip()
        somali = (row.get("Somali") or "").strip()
        if dholuo:
            metadata["dholuo"] = dholuo
        if somali:
            metadata["somali"] = somali
        tag = _TAG_RE.match(english)
        if tag:
            metadata["topic"] = tag.group(1).strip()

        rec = new_record(
            domain=(row.get("Domain") or "").strip(),
            english=english,  # verbatim, [Tag] prefix kept
            kiswahili=(row.get("Kiswahili") or "").strip(),
            ekegusii=(row.get("Ekegusii") or "").strip(),
            source=_SOURCE,
            url="",
            metadata=metadata,
            status="Validated",
        )
        rec["Date"] = ""  # gold rows carry no scrape/publication date
        records.append(rec)

    if verbose:
        print(f"[lecturer] imported {len(records)} gold rows from {csv_path.name} "
              f"({n_dupes} internal dupes collapsed, {n_skipped} empty skipped)")
    return records
