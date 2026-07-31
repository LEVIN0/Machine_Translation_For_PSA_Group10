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
- Gold text is kept VERBATIM (including [Tag] prefixes) — the tag is
  recorded in Metadata.topic, never stripped from the text. The ONE
  exception: mojibake (UTF-8 misread as Windows-1252, in the issued file
  sometimes three rounds deep, e.g. "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“" for an en-dash)
  is repaired on import via cleaning.repair_mojibake — 173 English,
  79 Kiswahili, 13 Ekegusii, 1 Dholuo and 1 Somali rows are affected.
- Dedupe within the file on normalized English (keep first).
- Rows with empty English are skipped with a warning.
"""

import re
from pathlib import Path

import pandas as pd

from ..cleaning import normalize_text, repair_mojibake
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
    n_repaired = 0
    for i, row in df.iterrows():
        english = repair_mojibake((row.get("English") or "").strip())
        if not english:
            print(f"[lecturer] WARNING: row {i + 2} has empty English; skipped.")
            n_skipped += 1
            continue
        key = normalize_text(english).lower()
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)

        kiswahili = repair_mojibake((row.get("Kiswahili") or "").strip())
        ekegusii = repair_mojibake((row.get("Ekegusii") or "").strip())
        dholuo = repair_mojibake((row.get("Dholuo") or "").strip())
        somali = repair_mojibake((row.get("Somali") or "").strip())
        n_repaired += sum(
            repair_mojibake((row.get(c) or "")) != (row.get(c) or "")
            for c in ("English", "Kiswahili", "Ekegusii", "Dholuo", "Somali"))

        metadata = {
            "type": "gold",
            "license": "lecturer-provided",
            "psa_class": "PSA",
            "lecturer_id": (row.get("PSA_Id") or "").strip(),
        }
        if dholuo:
            metadata["dholuo"] = dholuo
        if somali:
            metadata["somali"] = somali
        tag = _TAG_RE.match(english)
        if tag:
            metadata["topic"] = tag.group(1).strip()

        rec = new_record(
            domain=(row.get("Domain") or "").strip(),
            english=english,  # mojibake-repaired; otherwise verbatim, [Tag] kept
            kiswahili=kiswahili,
            ekegusii=ekegusii,
            source=_SOURCE,
            url="",
            metadata=metadata,
            status="Validated",
        )
        rec["Date"] = ""  # gold rows carry no scrape/publication date
        records.append(rec)

    if verbose:
        print(f"[lecturer] imported {len(records)} gold rows from {csv_path.name} "
              f"({n_dupes} internal dupes collapsed, {n_skipped} empty skipped, "
              f"{n_repaired} mojibake cells repaired)")
    return records
