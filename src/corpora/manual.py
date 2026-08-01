"""Team-written manual PSA importer (Week 2 topic expansion).

Reads every `*.csv` in `data/manual/` with columns
`Domain,English,Kiswahili,Ekegusii,Notes` (only Domain and English are
required; the team fills Kiswahili — and Ekegusii when a speaker is
available — in later passes).

Files named `*_template.csv` (e.g. the shipped `team_psas_template.csv`)
are SKIPPED: they are worked examples for the team, not submissions. Team
members save real submissions as `data/manual/<name>.csv`.

Each valid row becomes a schema record with Source "Team-written",
metadata {"type": "manual", "license": "original work"}, Status "Pending".
"""

import csv
from pathlib import Path

from .. import config
from ..config import DOMAINS
from ..schema import new_record

_SOURCE = "Team-written"
_REQUIRED = ("Domain", "English")
_OPTIONAL = ("Kiswahili", "Ekegusii", "Notes")


def _default_manual_dir():
    """data/manual/ next to data/external/ (resolved at call time)."""
    return Path(config.EXTERNAL_DIR).parent / "manual"


def import_manual(manual_dir=None, verbose=True):
    """Import team-written PSA CSVs from data/manual/.

    Returns a list of schema records ([] with a printed warning when the
    directory or files are missing — manual data is optional by design).
    Rows with an invalid domain or empty English are skipped with a warning.
    """
    manual_dir = Path(manual_dir) if manual_dir else _default_manual_dir()
    if not manual_dir.is_dir():
        if verbose:
            print(f"[manual] WARNING: {manual_dir} not found; "
                  "no team-written rows imported.")
        return []

    csv_files = [p for p in sorted(manual_dir.glob("*.csv"))
                 if not p.stem.endswith("_template")]
    if not csv_files:
        if verbose:
            print(f"[manual] WARNING: no submission CSVs in {manual_dir} "
                  "(template files are skipped); nothing imported.")
        return []

    records = []
    skipped = 0
    for path in csv_files:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            missing_cols = [c for c in _REQUIRED
                            if c not in (reader.fieldnames or [])]
            if missing_cols:
                print(f"[manual] WARNING: {path.name} missing required "
                      f"columns {missing_cols}; file skipped.")
                continue
            for lineno, row in enumerate(reader, start=2):
                domain = (row.get("Domain") or "").strip()
                english = " ".join((row.get("English") or "").split())
                if not english:
                    skipped += 1
                    continue  # silently skip blank filler lines
                if domain not in DOMAINS:
                    print(f"[manual] WARNING: {path.name}:{lineno} invalid "
                          f"domain '{domain}' (allowed: {DOMAINS}); row skipped.")
                    skipped += 1
                    continue
                records.append(new_record(
                    domain=domain,
                    english=english,
                    kiswahili=" ".join((row.get("Kiswahili") or "").split()),
                    ekegusii=" ".join((row.get("Ekegusii") or "").split()),
                    source=_SOURCE,
                    url="",
                    metadata={"type": "manual", "license": "original work"},
                    status="Pending",
                ))
    if verbose:
        print(f"[manual] imported {len(records)} team-written rows "
              f"from {len(csv_files)} file(s) ({skipped} rows skipped)")
    return records
