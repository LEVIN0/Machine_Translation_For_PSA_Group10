#!/usr/bin/env python3
"""Week 4 augmentation tests — pure functions only, no torch/datasets.

Covers training/augment.py's row selection (which train rows get
back-translated per target) and the empty/copy-through rejection filter
that keeps the model's known failure modes out of the synthetic data.

Run from the project root:  python tests/test_week4_augment.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.augment import _usable, select_backtranslation_rows  # noqa: E402


def _df() -> pd.DataFrame:
    return pd.DataFrame([
        # English-only row: eligible for both swa and guz back-translation
        {"PSA_ID": "A", "Domain": "Health", "English": "Wash hands.",
         "Kiswahili": "", "Ekegusii": ""},
        # EN-SW row without Ekegusii: eligible for guz only
        {"PSA_ID": "B", "Domain": "Health", "English": "Get vaccinated.",
         "Kiswahili": "Pata chanjo.", "Ekegusii": ""},
        # Fully translated row: eligible for neither
        {"PSA_ID": "C", "Domain": "Health", "English": "Boil water.",
         "Kiswahili": "Chemsha maji.", "Ekegusii": "Ruga amasi."},
        # Empty English: never eligible
        {"PSA_ID": "D", "Domain": "Health", "English": "",
         "Kiswahili": "", "Ekegusii": ""},
    ])


def test_select_swa_picks_english_only_rows():
    picked = select_backtranslation_rows(_df(), "swa")
    assert set(picked["PSA_ID"]) == {"A"}


def test_select_guz_picks_all_rows_lacking_ekegusii():
    picked = select_backtranslation_rows(_df(), "guz")
    assert set(picked["PSA_ID"]) == {"A", "B"}


def test_select_max_rows_caps_and_zero_means_all():
    df = pd.concat([_df()] * 3, ignore_index=True)  # 6 guz-eligible rows
    assert len(select_backtranslation_rows(df, "guz", max_rows=4)) == 4
    assert len(select_backtranslation_rows(df, "guz", max_rows=0)) == 6


def test_select_is_seeded_and_deterministic():
    a = select_backtranslation_rows(_df(), "guz", seed=7)
    b = select_backtranslation_rows(_df(), "guz", seed=7)
    assert list(a["PSA_ID"]) == list(b["PSA_ID"])


def test_usable_rejects_empty_and_copythrough():
    assert _usable("  ", "Wash hands.") is False
    assert _usable("Wash hands.", "Wash hands.") is False
    assert _usable("wash hands", "Wash hands.") is False  # case/punct-insensitive
    assert _usable("Osa amaboko.", "Wash hands.") is True


def run() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"ok  {len(tests)} Week 4 augmentation tests")


if __name__ == "__main__":
    run()
