#!/usr/bin/env python3
"""Tests for the FROZEN PSA framework classifier (SPEC_REMEDIATION.md §1/§5).

Plain asserts, no pytest required; offline, fast, no optional deps.
Exposes run() which prints "ok  test_psa_classify" on success.

Run from the project root:  python tests/test_psa_classify.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SRC.psa_classify import classify_frame, classify_psa  # noqa: E402

# Must-pass reference examples from SPEC_REMEDIATION.md §1.
KEPT_PSA = [
    "If you believe you may be infected, call a hospital or local emergency "
    "medical services instead of going in person to avoid infecting others.",
    "Importers are required to obtain EPR import certificates through the "
    "National Electronic Single Window System (NESWIS).",
    "The compliance deadline for this requirement was 30th June, 2025.",
    "Control fall armyworm in an environmentally friendly way using "
    "Integrated Pest Management (IPM) techniques suitable for onions.",
    "Let's continue playing our part in thwarting terror by sharing "
    "information on suspicious activities with relevant authorities.",
]
NON_PSA = [
    "However, the disease may present in many atypical forms.",
    "Chagas disease can be treated with benznidazole or nifurtimox.",
    "By 27 March, nearly 90 percent of the world's student population was "
    "out of class.",
]


def test_reference_examples():
    for text in KEPT_PSA:
        label, score = classify_psa(text)
        assert label == "PSA", f"must keep as PSA (got {label}/{score}): {text}"
        assert score >= 2
    for text in NON_PSA:
        label, score = classify_psa(text)
        assert label != "PSA", f"must NOT be PSA (got {label}/{score}): {text}"
    label, _ = classify_psa("KRA wins case against Dubai firm")
    assert label == "PressRelease", label
    label, _ = classify_psa(
        "It is notified for the general information that the tender "
        "document is available.")
    assert label == "Legal", label
    print("ok  test_reference_examples")


def test_scoring_edge_cases():
    # empty text -> Drop / -9
    assert classify_psa("") == ("Drop", -9)
    assert classify_psa("   ") == ("Drop", -9)
    assert classify_psa(None) == ("Drop", -9)

    # connective start costs exactly -1 vs the same text without it
    base = "the disease may present in many atypical forms."
    _, s_plain = classify_psa(base)
    label, s_conn = classify_psa("However, " + base)
    assert s_conn == s_plain - 1, (s_plain, s_conn)
    assert label != "PSA"

    # ENCYC penalty applies on a plain encyclopedic sentence ...
    label, s_encyc = classify_psa(
        "Cholera is a disease caused by contaminated water.")
    assert s_encyc < 0 and label == "Informational", (label, s_encyc)
    # ... but is suppressed when strong PSA signals are present (score>=2
    # before applying): "wash your" (+2) + imperative start (+2) = 4,
    # so "is a disease" must NOT subtract.
    label, score = classify_psa(
        "Wash your hands because cholera is a disease caused by dirty water.")
    assert label == "PSA" and score >= 4, (label, score)

    # imperative first word alone reaches the PSA threshold
    label, _ = classify_psa("Boil water before drinking it every day.")
    assert label == "PSA", label
    print("ok  test_scoring_edge_cases")


def test_classify_frame():
    df = pd.DataFrame({
        "English": KEPT_PSA[:1] + NON_PSA[:1] + ["KRA wins case against Dubai firm"],
        "Domain": ["Health", "Health", "Governance"],
    })
    out = classify_frame(df)
    assert "psa_class" in out.columns and "psa_score" in out.columns
    assert list(out["psa_class"]) == ["PSA", "Informational", "PressRelease"]
    # original frame untouched
    assert "psa_class" not in df.columns
    print("ok  test_classify_frame")


def run() -> int:
    """Run all classifier tests; return 0 on success."""
    test_reference_examples()
    test_scoring_edge_cases()
    test_classify_frame()
    print("ok  test_psa_classify")
    return 0


if __name__ == "__main__":
    sys.exit(run())
