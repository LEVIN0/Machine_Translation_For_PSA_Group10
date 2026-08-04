#!/usr/bin/env python3
"""Week 4 pure-function tests (SPEC_WEEK4.md). Plain asserts, no pytest.

Covers the repetition-loop heuristic and the DIRECTIONS/COLS registries
kept in sync between scripts/run_week4_eval.py and scripts/error_analysis.py
— no torch, no checkpoint, no GPU needed.

Run from the project root:  python tests/test_week4_error_analysis.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from error_analysis import COLS as ERROR_COLS  # noqa: E402
from error_analysis import DIRECTIONS as ERROR_DIRECTIONS  # noqa: E402
from run_week4_eval import DIRECTIONS  # noqa: E402
from run_week4_eval import repetition_flag  # noqa: E402


def test_directions_registry_agrees_with_error_analysis():
    assert set(ERROR_DIRECTIONS) == set(DIRECTIONS)
    for direction in DIRECTIONS:
        src_col, tgt_col, _, _ = DIRECTIONS[direction]
        assert ERROR_COLS[direction] == (src_col, tgt_col)


def test_repetition_flag_detects_back_to_back_repeats():
    looped = "wash your hands wash your hands wash your hands"
    assert repetition_flag(looped) is True


def test_repetition_flag_ignores_normal_sentences():
    normal = "Wash your hands with soap and clean water for twenty seconds."
    assert repetition_flag(normal) is False


def test_repetition_flag_ignores_short_text():
    # Too short to ever contain min_repeats copies of an n-gram.
    assert repetition_flag("hello there") is False


def test_repetition_flag_requires_back_to_back_not_scattered():
    # Same trigram appears twice but not back-to-back three times.
    scattered = "wash your hands then dry them then wash your hands again"
    assert repetition_flag(scattered, n=3, min_repeats=3) is False


def run() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"ok  {len(tests)} Week 4 error-analysis tests")


if __name__ == "__main__":
    run()
