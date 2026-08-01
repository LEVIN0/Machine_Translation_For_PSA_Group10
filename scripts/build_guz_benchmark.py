#!/usr/bin/env python3
"""Build the held-out Ekegusii benchmark TSV from the PSA test split.

Background (SPEC_WEEK3.md §2.2 addendum): FLORES-200 contains no Ekegusii
(204 languages in the archive; ``guz_Latn`` absent) and NLLB-200's tokenizer
has no ``guz_Latn`` token either — neither Week 3 model has any Ekegusii
pretraining. Our own held-out test split is therefore the Ekegusii benchmark.

Reads ``data/processed/splits/test.csv``, keeps rows with non-empty English
AND Ekegusii, and writes ``data/external/guz_benchmark/guz_test.tsv``
(header ``eng<TAB>guz``, UTF-8, one sentence pair per line).

The output is EVALUATION-ONLY: the test split is never used for training.
pathlib + UTF-8 everywhere; Windows-compatible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "splits"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "external" / "guz_benchmark"


def main(splits_dir: Path = DEFAULT_SPLITS,
         out_dir: Path = DEFAULT_OUT_DIR) -> int:
    splits_dir, out_dir = Path(splits_dir), Path(out_dir)
    test_csv = splits_dir / "test.csv"
    if not test_csv.is_file():
        print(f"build_guz_benchmark: ERROR split file not found: {test_csv}\n"
              "  run the Week 1 and Week 2 pipelines first "
              "(scripts/run_week1.py, scripts/run_week2.py).",
              file=sys.stderr)
        return 1

    df = pd.read_csv(test_csv, dtype=str, encoding="utf-8").fillna("")
    df = df[(df["English"].str.strip() != "") & (df["Ekegusii"].str.strip() != "")]
    if df.empty:
        print("build_guz_benchmark: ERROR no eng-guz pairs in the test split.",
              file=sys.stderr)
        return 1

    bench = df[["English", "Ekegusii"]].rename(
        columns={"English": "eng", "Ekegusii": "guz"})
    # Canonical format is one pair per line: collapse any embedded newlines
    # so no consumer (pandas, shell tools, sacrebleu-style readers) ever sees
    # a quoted multi-line field.
    for col in ("eng", "guz"):
        bench[col] = bench[col].map(lambda s: " ".join(str(s).split()))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "guz_test.tsv"
    bench.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

    print(f"build_guz_benchmark: {len(bench)} eng-guz pairs -> {out_path}")
    print("build_guz_benchmark: done. Reminder: guz_test.tsv is "
          "EVALUATION-ONLY (never used for training).")
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    sys.exit(main(out_dir=out))
