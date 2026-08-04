#!/usr/bin/env python3
"""Week 4 error analysis (docs/SPEC_WEEK4.md).

Reads the per-row predictions written by ``scripts/run_week4_eval.py``
(``reports/week4_eval/predictions/<direction>.csv``) and writes a markdown
summary: per-domain BLEU/chrF, repetition-loop flagged examples, and the
k lowest sentence-chrF examples per direction for qualitative review.

Pure pandas + sacrebleu over already-generated predictions — no torch, no
model, no GPU needed. Can run on any machine once the predictions CSVs
exist (e.g. copied down from the Kinesis node).

Run from the project root:

    python scripts/error_analysis.py
    python scripts/error_analysis.py --eval-dir reports/week4_eval --k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DIRECTIONS = ("en-sw", "sw-en", "en-guz", "guz-en")
# direction -> (source column, target column)
COLS = {
    "en-sw": ("English", "Kiswahili"),
    "sw-en": ("Kiswahili", "English"),
    "en-guz": ("English", "Ekegusii"),
    "guz-en": ("Ekegusii", "English"),
}


def corpus_scores(hyps: list[str], refs: list[str]) -> tuple[float, float]:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return float(bleu), float(chrf)


def domain_table(df: pd.DataFrame, tgt_col: str) -> str:
    rows = []
    for domain, g in sorted(df.groupby("Domain"), key=lambda kv: kv[0]):
        bleu, chrf = corpus_scores(g["hypothesis"].tolist(), g[tgt_col].tolist())
        rows.append((domain, len(g), bleu, chrf, int(g["repetition_flag"].sum())))
    lines = ["| Domain | n | BLEU | chrF | Repetition-flagged |",
             "|---|---:|---:|---:|---:|"]
    for domain, n, bleu, chrf, rep in rows:
        lines.append(f"| {domain} | {n} | {bleu:.2f} | {chrf:.2f} | {rep} |")
    return "\n".join(lines)


def worst_examples_table(df: pd.DataFrame, src_col: str, tgt_col: str, k: int) -> str:
    worst = df.nsmallest(k, "sentence_chrf")
    lines = ["| Domain | chrF | Source | Reference | Model output |",
             "|---|---:|---|---|---|"]
    for _, row in worst.iterrows():
        src = str(row[src_col]).replace("|", "\\|")[:120]
        ref = str(row[tgt_col]).replace("|", "\\|")[:120]
        hyp = str(row["hypothesis"]).replace("|", "\\|")[:120]
        lines.append(f"| {row['Domain']} | {row['sentence_chrf']:.1f} | "
                     f"{src} | {ref} | {hyp} |")
    return "\n".join(lines)


def repetition_examples_table(df: pd.DataFrame, k: int) -> str:
    flagged = df[df["repetition_flag"]].head(k)
    if flagged.empty:
        return "_None flagged in this direction._"
    lines = ["| Domain | Model output (truncated) |", "|---|---|"]
    for _, row in flagged.iterrows():
        hyp = str(row["hypothesis"]).replace("|", "\\|")[:160]
        lines.append(f"| {row['Domain']} | {hyp} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 4 error analysis.")
    parser.add_argument("--eval-dir", default="reports/week4_eval")
    parser.add_argument("--k", type=int, default=8,
                        help="worst/flagged examples shown per direction")
    parser.add_argument("--out", default="reports/week4_error_analysis.md")
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    pred_dir = eval_dir / "predictions"
    if not pred_dir.is_dir():
        parser.error(f"{pred_dir} not found — run scripts/run_week4_eval.py first")

    sections = ["# Week 4 error analysis\n",
                "Generated from `scripts/run_week4_eval.py` predictions "
                f"in `{pred_dir}`. Repetition-loop flagging is a cheap "
                "n-gram heuristic (see `scripts/run_week4_eval.py::"
                "repetition_flag`) — flagged rows still need a human read, "
                "they are candidates, not confirmed failures.\n"]

    for direction in DIRECTIONS:
        path = pred_dir / f"{direction}.csv"
        if not path.is_file():
            continue
        src_col, tgt_col = COLS[direction]
        df = pd.read_csv(path, encoding="utf-8").fillna("")
        sections.append(f"## {direction}  (n={len(df)})\n")
        sections.append("### Per-domain scores\n")
        sections.append(domain_table(df, tgt_col) + "\n")
        sections.append(f"### {args.k} lowest-chrF examples\n")
        sections.append(worst_examples_table(df, src_col, tgt_col, args.k) + "\n")
        sections.append(f"### Repetition-loop flagged examples (up to {args.k})\n")
        sections.append(repetition_examples_table(df, args.k) + "\n")

    out_path = Path(args.out)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[error_analysis] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
