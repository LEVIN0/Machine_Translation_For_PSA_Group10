#!/usr/bin/env python3
"""Week 4 final evaluation (docs/SPEC_WEEK4.md).

Unlike ``scripts/run_eval.py`` (Week 3: capped ``n=200`` per spec, for fast
iteration during the ablation matrix), this runs the **full** held-out test
split for all four directions and writes per-row predictions so the error
analysis and human-eval steps have something to read.

For each direction (en-sw, sw-en, en-guz, guz-en):
  - loads every test-split row with non-empty source + target text
  - translates the full set with the given checkpoint
  - scores corpus BLEU/chrF overall and per PSA domain
  - flags likely repetition-loop outputs (see week3_report.md Challenges:
    the model repeats itself on longer, low-resource guz sentences)
  - writes predictions/<direction>.csv (one row per example, for
    scripts/error_analysis.py and scripts/build_human_eval_sheet.py)

Run from the project root (GPU recommended — same checkpoint used for
Week 3 training/eval, e.g. on the Kinesis node, docs/week3_kinesis_guide.md):

    python scripts/run_week4_eval.py \\
        --checkpoint runs/ft_nllb_guz_all/checkpoint-best

    python scripts/run_week4_eval.py --checkpoint ... \\
        --directions en-sw,sw-en --batch-size 32

With no --checkpoint, the newest runs/*/checkpoint-best is auto-discovered
(same rule as scripts/translate.py).

Results:
    reports/week4_eval/summary.json                 overall + per-domain
    reports/week4_eval/predictions/<direction>.csv   per-row predictions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# direction -> (source column, target column, source code, target code)
# codes are training.config.LANGS keys ("eng"/"swa"/"guz").
DIRECTIONS: dict[str, tuple[str, str, str, str]] = {
    "en-sw": ("English", "Kiswahili", "eng", "swa"),
    "sw-en": ("Kiswahili", "English", "swa", "eng"),
    "en-guz": ("English", "Ekegusii", "eng", "guz"),
    "guz-en": ("Ekegusii", "English", "guz", "eng"),
}

DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "week4_eval"


def discover_checkpoint(runs_root: Path = Path("runs")) -> Path | None:
    """Newest runs/*/checkpoint-best by modification time (scripts/translate.py rule)."""
    if not runs_root.is_dir():
        return None
    candidates = [p for p in runs_root.glob("*/checkpoint-best") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_rows(test_csv: Path, src_col: str, tgt_col: str) -> pd.DataFrame:
    """Every test-split row with non-empty src AND tgt text, PSA_ID + Domain kept."""
    df = pd.read_csv(test_csv, encoding="utf-8").fillna("")
    df = df[(df[src_col].str.strip() != "") & (df[tgt_col].str.strip() != "")]
    return df[["PSA_ID", "Domain", src_col, tgt_col]].reset_index(drop=True)


def repetition_flag(text: str, n: int = 3, min_repeats: int = 3) -> bool:
    """True if an n-gram repeats >= min_repeats times back-to-back.

    Cheap heuristic for the "repetition loop" failure mode noted in
    reports/week3_report.md §8 on longer, low-resource (guz) generations.
    """
    words = text.split()
    if len(words) < n * min_repeats:
        return False
    for i in range(len(words) - n * min_repeats + 1):
        gram = words[i:i + n]
        if all(words[i + n * k:i + n * (k + 1)] == gram for k in range(min_repeats)):
            return True
    return False


def corpus_scores(hyps: list[str], refs: list[str]) -> tuple[float, float]:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return float(bleu), float(chrf)


def run_direction(translator, df: pd.DataFrame, src_col: str, tgt_col: str,
                   src_code: str, tgt_code: str, batch_size: int) -> tuple[dict, pd.DataFrame]:
    import sacrebleu

    sources = df[src_col].tolist()
    refs = df[tgt_col].tolist()
    hyps: list[str] = []
    for i in range(0, len(sources), batch_size):
        hyps.extend(translator.translate(sources[i:i + batch_size],
                                          src=src_code, tgt=tgt_code))

    out = df.copy()
    out["hypothesis"] = hyps
    out["sentence_chrf"] = [sacrebleu.sentence_chrf(h, [r]).score
                             for h, r in zip(hyps, refs)]
    out["repetition_flag"] = [repetition_flag(h) for h in hyps]

    bleu, chrf = corpus_scores(hyps, refs)
    domain_rows = []
    for domain, g in out.groupby("Domain"):
        d_bleu, d_chrf = corpus_scores(g["hypothesis"].tolist(), g[tgt_col].tolist())
        domain_rows.append({"domain": domain, "n": len(g),
                             "bleu": round(d_bleu, 2), "chrf": round(d_chrf, 2),
                             "repetition_flagged": int(g["repetition_flag"].sum())})
    summary = {"n": len(df), "bleu": round(bleu, 2), "chrf": round(chrf, 2),
               "repetition_flagged": int(out["repetition_flag"].sum()),
               "by_domain": sorted(domain_rows, key=lambda r: r["domain"])}
    return summary, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 4 final evaluation.")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint dir/hub id; auto-discovered from "
                             "newest runs/*/checkpoint-best if omitted")
    parser.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--directions", default="en-sw,sw-en,en-guz,guz-en",
                        help=f"comma-separated, from {sorted(DIRECTIONS)}")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    ckpt = args.checkpoint
    if ckpt is None:
        found = discover_checkpoint()
        if found is None:
            parser.error("no --checkpoint given and no runs/*/checkpoint-best "
                         "found — train a model first (scripts/run_training.py)")
        ckpt = str(found)
        print(f"[week4_eval] auto-discovered checkpoint: {ckpt}")

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    unknown = [d for d in directions if d not in DIRECTIONS]
    if unknown:
        parser.error(f"unknown directions {unknown}; available: {sorted(DIRECTIONS)}")

    test_csv = Path(args.test_csv)
    if not test_csv.is_file():
        parser.error(f"test split not found: {test_csv} (run scripts/run_week2.py first)")

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    from training.inference import MTTranslator  # noqa: lazy import (torch inside)

    translator = MTTranslator(ckpt)
    print(f"[week4_eval] model: {translator.model_key} "
          f"({translator.family}) on {translator.device}")

    results: dict[str, dict] = {}
    for direction in directions:
        src_col, tgt_col, src_code, tgt_code = DIRECTIONS[direction]
        df = load_rows(test_csv, src_col, tgt_col)
        if df.empty:
            print(f"[week4_eval] {direction}: skipped (0 paired rows)")
            continue
        print(f"[week4_eval] {direction}: {len(df)} pairs ...", flush=True)
        t0 = time.perf_counter()
        summary, out = run_direction(translator, df, src_col, tgt_col,
                                      src_code, tgt_code, args.batch_size)
        summary["seconds"] = round(time.perf_counter() - t0, 1)
        results[direction] = summary
        out.to_csv(pred_dir / f"{direction}.csv", index=False, encoding="utf-8")
        print(f"[week4_eval] {direction}: BLEU {summary['bleu']:.2f} | "
              f"chrF {summary['chrf']:.2f} | n={summary['n']} | "
              f"repetition_flagged={summary['repetition_flagged']} | "
              f"{summary['seconds']:.1f}s")

    (out_dir / "summary.json").write_text(
        json.dumps({"checkpoint": str(ckpt), "model_key": translator.model_key,
                    "results": results}, indent=2), encoding="utf-8")

    print()
    print(f"{'direction':<10} {'n':>5} {'BLEU':>8} {'chrF':>8} {'rep_flag':>9}")
    print("-" * 44)
    for direction, r in results.items():
        print(f"{direction:<10} {r['n']:>5} {r['bleu']:>8.2f} {r['chrf']:>8.2f} "
              f"{r['repetition_flagged']:>9}")
    print(f"\nwritten to {out_dir}/summary.json and {pred_dir}/<direction>.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
