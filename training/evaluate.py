"""Week 3 evaluation — sacreBLEU + chrF of a checkpoint (SPEC_WEEK3.md §2.6).

Metric paths (sacrebleu, registry) import cleanly without torch; generation
goes through training.inference.MTTranslator, imported lazily inside
``evaluate_checkpoint``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

# name -> (loader, split, src, tgt); frozen registry (SPEC_WEEK3.md §2.6).
# "guzbench" reads data/external/guz_benchmark/guz_test.tsv — the held-out
# Ekegusii benchmark built from the PSA test split (FLORES-200 has no
# Ekegusii; see scripts/build_guz_benchmark.py).
EVAL_SPECS = {
    "psa_dev_en-sw":   ("psa", "dev",  "eng", "swa"),
    "psa_dev_sw-en":   ("psa", "dev",  "eng", "swa"),   # reversed at load
    "psa_test_en-sw":  ("psa", "test", "eng", "swa"),
    "psa_test_sw-en":  ("psa", "test", "eng", "swa"),
    "psa_test_en-guz": ("guzbench", "test", "eng", "guz"),
    "psa_test_guz-en": ("guzbench", "test", "guz", "eng"),
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_dir(rel: str) -> Path:
    """Find a data dir relative to the cwd, else to the project root."""
    cwd_path = Path(rel)
    if cwd_path.is_dir():
        return cwd_path
    return _PROJECT_ROOT / rel


def splits_dir() -> Path:
    return _resolve_dir("data/processed/splits")


def guz_benchmark_dir() -> Path:
    return _resolve_dir("data/external/guz_benchmark")


def _load_psa_pairs(split: str, src: str, tgt: str,
                    n: int | None, seed: int) -> tuple[list[str], list[str]]:
    """Read data/processed/splits/<split>.csv paired rows only."""
    import pandas as pd

    path = splits_dir() / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"PSA split file not found: {path} (run the Week 2 pipeline first)")
    df = pd.read_csv(path, encoding="utf-8").fillna("")
    df = df[(df["English"].str.strip() != "") & (df["Kiswahili"].str.strip() != "")]
    if df.empty:
        raise ValueError(f"no paired EN-SW rows in {path}")
    pairs = list(zip(df["English"].tolist(), df["Kiswahili"].tolist()))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if n is not None:
        pairs = pairs[:n]
    if (src, tgt) == ("eng", "swa"):
        return [p[0] for p in pairs], [p[1] for p in pairs]
    # reversed at load (sw -> en)
    return [p[1] for p in pairs], [p[0] for p in pairs]


def _load_guz_pairs(src: str, tgt: str,
                    n: int | None, seed: int) -> tuple[list[str], list[str]]:
    """Held-out Ekegusii benchmark via training.data.load_guz_benchmark."""
    from .data import load_guz_benchmark  # noqa: lazy import

    ds = load_guz_benchmark(guz_benchmark_dir(), n=n, seed=seed)
    eng, guz = list(ds["eng"]), list(ds["guz"])
    if (src, tgt) == ("eng", "guz"):
        return eng, guz
    return guz, eng  # guz -> en


def corpus_scores(hyps: list[str], refs: list[str]) -> tuple[float, float]:
    """(corpus BLEU, corpus chrF) via sacrebleu."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return float(bleu), float(chrf)


def evaluate_checkpoint(ckpt: str | Path, eval_spec: str, n: int | None = 200,
                        batch_size: int = 16, seed: int = 42,
                        out_dir: Path | None = None) -> dict:
    """Evaluate one checkpoint on one registered eval spec.

    Returns {"eval_spec", "ckpt", "n", "bleu", "chrf", "seconds",
    "model_key"}; if out_dir is given, writes <out_dir>/<eval_spec>.json.
    """
    if eval_spec not in EVAL_SPECS:
        raise KeyError(
            f"unknown eval_spec '{eval_spec}' (have: {sorted(EVAL_SPECS)})")
    loader, split, src, tgt = EVAL_SPECS[eval_spec]
    if loader == "psa":
        sources, refs = _load_psa_pairs(split, src, tgt, n, seed)
    else:
        sources, refs = _load_guz_pairs(src, tgt, n, seed)

    from .inference import MTTranslator  # noqa: lazy import (torch inside)

    t0 = time.perf_counter()
    translator = MTTranslator(ckpt)
    hyps: list[str] = []
    for i in range(0, len(sources), batch_size):
        hyps.extend(translator.translate(sources[i:i + batch_size],
                                         src=src, tgt=tgt))
    seconds = time.perf_counter() - t0

    bleu, chrf = corpus_scores(hyps, refs)
    result = {
        "eval_spec": eval_spec,
        "ckpt": str(ckpt),
        "n": len(sources),
        "bleu": bleu,
        "chrf": chrf,
        "seconds": round(seconds, 2),
        "model_key": translator.model_key,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{eval_spec}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result
