"""Week 3 data builders — PSA splits + held-out guz benchmark (SPEC_WEEK3.md §2.2).

Canonical example columns (HF ``datasets.Dataset``):
    src_text, tgt_text, src_lang, tgt_lang, domain, provenance

``src_lang``/``tgt_lang`` in {"eng", "swa", "guz"}; ``provenance`` in
{"psa", "backtranslation"}.

Non-negotiables (spec §4):
- The guz benchmark (``guz_test.tsv``, built from the TEST split by
  scripts/build_guz_benchmark.py) is EVALUATION-ONLY and never enters
  training; build_train_dataset raises if asked for few-shot guz pairs
  while the train split has no Ekegusii text.
- Ekegusii comes exclusively from real PSA pairs (lecturer gold data in
  the Week 1 dataset). FLORES-200 was evaluated and dropped: the archive
  has 204 languages and no guz_Latn, and NLLB-200's tokenizer has no
  guz_Latn token either — no off-the-shelf Ekegusii benchmark exists.
- Seeded everything (seed=42 default).
- No torch/transformers import at module import time; ``datasets`` itself is
  imported lazily inside functions so this module imports cleanly on a bare
  CPU box.
- pathlib everywhere, UTF-8 everywhere, Windows-compatible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.config import TrainConfig

CANONICAL_COLUMNS = [
    "src_text", "tgt_text", "src_lang", "tgt_lang", "domain", "provenance",
]

# PSA split CSV column per language code (Week 1 schema is the source of truth).
PSA_TEXT_COLUMNS = {"eng": "English", "swa": "Kiswahili", "guz": "Ekegusii"}

PSA_DIRECTIONS = ("en-sw", "sw-en")        # available from the PSA splits
GUZ_DIRECTIONS = ("en-guz", "sw-guz")      # PSA rows with non-empty Ekegusii

PROVENANCE_PSA = "psa"
PROVENANCE_BACKTRANSLATION = "backtranslation"

DEFAULT_GUZ_DIRS: list[str] = ["en-guz", "sw-guz"]


# ---------------------------------------------------------------------------
# Direction expansion (single place — spec §2.1 footnote)
# ---------------------------------------------------------------------------

def expand_directions(direction: str) -> list[str]:
    """Expand direction aliases to explicit direction codes.

    "both" -> ["en-sw", "sw-en"]; "all" -> ["en-sw", "sw-en", "en-guz",
    "sw-guz"]; single codes stay as-is. Unknown codes raise ValueError.
    """
    if direction == "both":
        return ["en-sw", "sw-en"]
    if direction == "all":
        return ["en-sw", "sw-en", "en-guz", "sw-guz"]
    if direction in PSA_DIRECTIONS + GUZ_DIRECTIONS:
        return [direction]
    raise ValueError(f"unknown direction '{direction}'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _patch_datasets_pickling() -> None:
    """Compat shim for ``datasets`` fingerprint hashing (pyarrow>=21 / numpy>=2.4).

    ``datasets`` fingerprints every new Dataset by pickling its state with dill
    in *recurse* mode, which serialises whole module namespaces by value. That
    dies on recent dependency stacks:

    - pyarrow's ``MonthDayNano`` class claims ``__module__='builtins'`` but is
      not registered there, so global-reference pickling fails;
    - numpy's internal ``*_with_like`` dispatch helpers are tagged
      ``__module__='numpy'`` without being identical to the public functions.

    Fingerprinting does not need by-value semantics for our ephemeral in-memory
    datasets, so force ``recurse=False`` (modules/functions pickle by import
    reference). Idempotent; silently no-ops if datasets internals change.
    """
    try:
        import datasets.utils._dill as _ddill
    except Exception:
        return
    if getattr(_ddill.Pickler, "_psa_no_recurse", False):
        return
    _OrigPickler = _ddill.Pickler

    class _NoRecursePickler(_OrigPickler):
        _psa_no_recurse = True

        def __init__(self, file, *args, **kwargs):
            kwargs["recurse"] = False
            super().__init__(file, *args, **kwargs)

    _ddill.Pickler = _NoRecursePickler


def _require_datasets():
    """Import HF datasets lazily; raise a helpful error if it is missing."""
    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "training.data needs the 'datasets' package: pip install datasets"
        ) from exc
    _patch_datasets_pickling()
    return Dataset, concatenate_datasets


def _dataset_from_rows(rows: list[dict]):
    """Build a Dataset from canonical rows (empty -> empty-typed Dataset)."""
    Dataset, _ = _require_datasets()
    return Dataset.from_dict({c: [r[c] for r in rows] for c in CANONICAL_COLUMNS})


def _read_tsv(path: Path) -> pd.DataFrame:
    """Read a canonical benchmark TSV (header ``eng\\tguz``), UTF-8."""
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8").fillna("")


# ---------------------------------------------------------------------------
# Public loaders (spec §2.2)
# ---------------------------------------------------------------------------

def load_psa_pairs(splits_dir: Path, split: str, directions: list[str]):
    """Load PSA pairs from ``<splits_dir>/<split>.csv`` as a canonical Dataset.

    For en-sw / sw-en: keeps rows with Kiswahili != "" (and English != "").
    For guz directions: additionally emits pairs from
    rows with non-empty ``Ekegusii`` — en-guz (English->Ekegusii) for every
    such row, sw-guz (Kiswahili->Ekegusii) only when Kiswahili is non-empty.
    Domain is carried; provenance="psa" for all directions.
    """
    splits_dir = Path(splits_dir)
    csv_path = splits_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"split file not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")
    paired = df[(df["Kiswahili"].str.strip() != "")
                & (df["English"].str.strip() != "")]
    guz_rows = df[(df["Ekegusii"].str.strip() != "")
                  & (df["English"].str.strip() != "")]
    rows: list[dict] = []
    for direction in directions:
        if direction == "en-sw":
            src_col, tgt_col, src_lang, tgt_lang = ("English", "Kiswahili", "eng", "swa")
            for _, r in paired.iterrows():
                rows.append({
                    "src_text": r[src_col], "tgt_text": r[tgt_col],
                    "src_lang": src_lang, "tgt_lang": tgt_lang,
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        elif direction == "sw-en":
            for _, r in paired.iterrows():
                rows.append({
                    "src_text": r["Kiswahili"], "tgt_text": r["English"],
                    "src_lang": "swa", "tgt_lang": "eng",
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        elif direction == "en-guz":
            for _, r in guz_rows.iterrows():
                rows.append({
                    "src_text": r["English"], "tgt_text": r["Ekegusii"],
                    "src_lang": "eng", "tgt_lang": "guz",
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        elif direction == "sw-guz":
            for _, r in guz_rows.iterrows():
                if r["Kiswahili"].strip() == "":
                    continue  # sw-guz only when Kiswahili is non-empty
                rows.append({
                    "src_text": r["Kiswahili"], "tgt_text": r["Ekegusii"],
                    "src_lang": "swa", "tgt_lang": "guz",
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        else:
            raise ValueError(f"unknown direction '{direction}'")
    return _dataset_from_rows(rows)


def load_guz_benchmark(benchmark_dir: Path, n: int | None = 200, seed: int = 42):
    """Load the held-out Ekegusii benchmark pairs for evaluation ONLY.

    Reads ``<benchmark_dir>/guz_test.tsv`` (built from the PSA **test** split
    by scripts/build_guz_benchmark.py); deterministic seeded sample of n
    (n=None -> all rows, in file order). Returns raw ``eng``/``guz`` columns;
    evaluate.py normalizes them to src/tgt per eval spec. Never used for
    training.
    """
    Dataset, _ = _require_datasets()
    bench_path = Path(benchmark_dir) / "guz_test.tsv"
    if not bench_path.exists():
        raise FileNotFoundError(
            f"guz benchmark file not found: {bench_path} — run "
            "scripts/build_guz_benchmark.py first.")
    df = _read_tsv(bench_path)
    if n is not None:
        df = df.sample(frac=1.0, random_state=seed).head(n)
    return Dataset.from_dict({"eng": df["eng"].tolist(),
                              "guz": df["guz"].tolist()})


def build_train_dataset(cfg: TrainConfig, splits_dir: Path,
                        augmented_csv: Path | None = None):
    """Build the full training Dataset for ``cfg`` (spec §2.2).

    Concatenates, in order:
      1. PSA pairs from the train split (en-sw / sw-en as requested);
      2. PSA-sourced guz pairs (rows with non-empty Ekegusii), gated by
         ``cfg.fewshot_guz``: 0 = exclude, N = seeded cap, -1 = all;
      3. augmented_csv rows (provenance="backtranslation").

    Then applies a seeded shuffle and the cfg.max_samples cap (AFTER
    concatenation). HARD GUARD: if cfg.fewshot_guz > 0 but the train split
    has no Ekegusii text at all, raise — the guz benchmark (built from the
    test split) is evaluation-only and must never train.
    """
    _, concatenate_datasets = _require_datasets()
    directions = cfg.directions()
    parts = []

    psa_dirs = [d for d in directions if d in PSA_DIRECTIONS]
    if psa_dirs:
        parts.append(load_psa_pairs(splits_dir, "train", psa_dirs))

    if cfg.fewshot_guz != 0:
        guz_dirs = [d for d in directions if d in GUZ_DIRECTIONS] or list(DEFAULT_GUZ_DIRS)
        psa_guz = load_psa_pairs(splits_dir, "train", guz_dirs)
        if cfg.fewshot_guz > 0 and len(psa_guz) > cfg.fewshot_guz:
            psa_guz = psa_guz.shuffle(seed=cfg.seed).select(range(cfg.fewshot_guz))
        if len(psa_guz):
            parts.append(psa_guz)
        elif cfg.fewshot_guz > 0:
            raise FileNotFoundError(
                f"fewshot_guz={cfg.fewshot_guz} requested but the train split "
                f"has no Ekegusii text ({splits_dir}/train.csv). The guz "
                "benchmark (guz_test.tsv) is built from the test split, is "
                "evaluation-only, and may never be used for training.")

    if augmented_csv is not None:
        parts.append(_load_augmented(augmented_csv,
                                     [d for d in directions if d in PSA_DIRECTIONS]
                                     or list(PSA_DIRECTIONS)))

    if not parts:
        raise ValueError(
            f"nothing to train on for direction '{cfg.direction}' "
            f"(fewshot_guz={cfg.fewshot_guz}, augmented_csv={augmented_csv})")
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    ds = ds.shuffle(seed=cfg.seed)
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    return ds


def _load_augmented(augmented_csv: Path, directions: list[str]):
    """Load back-translated rows (dataset-schema CSV) as canonical pairs.

    Supports all four directions: en-sw/sw-en from the English+Kiswahili
    columns (Week 3 eng->swa augmentation) and en-guz/sw-guz from rows with
    the Ekegusii column filled (Week 4 eng->guz augmentation — existing
    Kiswahili text is carried over by training/augment.py, so one CSV can
    feed both).
    """
    augmented_csv = Path(augmented_csv)
    if not augmented_csv.exists():
        raise FileNotFoundError(f"augmented csv not found: {augmented_csv}")
    df = pd.read_csv(augmented_csv, dtype=str, encoding="utf-8").fillna("")
    cols = {
        "en-sw": ("English", "Kiswahili", "eng", "swa"),
        "sw-en": ("Kiswahili", "English", "swa", "eng"),
        "en-guz": ("English", "Ekegusii", "eng", "guz"),
        "sw-guz": ("Kiswahili", "Ekegusii", "swa", "guz"),
    }
    rows: list[dict] = []
    for direction in directions:
        if direction not in cols:
            continue
        src_col, tgt_col, src_lang, tgt_lang = cols[direction]
        usable = df[(df[src_col].str.strip() != "")
                    & (df[tgt_col].str.strip() != "")]
        for _, r in usable.iterrows():
            rows.append({
                "src_text": r[src_col], "tgt_text": r[tgt_col],
                "src_lang": src_lang, "tgt_lang": tgt_lang,
                "domain": r["Domain"], "provenance": PROVENANCE_BACKTRANSLATION,
            })
    return _dataset_from_rows(rows)
