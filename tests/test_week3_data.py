#!/usr/bin/env python3
"""Week 3 data/augment tests (Agent A) — offline, fast, no pytest needed.

Exposes run() which prints "ok  test_week3_data" on success (spec §4).
If the 'datasets' package is unavailable, the dataset-dependent tests SKIP
gracefully (printed reason, still exit 0); all import/compile/logic tests
always run.

Run from the project root:  python tests/test_week3_data.py
"""

import py_compile
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BENCH_FIXTURE = FIXTURES / "guz_benchmark"
SPLITS_FIXTURE = FIXTURES / "week3" / "splits"

_SKIPPED: list[str] = []


def _skip(name: str, reason: str) -> None:
    _SKIPPED.append(f"{name}: {reason}")
    print(f"skipped: {name} ({reason})")


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 1. Direction expansion (no heavy deps)
# ---------------------------------------------------------------------------

def test_direction_expansion():
    from training.data import expand_directions
    assert expand_directions("both") == ["en-sw", "sw-en"]
    assert expand_directions("all") == ["en-sw", "sw-en", "en-guz", "sw-guz"]
    assert expand_directions("en-sw") == ["en-sw"]
    assert expand_directions("sw-guz") == ["sw-guz"]
    try:
        expand_directions("xx-yy")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown direction must raise ValueError")
    # TrainConfig.directions() (frozen contract) agrees with expand_directions.
    from training.config import TrainConfig
    base = dict(run_name="t", model_key="mt5_small")
    assert TrainConfig(direction="both", **base).directions() == ["en-sw", "sw-en"]
    assert TrainConfig(direction="all", **base).directions() == [
        "en-sw", "sw-en", "en-guz", "sw-guz"]
    print("ok  test_direction_expansion")


# ---------------------------------------------------------------------------
# 2. load_psa_pairs from fixture splits
# ---------------------------------------------------------------------------

def test_load_psa_pairs():
    from training.data import CANONICAL_COLUMNS, load_psa_pairs
    ds = load_psa_pairs(SPLITS_FIXTURE, "train", ["en-sw"])
    # train fixture: 10 rows, 6 paired (Kiswahili != "") -> 6 en-sw pairs
    assert len(ds) == 6, f"expected 6 en-sw pairs, got {len(ds)}"
    assert ds.column_names == CANONICAL_COLUMNS, ds.column_names
    assert set(ds["src_lang"]) == {"eng"} and set(ds["tgt_lang"]) == {"swa"}
    assert set(ds["provenance"]) == {"psa"}
    assert all(d for d in ds["domain"])  # domain carried, non-empty

    both = load_psa_pairs(SPLITS_FIXTURE, "train", ["en-sw", "sw-en"])
    assert len(both) == 12, f"both directions must double pairs, got {len(both)}"
    rev = [r for r in both if r["src_lang"] == "swa"]
    assert len(rev) == 6
    # sw-en reverses src/tgt of the corresponding en-sw rows
    fwd = sorted(zip(ds["src_text"], ds["tgt_text"]))
    rev_pairs = sorted((r["tgt_text"], r["src_text"]) for r in rev)
    assert fwd == rev_pairs

    dev = load_psa_pairs(SPLITS_FIXTURE, "dev", ["en-sw"])
    assert len(dev) == 2, f"dev fixture has 2 paired rows, got {len(dev)}"
    # guz directions are ignored for PSA splits (no guz text yet)
    assert len(load_psa_pairs(SPLITS_FIXTURE, "train", ["en-guz"])) == 0
    print("ok  test_load_psa_pairs")


# ---------------------------------------------------------------------------
# 3. load_guz_benchmark: n-slicing + determinism (eval-only benchmark)
# ---------------------------------------------------------------------------

def test_load_guz_benchmark():
    from training.data import load_guz_benchmark
    ds = load_guz_benchmark(BENCH_FIXTURE, 3)
    assert len(ds) == 3, f"expected exactly n=3 benchmark rows, got {len(ds)}"
    assert ds.column_names == ["eng", "guz"], ds.column_names

    # determinism: same seed -> same pairs; n capped by fixture size (8)
    again = load_guz_benchmark(BENCH_FIXTURE, 3)
    assert ds["eng"] == again["eng"] and ds["guz"] == again["guz"]
    full = load_guz_benchmark(BENCH_FIXTURE, 100)
    assert len(full) == 8, f"fixture has 8 benchmark pairs, got {len(full)}"
    # n=None -> all rows, in file order
    ordered = load_guz_benchmark(BENCH_FIXTURE, None)
    assert ordered["eng"][0] == "Wash your hands with soap and clean water."
    print("ok  test_load_guz_benchmark")


# ---------------------------------------------------------------------------
# 4. build_train_dataset: concatenation + max_samples cap
# ---------------------------------------------------------------------------

def test_build_train_dataset():
    from training.config import TrainConfig
    from training.data import build_train_dataset
    base = dict(run_name="t", model_key="mt5_small")

    cfg = TrainConfig(direction="both", **base)
    ds = build_train_dataset(cfg, SPLITS_FIXTURE)
    assert len(ds) == 12, f"both directions -> 12 psa pairs, got {len(ds)}"
    assert set(ds["provenance"]) == {"psa"}

    cfg = TrainConfig(direction="both", max_samples=5, **base)
    ds = build_train_dataset(cfg, SPLITS_FIXTURE)
    assert len(ds) == 5, f"max_samples cap must apply, got {len(ds)}"
    # cap is deterministic (seeded shuffle before select)
    ds2 = build_train_dataset(cfg, SPLITS_FIXTURE)
    assert ds["src_text"] == ds2["src_text"]
    print("ok  test_build_train_dataset")


# ---------------------------------------------------------------------------
# 5. build_guz_benchmark script: writes guz_test.tsv from a test split
# ---------------------------------------------------------------------------

def test_build_guz_benchmark_script():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import build_guz_benchmark as bgb

    with tempfile.TemporaryDirectory(prefix="guz_bench_") as tmp:
        tmp = Path(tmp)
        splits = tmp / "splits"
        splits.mkdir()
        # minimal split fixture: 2 guz rows + 1 non-guz row (excluded)
        (splits / "test.csv").write_text(
            "PSA_ID,Domain,English,Kiswahili,Ekegusii,Source,Date,URL,Metadata,Status\n"
            'P1,Health,Boil water.,Chemsha maji.,Tancha amache.,X,,,,Validated\n'
            'P2,Security,Lock the door.,Funga mlango.,,X,,,,Validated\n'
            'P3,Education,Go to school.,Nenda shuleni.,"Karia\nesomero.",X,,,,Validated\n',
            encoding="utf-8")
        out = tmp / "out"
        assert bgb.main(splits_dir=splits, out_dir=out) == 0
        tsv = (out / "guz_test.tsv").read_text(encoding="utf-8").splitlines()
        assert tsv[0] == "eng\tguz"
        assert len(tsv) == 3, f"header + 2 guz pairs expected, got {len(tsv)}"
        assert "Boil water.\tTancha amache." in tsv
        # one pair per line even when a source cell contains a newline
        assert all(len(l.split("\t")) == 2 for l in tsv), tsv
        assert "Go to school.\tKaria esomero." in tsv

        # missing test.csv -> exit code 1
        assert bgb.main(splits_dir=tmp / "nope", out_dir=out) == 1
    print("ok  test_build_guz_benchmark_script")


# ---------------------------------------------------------------------------
# 6. augment.py imports cleanly without torch
# ---------------------------------------------------------------------------

def test_augment_import_no_torch():
    sys.modules.pop("training.augment", None)
    before = set(sys.modules)
    import training.augment as augment  # noqa: F401
    new_mods = set(sys.modules) - before
    assert "torch" not in new_mods, "importing augment must not pull in torch"
    assert not any(m.startswith("torch.") for m in new_mods)
    assert callable(augment.backtranslate)
    print("ok  test_augment_import_no_torch")


# ---------------------------------------------------------------------------
# 7. scripts/build_guz_benchmark.py + training/lang_tokens.py compile
# ---------------------------------------------------------------------------

def test_new_modules_compile():
    for path in (PROJECT_ROOT / "scripts" / "build_guz_benchmark.py",
                 PROJECT_ROOT / "training" / "lang_tokens.py"):
        assert path.exists(), f"missing {path}"
        py_compile.compile(str(path), doraise=True)
    print("ok  test_new_modules_compile")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> int:
    """Run all Week 3 data tests; return 0 on success."""
    have_datasets = _datasets_available()

    test_direction_expansion()
    if have_datasets:
        test_load_psa_pairs()
        test_load_guz_benchmark()
        test_build_train_dataset()
    else:
        for name in ("test_load_psa_pairs", "test_load_guz_benchmark",
                     "test_build_train_dataset"):
            _skip(name, "python package 'datasets' not installed "
                        "(pip install datasets)")
    test_build_guz_benchmark_script()
    test_augment_import_no_torch()
    test_new_modules_compile()

    if _SKIPPED:
        print(f"note: {len(_SKIPPED)} test(s) skipped gracefully:")
        for s in _SKIPPED:
            print(f"  - {s}")
    print("ok  test_week3_data")
    return 0


if __name__ == "__main__":
    sys.exit(run())
