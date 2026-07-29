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
FLORES_FIXTURE = FIXTURES / "flores"
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
# 3. load_flores_seed: n-slicing + determinism + sw-guz skip
# ---------------------------------------------------------------------------

def test_load_flores_seed():
    from training.data import load_flores_seed
    ds = load_flores_seed(FLORES_FIXTURE, 3)
    assert len(ds) == 3, f"expected exactly n=3 seed rows, got {len(ds)}"
    assert set(ds["src_lang"]) == {"eng"} and set(ds["tgt_lang"]) == {"guz"}
    assert set(ds["provenance"]) == {"flores_dev_seed"}

    # determinism: same seed -> same pairs; n capped by fixture size (8)
    again = load_flores_seed(FLORES_FIXTURE, 3)
    assert ds["src_text"] == again["src_text"] and ds["tgt_text"] == again["tgt_text"]
    full = load_flores_seed(FLORES_FIXTURE, 100)
    assert len(full) == 8, f"fixture has 8 dev pairs, got {len(full)}"

    # fixture has no swa column -> sw-guz rows skipped with a warning, no crash
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sw_ds = load_flores_seed(FLORES_FIXTURE, 4, directions=["sw-guz"])
    assert len(sw_ds) == 0
    assert any("swa" in str(w.message) for w in caught)
    print("ok  test_load_flores_seed")


# ---------------------------------------------------------------------------
# 4. build_train_dataset: concatenation + max_samples cap
# ---------------------------------------------------------------------------

def test_build_train_dataset():
    from training.config import TrainConfig
    from training.data import build_train_dataset
    base = dict(run_name="t", model_key="mt5_small")

    cfg = TrainConfig(direction="both", **base)
    ds = build_train_dataset(cfg, SPLITS_FIXTURE, FLORES_FIXTURE)
    assert len(ds) == 12, f"both directions -> 12 psa pairs, got {len(ds)}"
    assert set(ds["provenance"]) == {"psa"}

    cfg = TrainConfig(direction="all", fewshot_guz=4, **base)
    ds = build_train_dataset(cfg, SPLITS_FIXTURE, FLORES_FIXTURE)
    # 12 psa pairs + 4 en-guz seeds (fixture has no swa column -> no sw-guz)
    assert len(ds) == 16, f"expected 16 pairs, got {len(ds)}"
    prov = set(ds["provenance"])
    assert prov == {"psa", "flores_dev_seed"}, prov
    guz = [r for r in ds if r["tgt_lang"] == "guz"]
    assert len(guz) == 4

    cfg = TrainConfig(direction="both", max_samples=5, **base)
    ds = build_train_dataset(cfg, SPLITS_FIXTURE, FLORES_FIXTURE)
    assert len(ds) == 5, f"max_samples cap must apply, got {len(ds)}"
    # cap is deterministic (seeded shuffle before select)
    ds2 = build_train_dataset(cfg, SPLITS_FIXTURE, FLORES_FIXTURE)
    assert ds["src_text"] == ds2["src_text"]
    print("ok  test_build_train_dataset")


# ---------------------------------------------------------------------------
# 5. devtest guard: raise when only devtest exists and fewshot_guz > 0
# ---------------------------------------------------------------------------

def test_devtest_guard():
    from training.config import TrainConfig
    from training.data import build_train_dataset, load_flores_seed
    with tempfile.TemporaryDirectory(prefix="flores_guard_") as tmp:
        bad_dir = Path(tmp)
        # devtest present, dev ABSENT -> training must refuse to proceed
        (bad_dir / "guz_devtest.tsv").write_text(
            (FLORES_FIXTURE / "guz_devtest.tsv").read_text(encoding="utf-8"),
            encoding="utf-8")
        cfg = TrainConfig(run_name="t", model_key="mt5_small",
                          direction="all", fewshot_guz=10)
        for fn in (lambda: build_train_dataset(cfg, SPLITS_FIXTURE, bad_dir),
                   lambda: load_flores_seed(bad_dir, 10)):
            try:
                fn()
            except FileNotFoundError as exc:
                assert "devtest" in str(exc).lower() or "guz_dev" in str(exc)
            else:
                raise AssertionError("devtest guard must raise FileNotFoundError")
    print("ok  test_devtest_guard")


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
# 7. scripts/fetch_flores.py compiles
# ---------------------------------------------------------------------------

def test_fetch_flores_compiles():
    script = PROJECT_ROOT / "scripts" / "fetch_flores.py"
    assert script.exists(), f"missing {script}"
    py_compile.compile(str(script), doraise=True)
    print("ok  test_fetch_flores_compiles")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> int:
    """Run all Week 3 data tests; return 0 on success."""
    have_datasets = _datasets_available()

    test_direction_expansion()
    if have_datasets:
        test_load_psa_pairs()
        test_load_flores_seed()
        test_build_train_dataset()
        test_devtest_guard()
    else:
        for name in ("test_load_psa_pairs", "test_load_flores_seed",
                     "test_build_train_dataset", "test_devtest_guard"):
            _skip(name, "python package 'datasets' not installed "
                        "(pip install datasets)")
    test_augment_import_no_torch()
    test_fetch_flores_compiles()

    if _SKIPPED:
        print(f"note: {len(_SKIPPED)} test(s) skipped gracefully:")
        for s in _SKIPPED:
            print(f"  - {s}")
    print("ok  test_week3_data")
    return 0


if __name__ == "__main__":
    sys.exit(run())
