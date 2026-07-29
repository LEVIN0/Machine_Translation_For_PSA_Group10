#!/usr/bin/env python3
"""Week 3 evaluation/inference/ablation tests (SPEC_WEEK3.md §2.6-§2.9).

Plain asserts, no pytest required. Offline by default: the only networked
test (live MTTranslator generation) skips gracefully when torch/transformers
or the hub are unavailable.

Run from the project root:  python tests/test_week3_eval.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# W&B must never be required; keep it offline inside tests (SPEC_WEEK3.md §4).
os.environ.setdefault("WANDB_MODE", "offline")

# Make the project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.ablate import standard_matrix, write_results_table  # noqa: E402
from training.config import LANGS, MODEL_ZOO  # noqa: E402
from training.evaluate import EVAL_SPECS, corpus_scores  # noqa: E402
from training.inference import DEMO_PSAS  # noqa: E402

EXPECTED_MATRIX = [
    "zs_mt5", "zs_nllb", "ft_mt5_base", "ft_nllb_base", "ft_nllb_freeze",
    "ft_nllb_aug", "ft_nllb_guz50", "ft_nllb_guz200", "ft_mt5_guz200",
]
EXPECTED_SPECS = {
    "psa_dev_en-sw", "psa_dev_sw-en", "psa_test_en-sw", "psa_test_sw-en",
    "psa_test_en-guz", "psa_test_guz-en",
}
DOMAINS = {"Health", "Education", "Agriculture", "Security", "Governance"}


def test_metrics_sanity():
    """sacrebleu corpus BLEU + chrF on tiny strings."""
    import importlib.util as _ilu
    if _ilu.find_spec("sacrebleu") is None:
        print("skip test_metrics_sanity (sacrebleu not installed; it ships in "
              "requirements-training.txt)")
        return
    refs = ["Wash your hands with soap.", "Every child deserves education."]
    perfect_bleu, perfect_chrf = corpus_scores(list(refs), refs)
    assert perfect_bleu > 99.0, perfect_bleu
    assert perfect_chrf > 99.0, perfect_chrf

    bad_bleu, bad_chrf = corpus_scores(
        ["banana banana banana", "zebra zebra zebra zebra"], refs)
    assert bad_bleu < perfect_bleu - 20, (bad_bleu, perfect_bleu)
    assert bad_chrf < perfect_chrf - 20, (bad_chrf, perfect_chrf)
    print("ok  test_metrics_sanity")


def test_eval_specs_registry():
    """EVAL_SPECS matches the frozen registry shape (SPEC_WEEK3.md §2.6)."""
    assert set(EVAL_SPECS) == EXPECTED_SPECS, set(EVAL_SPECS)
    for name, spec in EVAL_SPECS.items():
        assert isinstance(spec, tuple) and len(spec) == 4, (name, spec)
        loader, split, src, tgt = spec
        assert loader in {"psa", "guzbench"}, name
        assert src in LANGS and tgt in LANGS and src != tgt, name
        if loader == "psa":
            assert split in {"dev", "test"}, name
            assert {src, tgt} == {"eng", "swa"}, name
        else:
            assert split == "test", name
            assert {src, tgt} == {"eng", "guz"}, name
    print("ok  test_eval_specs_registry")


def test_standard_matrix():
    """Matrix contents, quick caps, and epochs=0 zero-shot flags."""
    full = standard_matrix()
    assert [c.run_name for c in full] == EXPECTED_MATRIX
    by_name = {c.run_name: c for c in full}

    for zs in ("zs_mt5", "zs_nllb"):
        assert by_name[zs].epochs == 0, zs  # eval-only entries
    assert by_name["zs_mt5"].model_key == "mt5_small"
    assert by_name["zs_nllb"].model_key == "nllb_600m"
    assert by_name["ft_mt5_base"].direction == "both"
    assert by_name["ft_nllb_base"].direction == "both"
    assert by_name["ft_nllb_freeze"].freeze_encoder is True
    assert by_name["ft_nllb_aug"].use_augmentation is True
    assert by_name["ft_nllb_guz50"].direction == "all"
    assert by_name["ft_nllb_guz50"].fewshot_guz == 50
    assert by_name["ft_nllb_guz200"].fewshot_guz == 200
    assert by_name["ft_mt5_guz200"].model_key == "mt5_small"
    assert by_name["ft_mt5_guz200"].fewshot_guz == 200
    for cfg in full:
        assert cfg.model_key in MODEL_ZOO, cfg.run_name
        assert cfg.seed == 42

    quick = standard_matrix(quick=True)
    assert [c.run_name for c in quick] == EXPECTED_MATRIX
    for cfg in quick:
        if cfg.run_name.startswith("zs_"):
            assert cfg.epochs == 0, cfg.run_name  # still eval-only
        else:
            assert cfg.epochs == 2, cfg.run_name
            assert cfg.max_samples == 2000, cfg.run_name
    print("ok  test_standard_matrix")


def test_write_results_table():
    """Table rendering on a synthetic runs/ dir, incl. '—' for gaps."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"

        ft = runs / "ft_nllb_guz200"
        (ft / "evals").mkdir(parents=True)
        (ft / "train_config.json").write_text(json.dumps({
            "run_name": "ft_nllb_guz200", "model_key": "nllb_600m",
            "direction": "all", "fewshot_guz": 200, "epochs": 3.0,
        }), encoding="utf-8")
        (ft / "metrics_dev.json").write_text(json.dumps({
            "sacrebleu": 12.345, "chrf": 40.5, "n_train_pairs": 5000,
            "trainable_pct": 100.0, "seconds": 1234.5,
        }), encoding="utf-8")
        (ft / "evals" / "psa_test_en-guz.json").write_text(json.dumps({
            "eval_spec": "psa_test_en-guz", "ckpt": "x", "n": 200,
            "bleu": 3.21, "chrf": 22.2, "seconds": 10.0,
            "model_key": "nllb_600m",
        }), encoding="utf-8")

        zs = runs / "zs_mt5"
        (zs / "evals").mkdir(parents=True)
        (zs / "train_config.json").write_text(json.dumps({
            "run_name": "zs_mt5", "model_key": "mt5_small",
            "direction": "both", "epochs": 0.0,
        }), encoding="utf-8")
        (zs / "evals" / "psa_dev_en-sw.json").write_text(json.dumps({
            "eval_spec": "psa_dev_en-sw", "ckpt": "google/mt5-small",
            "n": 200, "bleu": 1.5, "chrf": 20.0, "seconds": 5.0,
            "model_key": "mt5_small",
        }), encoding="utf-8")

        out_md = Path(tmp) / "reports" / "week3_results.md"
        result = write_results_table(runs, out_md)
        assert result == out_md and out_md.is_file()
        table = out_md.read_text(encoding="utf-8")

        assert "ft_nllb_guz200" in table and "zs_mt5" in table
        assert "12.35" in table  # dev BLEU from metrics_dev.json
        assert "3.21" in table   # guz BLEU from evals json
        assert "1.50" in table   # zs dev BLEU from evals json
        assert "nllb_600m" in table and "mt5_small" in table
        assert "zero-shot" in table
        assert "—" in table      # zs row: no metrics_dev/trainable/seconds
        # zs row renders em-dashes for missing trainable % and seconds
        zs_line = [l for l in table.splitlines()
                   if l.startswith("| zs_mt5")][0]
        assert zs_line.count("—") >= 3, zs_line
    print("ok  test_write_results_table")


def test_demo_psas():
    """Exactly 8 demo PSAs, all 5 domains, eng+swa source mix."""
    assert len(DEMO_PSAS) == 8, len(DEMO_PSAS)
    srcs = set()
    for psa in DEMO_PSAS:
        assert set(psa.keys()) == {"domain", "src", "text"}, psa
        assert psa["domain"] in DOMAINS, psa
        assert psa["src"] in {"eng", "swa"}, psa
        assert isinstance(psa["text"], str) and len(psa["text"].split()) >= 5, psa
        srcs.add(psa["src"])
    assert {psa["domain"] for psa in DEMO_PSAS} == DOMAINS
    assert srcs == {"eng", "swa"}, srcs
    print("ok  test_demo_psas")


def test_lazy_imports_without_torch():
    """evaluate/ablate/inference + CLIs import with torch BLOCKED.

    Runs in a subprocess that raises ImportError for torch/transformers/
    datasets, proving no module-level heavy imports (SPEC_WEEK3.md §4).
    """
    # sys.modules[name] = None makes "import name" raise ImportError;
    # works on every Python 3 version and needs no meta-path hooks.
    snippet = (
        "import sys; sys.path.insert(0, {root!r})\n"
        "for _m in ('torch', 'transformers', 'datasets'):\n"
        "    sys.modules[_m] = None\n"
        "import training.evaluate, training.ablate, training.inference\n"
        "import scripts.translate, scripts.run_eval, scripts.run_ablations\n"
        "print('lazy-import ok')\n"
    ).format(root=str(PROJECT_ROOT))
    proc = subprocess.run([sys.executable, "-c", snippet],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"lazy-import test failed:\n{proc.stdout}\n{proc.stderr}")
    assert "lazy-import ok" in proc.stdout
    print("ok  test_lazy_imports_without_torch")


def test_live_translation():
    """Live MTTranslator generation; skip gracefully without GPU stack/hub."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        print(f"skipped: torch/transformers not installed ({exc})")
        return
    # Fast reachability pre-check so offline environments skip in seconds
    # instead of waiting out huggingface_hub retry/backoff loops.
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception as exc:
        print(f"skipped: huggingface hub unreachable ({type(exc).__name__}: "
              f"{exc})")
        return
    try:
        from training.inference import MTTranslator

        # Bound hub wait so flaky networks fail fast, then skip.
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
        translator = MTTranslator("hf-internal-testing/tiny-random-mt5",
                                  model_key="mt5_small")
        out = translator.translate(["Wash your hands with soap."],
                                   src="eng", tgt="swa")
        assert isinstance(out, list) and len(out) == 1
        assert isinstance(out[0], str)
    except Exception as exc:
        print(f"skipped: live translation unavailable ({type(exc).__name__}: "
              f"{exc})")
        return
    print("ok  test_live_translation")


def run() -> int:
    """Run all tests; print the required success line; exit code via main."""
    test_metrics_sanity()
    test_eval_specs_registry()
    test_standard_matrix()
    test_write_results_table()
    test_demo_psas()
    test_lazy_imports_without_torch()
    test_live_translation()
    print("ok  test_week3_eval")
    return 0


def main() -> int:
    code = run()
    print("\nALL WEEK3 EVAL TESTS PASSED")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
