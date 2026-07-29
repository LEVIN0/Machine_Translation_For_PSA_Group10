#!/usr/bin/env python3
"""Remediation + Ekegusii tests (SPEC_REMEDIATION.md §5).

Covers: lecturer gold loader (fixture CSV), the one-command remediation
pipeline on a tiny synthetic dataset, and guz pair loading in training.data.

Plain asserts, no pytest required; offline, fast. Exposes run() which
prints "ok  test_week3_guz" on success. Dataset-builder tests SKIP
gracefully when the optional 'datasets' package is unavailable.

Run from the project root:  python tests/test_week3_guz.py
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LECTURER_FIXTURE = FIXTURES / "lecturer_fixture.csv"
GUZ_SPLITS_FIXTURE = FIXTURES / "week3_guz" / "splits"
FLORES_FIXTURE = FIXTURES / "flores"

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


def _meta(cell):
    """Parse a Metadata cell (JSON string or dict) into a dict."""
    if isinstance(cell, dict):
        return cell
    text = str(cell or "").strip()
    return json.loads(text) if text else {}


# ---------------------------------------------------------------------------
# 1. Lecturer gold loader on the fixture CSV
# ---------------------------------------------------------------------------

def test_lecturer_loader():
    from SRC.corpora.lecturer import load_lecturer
    from SRC.schema import assign_ids, validate_schema

    records = load_lecturer(LECTURER_FIXTURE, verbose=False)
    # fixture: 5 data rows, 1 internal dupe -> 4 records
    assert len(records) == 4, f"expected 4 records (dupe collapsed), got {len(records)}"

    metas = [_meta(r["Metadata"]) for r in records]
    for rec, m in zip(records, metas):
        assert rec["Source"] == "Lecturer dataset (PSA_KE_Final)"
        assert rec["Status"] == "Validated"
        assert rec["Date"] == "" and rec["URL"] == ""
        assert m["type"] == "gold"
        assert m["license"] == "lecturer-provided"
        assert m["psa_class"] == "PSA"
        assert m["lecturer_id"].startswith("L"), m

    # topic tag extracted into Metadata.topic; text kept verbatim (with [Tag])
    assert metas[0]["topic"] == "Cholera", metas[0]
    assert records[0]["English"].startswith("[Cholera] "), records[0]["English"]
    assert metas[1]["topic"] == "Schools"

    # dholuo/somali keys omitted when empty; present when filled
    assert "dholuo" in metas[0] and "somali" in metas[0]
    assert "dholuo" not in metas[1] and "somali" not in metas[1]
    assert "somali" in metas[3] and "dholuo" not in metas[3]

    # one fixture row has empty Ekegusii (L0003)
    assert records[2]["Ekegusii"] == "" and records[0]["Ekegusii"] != ""

    # dupe collapse: the L0001 text appears exactly once
    texts = [r["English"] for r in records]
    assert len(set(texts)) == len(texts)

    df = assign_ids(pd.DataFrame(records))
    assert validate_schema(df) == [], validate_schema(df)

    # missing file -> warn + []
    assert load_lecturer(LECTURER_FIXTURE.parent / "nope.csv",
                         verbose=False) == []
    print("ok  test_lecturer_loader")


# ---------------------------------------------------------------------------
# 2. Remediation pipeline on a tiny synthetic dataset
# ---------------------------------------------------------------------------

def _write_synthetic_dataset(csv_path: Path) -> list[str]:
    """Write a tiny dataset CSV with scraped/corpus/manual/gold rows;
    return the English texts expected to SURVIVE remediation."""
    from SRC.schema import assign_ids, new_record

    scraped_psa = "Wash your hands with soap and clean water daily."
    scraped_info = "However, the disease may present in many atypical forms."
    scraped_press = "KRA wins case against Dubai firm in court"
    corpus_info = "Chagas disease can be treated with benznidazole or nifurtimox."
    manual_psa = "Enrol your child in the nearest public school today."
    gold_encyc = "[Outbreak] Cholera is a disease caused by contaminated water."
    df = pd.DataFrame([
        new_record(domain="Health", english=scraped_psa, source="MOH Kenya",
                   metadata={"type": "scraped", "tool": "requests+bs4"}),
        new_record(domain="Health", english=scraped_info, source="HealthBlog",
                   metadata={"type": "scraped"}),
        new_record(domain="Governance", english=scraped_press, source="NewsSite",
                   metadata={"type": "scraped"}),
        new_record(domain="Health", english=corpus_info, source="TICO-19",
                   kiswahili="Ugonjwa wa Chagas unaweza kutibiwa kwa dawa.",
                   metadata={"type": "corpus"}),
        new_record(domain="Education", english=manual_psa, source="Team-written",
                   metadata={"type": "manual", "license": "original work"}),
        new_record(domain="Health", english=gold_encyc,
                   source="Lecturer dataset (PSA_KE_Final)",
                   metadata={"type": "gold", "psa_class": "PSA"}),
    ])
    assign_ids(df).to_csv(csv_path, index=False, encoding="utf-8")
    return [scraped_psa, corpus_info, manual_psa, gold_encyc]


def test_remediate_pipeline(tmp_out=None):
    import remediate_dataset
    from SRC import config

    tmp_out = Path(tmp_out or tempfile.mkdtemp(prefix="psa_remediate_"))
    csv_path = tmp_out / "psa_parallel_week1.csv"
    survivors = _write_synthetic_dataset(csv_path)

    orig_reports = config.REPORTS_DIR
    config.REPORTS_DIR = tmp_out / "reports"
    try:
        rc = remediate_dataset.main([
            "--lecturer", str(LECTURER_FIXTURE),
            "--dataset", str(csv_path),
        ])
        assert rc == 0, "remediate_dataset.main must return 0"

        out = pd.read_csv(csv_path, dtype=str).fillna("")
        texts = set(out["English"])
        # scraped non-PSA rows deleted entirely
        assert "However, the disease may present in many atypical forms." not in texts
        assert "KRA wins case against Dubai firm in court" not in texts
        # corpus/manual/gold rows EXEMPT (kept even when non-PSA)
        for t in survivors:
            assert t in texts, f"exempt/PSA row missing: {t}"
        # lecturer gold merged (4 unique fixture rows) -> 4 + 4 = 8 rows
        assert len(out) == 8, f"expected 8 rows, got {len(out)}"
        # IDs re-assigned contiguously from PSA000001
        assert list(out["PSA_ID"]) == [f"PSA{i + 1:06d}" for i in range(len(out))]

        metas = out["Metadata"].map(_meta)
        types = metas.map(lambda m: m.get("type", ""))
        assert dict(types.value_counts()) == {
            "gold": 5, "scraped": 1, "corpus": 1, "manual": 1}, \
            dict(types.value_counts())
        # every row stamped with psa_class; surviving scraped row == PSA
        assert all("psa_class" in m for m in metas)
        scraped_meta = metas[types == "scraped"].iloc[0]
        assert scraped_meta["psa_class"] == "PSA", scraped_meta
        # Ekegusii arrived via the lecturer merge (3 of 4 fixture rows)
        assert int((out["Ekegusii"].str.strip() != "").sum()) == 3

        # audit report written
        audit = tmp_out / "reports" / "framework_audit.md"
        assert audit.exists(), audit
        text = audit.read_text(encoding="utf-8")
        for needle in ("Framework audit", "Methodology", "DELETED",
                       "Lecturer", "Ekegusii"):
            assert needle in text, needle
        assert "KRA wins case" in text  # deleted row sampled in the audit

        # build_stats.json next to the dataset
        stats = json.loads((tmp_out / "build_stats.json")
                           .read_text(encoding="utf-8"))
        assert stats["mode"] == "remediate"
        assert stats["rows_before"] == 6 and stats["rows_after"] == 8
        assert stats["deleted_per_source"] == {"scraped": 2}
        assert stats["lecturer_rows"] == 4
        assert stats["class_distribution"]["PSA"] >= 2

        # --dry-run: CSV untouched, report + stats still written
        before_bytes = csv_path.read_bytes()
        rc = remediate_dataset.main([
            "--lecturer", str(LECTURER_FIXTURE),
            "--dataset", str(csv_path), "--dry-run",
        ])
        assert rc == 0
        assert csv_path.read_bytes() == before_bytes, \
            "dry-run must not modify the dataset CSV"
        assert audit.exists() and (tmp_out / "build_stats.json").exists()
    finally:
        config.REPORTS_DIR = orig_reports
    print("ok  test_remediate_pipeline")


# ---------------------------------------------------------------------------
# 3. guz pair loading from fixture splits (needs 'datasets')
# ---------------------------------------------------------------------------

def test_guz_pairs():
    from training.data import load_psa_pairs

    en_guz = load_psa_pairs(GUZ_SPLITS_FIXTURE, "train", ["en-guz"])
    # fixture: 5 rows, 3 with non-empty Ekegusii -> 3 en-guz pairs
    assert len(en_guz) == 3, f"expected 3 en-guz pairs, got {len(en_guz)}"
    assert set(en_guz["src_lang"]) == {"eng"}
    assert set(en_guz["tgt_lang"]) == {"guz"}
    assert set(en_guz["provenance"]) == {"psa"}

    sw_guz = load_psa_pairs(GUZ_SPLITS_FIXTURE, "train", ["sw-guz"])
    # sw-guz only when Kiswahili non-empty: 2 of the 3 guz rows
    assert len(sw_guz) == 2, f"expected 2 sw-guz pairs, got {len(sw_guz)}"
    assert set(sw_guz["src_lang"]) == {"swa"}
    assert set(sw_guz["tgt_lang"]) == {"guz"}
    assert set(sw_guz["provenance"]) == {"psa"}

    # guz rows line up with the fixture Ekegusii cells
    assert set(en_guz["tgt_text"]) == {
        "Tancha amache mbee okunywa kila risonde.",
        "Andika omwana wawe ego esomero.",
        "Ronda obusuma mbee kiora.",
    }
    print("ok  test_guz_pairs")


def test_build_train_dataset_guz(tmp_out=None):
    from training.config import TrainConfig
    from training.data import build_train_dataset

    base = dict(run_name="t", model_key="mt5_small")
    no_flores = Path(tmp_out or tempfile.mkdtemp(prefix="psa_noflores_"))

    # fewshot_guz=-1 -> ALL PSA-sourced guz pairs; FLORES not required
    cfg = TrainConfig(direction="en-guz", fewshot_guz=-1, **base)
    ds = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE, no_flores)
    assert len(ds) == 3, f"expected all 3 PSA guz pairs, got {len(ds)}"
    assert set(ds["provenance"]) == {"psa"}

    # fewshot_guz=2 -> seeded cap of 2 on PSA guz pairs; flores missing but
    # PSA pairs exist, so NO raise (FLORES dev is now an optional extra)
    cfg = TrainConfig(direction="en-guz", fewshot_guz=2, **base)
    ds = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE, no_flores)
    assert len(ds) == 2, f"expected capped 2 pairs, got {len(ds)}"
    assert set(ds["provenance"]) == {"psa"}
    # cap is deterministic
    ds2 = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE, no_flores)
    assert ds["src_text"] == ds2["src_text"]

    # fewshot_guz=2 WITH flores dev present -> 2 PSA pairs + 2 flores seeds
    cfg = TrainConfig(direction="en-guz", fewshot_guz=2, **base)
    ds = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE, FLORES_FIXTURE)
    assert len(ds) == 4, f"expected 2 psa + 2 flores pairs, got {len(ds)}"
    assert set(ds["provenance"]) == {"psa", "flores_dev_seed"}

    # fewshot_guz=0 -> guz pairs excluded entirely
    cfg = TrainConfig(direction="en-sw", fewshot_guz=0, **base)
    ds = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE, FLORES_FIXTURE)
    assert set(ds["tgt_lang"]) == {"swa"}
    print("ok  test_build_train_dataset_guz")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> int:
    """Run all remediation/guz tests; return 0 on success."""
    test_lecturer_loader()
    test_remediate_pipeline()
    if _datasets_available():
        test_guz_pairs()
        test_build_train_dataset_guz()
    else:
        for name in ("test_guz_pairs", "test_build_train_dataset_guz"):
            _skip(name, "python package 'datasets' not installed "
                        "(pip install datasets)")

    if _SKIPPED:
        print(f"note: {len(_SKIPPED)} test(s) skipped gracefully:")
        for s in _SKIPPED:
            print(f"  - {s}")
    print("ok  test_week3_guz")
    return 0


if __name__ == "__main__":
    sys.exit(run())
