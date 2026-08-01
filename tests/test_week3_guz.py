#!/usr/bin/env python3
"""Framework audit + Ekegusii tests.

Covers: the lecturer gold loader (fixture CSV), the PSA framework audit step
(src/audit.py) on a tiny synthetic dataset, the full build pipeline with
patched collectors, and guz pair loading in training.data.

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

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LECTURER_FIXTURE = FIXTURES / "lecturer_fixture.csv"
GUZ_SPLITS_FIXTURE = FIXTURES / "week3_guz" / "splits"
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
    from src.corpora.lecturer import load_lecturer
    from src.schema import assign_ids, validate_schema

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


def test_mojibake_repair(tmp_out=None):
    """repair_mojibake: multi-round cp1252/UTF-8 mojibake -> original text."""
    from src.cleaning import normalize_text, repair_mojibake
    from src.corpora.lecturer import load_lecturer

    original = "Schools close early – parents collect children NOW!"
    round1 = original.encode("utf-8").decode("cp1252")
    round3 = round1.encode("utf-8").decode("cp1252") \
                   .encode("utf-8").decode("cp1252")
    assert round1 != original and "Ã" not in round1  # sanity: corruption grew
    assert "Ã" in round3
    assert repair_mojibake(round1) == original
    assert repair_mojibake(round3) == original
    # clean text (incl. plain ASCII and Swahili) is untouched
    for clean in ("Boil water before drinking.",
                  "Chemsha maji kabla ya kunywa.",
                  "Tancha amache mbee okunywa."):
        assert repair_mojibake(clean) == clean
    assert normalize_text("  Boil   water\tfirst ") == "Boil water first"
    # residual patterns (chars substituted/lost before we ever saw the file)
    assert repair_mojibake("Itâ€TMs okay") == "It’s okay"
    assert repair_mojibake("bila malipo '¢'¬â€œ anzisha") == \
        "bila malipo – anzisha"

    # loader-level: corrupted English + Ekegusii cells are repaired on import
    tmp = Path(tmp_out or tempfile.mkdtemp(prefix="psa_mojibake_"))
    fixture = tmp / "lecturer_bad.csv"
    bad_eng = "[Schools] Register now " + round3
    bad_guz = "Andika omwana " + round1
    fixture.write_text(
        "PSA_Id,Domain,Class,English,Kiswahili,Ekegusii,Dholuo,Somali\n"
        f'L9001,Education,PSA,"{bad_eng}",Andikisha mtoto.,"{bad_guz}",,\n',
        encoding="utf-8")
    records = load_lecturer(fixture, verbose=False)
    assert len(records) == 1
    assert records[0]["English"] == f"[Schools] Register now {original}"
    assert records[0]["Ekegusii"] == f"Andika omwana {original}"
    print("ok  test_mojibake_repair")


# ---------------------------------------------------------------------------
# 2. Framework audit step (src/audit.py) on a tiny synthetic dataset
# ---------------------------------------------------------------------------

_SCRAPED_PSA = "Wash your hands with soap and clean water daily."
_SCRAPED_INFO = "However, the disease may present in many atypical forms."
_SCRAPED_PRESS = "KRA wins case against Dubai firm in court"
_CORPUS_INFO = "Chagas disease can be treated with benznidazole or nifurtimox."
_MANUAL_PSA = "Enrol your child in the nearest public school today."
_GOLD_ENCYC = "[Outbreak] Cholera is a disease caused by contaminated water."


def _synthetic_records() -> list[dict]:
    """Six rows across all four source types; the two scraped non-PSA rows
    are the only ones the audit may delete."""
    from src.schema import new_record

    return [
        new_record(domain="Health", english=_SCRAPED_PSA, source="MOH Kenya",
                   metadata={"type": "scraped", "tool": "requests+bs4"}),
        new_record(domain="Health", english=_SCRAPED_INFO, source="HealthBlog",
                   metadata={"type": "scraped"}),
        new_record(domain="Governance", english=_SCRAPED_PRESS, source="NewsSite",
                   metadata={"type": "scraped"}),
        new_record(domain="Health", english=_CORPUS_INFO, source="TICO-19",
                   kiswahili="Ugonjwa wa Chagas unaweza kutibiwa kwa dawa.",
                   metadata={"type": "corpus"}),
        new_record(domain="Education", english=_MANUAL_PSA, source="Team-written",
                   metadata={"type": "manual", "license": "original work"}),
        new_record(domain="Health", english=_GOLD_ENCYC,
                   source="Lecturer dataset (PSA_KE_Final)",
                   metadata={"type": "gold", "psa_class": "PSA"}),
    ]


def test_apply_framework_audit():
    from src.audit import apply_framework_audit, render_audit_report

    df = pd.DataFrame(_synthetic_records())
    kept, info = apply_framework_audit(df, verbose=False)

    texts = set(kept["English"])
    # scraped non-PSA rows deleted entirely
    assert _SCRAPED_INFO not in texts
    assert _SCRAPED_PRESS not in texts
    # corpus/manual/gold rows EXEMPT (kept even when non-PSA)
    for t in (_SCRAPED_PSA, _CORPUS_INFO, _MANUAL_PSA, _GOLD_ENCYC):
        assert t in texts, f"exempt/PSA row missing: {t}"
    assert info["rows_before"] == 6 and info["rows_after"] == 4
    assert info["deleted_total"] == 2 and info["n_scraped"] == 3
    assert info["class_distribution"]["PSA"] >= 2

    # every kept row stamped with psa_class; surviving scraped row == PSA
    metas = kept["Metadata"].map(_meta)
    assert all("psa_class" in m for m in metas)
    types = metas.map(lambda m: m.get("type", ""))
    scraped_meta = metas[types == "scraped"].iloc[0]
    assert scraped_meta["psa_class"] == "PSA", scraped_meta

    # audit report renders the key sections and samples a deleted row
    report = render_audit_report(info, cleaned=kept, lecturer_rows=0)
    for needle in ("Framework audit", "Methodology", "DELETED",
                   "Kept / dropped per source", "Totals"):
        assert needle in report, needle
    assert "KRA wins case" in report
    print("ok  test_apply_framework_audit")


def test_build_pipeline(tmp_out=None):
    """Full build(): patched collector -> audit -> gold merge -> clean."""
    import src.build_dataset as bd

    tmp_out = Path(tmp_out or tempfile.mkdtemp(prefix="psa_build_"))
    scraped_records = _synthetic_records()[:3]  # the three scraped rows

    orig = (bd.DATASET_CSV, bd.STATS_JSON, bd.REPORTS_DIR, bd.collect_all)
    bd.DATASET_CSV = tmp_out / "psa_parallel_week1.csv"
    bd.STATS_JSON = tmp_out / "build_stats.json"
    bd.REPORTS_DIR = tmp_out / "reports"
    bd.collect_all = lambda names=None, max_pages=None, verbose=True: \
        list(scraped_records)
    try:
        csv_path = bd.build(scrape=True, use_tico=False, use_tatoeba=False,
                            use_manual=False, lecturer_path=LECTURER_FIXTURE,
                            verbose=False)
        out = pd.read_csv(csv_path, dtype=str).fillna("")
        texts = set(out["English"])
        assert _SCRAPED_INFO not in texts and _SCRAPED_PRESS not in texts
        assert _SCRAPED_PSA in texts
        # 1 surviving scraped row + 4 unique lecturer fixture rows
        assert len(out) == 5, f"expected 5 rows, got {len(out)}"
        assert list(out["PSA_ID"]) == [f"PSA{i + 1:06d}" for i in range(5)]
        # Ekegusii arrived via the lecturer merge (3 of 4 fixture rows)
        assert int((out["Ekegusii"].str.strip() != "").sum()) == 3

        stats = json.loads(bd.STATS_JSON.read_text(encoding="utf-8"))
        assert stats["output_rows"] == 5
        assert stats["source_counts"]["scraped"] == 3
        assert stats["source_counts"]["lecturer_gold"] == 4
        assert stats["framework_audit"]["deleted_total"] == 2
        assert stats["output_rows_per_type"] == {"gold": 4, "scraped": 1}, \
            stats["output_rows_per_type"]

        audit = bd.REPORTS_DIR / "framework_audit.md"
        assert audit.exists(), audit
        text = audit.read_text(encoding="utf-8")
        for needle in ("Framework audit", "Methodology", "DELETED",
                       "Lecturer gold merge"):
            assert needle in text, needle
        assert "KRA wins case" in text  # deleted row sampled in the audit
    finally:
        (bd.DATASET_CSV, bd.STATS_JSON, bd.REPORTS_DIR,
         bd.collect_all) = orig
    print("ok  test_build_pipeline")


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

    # fewshot_guz=-1 -> ALL PSA-sourced guz pairs (the only guz source)
    cfg = TrainConfig(direction="en-guz", fewshot_guz=-1, **base)
    ds = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE)
    assert len(ds) == 3, f"expected all 3 PSA guz pairs, got {len(ds)}"
    assert set(ds["provenance"]) == {"psa"}

    # fewshot_guz=2 -> seeded cap of 2 on PSA guz pairs
    cfg = TrainConfig(direction="en-guz", fewshot_guz=2, **base)
    ds = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE)
    assert len(ds) == 2, f"expected capped 2 pairs, got {len(ds)}"
    assert set(ds["provenance"]) == {"psa"}
    # cap is deterministic
    ds2 = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE)
    assert ds["src_text"] == ds2["src_text"]

    # fewshot_guz=0 -> guz pairs excluded entirely
    cfg = TrainConfig(direction="en-sw", fewshot_guz=0, **base)
    ds = build_train_dataset(cfg, GUZ_SPLITS_FIXTURE)
    assert set(ds["tgt_lang"]) == {"swa"}

    # guard: few-shot guz requested but the train split has NO Ekegusii ->
    # must raise (the benchmark TSV is evaluation-only, never trainable)
    cfg = TrainConfig(direction="en-guz", fewshot_guz=2, **base)
    try:
        build_train_dataset(cfg, SPLITS_FIXTURE)
    except FileNotFoundError as exc:
        assert "ekegusii" in str(exc).lower() or "benchmark" in str(exc).lower()
    else:
        raise AssertionError("guz guard must raise FileNotFoundError")
    print("ok  test_build_train_dataset_guz")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> int:
    """Run all audit/guz tests; return 0 on success."""
    test_lecturer_loader()
    test_mojibake_repair()
    test_apply_framework_audit()
    test_build_pipeline()
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
