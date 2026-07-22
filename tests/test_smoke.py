#!/usr/bin/env python3
"""Smoke tests for the Week 1 package. Plain asserts, no pytest required.

Run from the project root:  python tests/test_smoke.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

# Make the project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SRC import build_dataset, report as report_mod  # noqa: E402
from SRC.cleaning import clean  # noqa: E402
from SRC.corpora.tico19 import parse_tmx  # noqa: E402
from SRC.schema import COLUMNS, assign_ids, new_record, validate_schema  # noqa: E402

FIXTURE_TMX = Path(__file__).parent / "fixtures" / "sample_en-sw.tmx"


def test_parse_tmx():
    """Fixture has 3 <tu>s; the one missing the sw <tuv> must be skipped."""
    pairs = parse_tmx(FIXTURE_TMX)
    assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}"
    en, sw = pairs[0]
    assert en.startswith("Wash your hands")
    assert sw.startswith("Osha mikono")
    print("ok  test_parse_tmx")


def test_schema():
    """new_record shape, exact column order, ID assignment, validation flags."""
    rec = new_record(domain="Health", english="Wash your hands with soap.",
                     metadata={"type": "scraped"})
    assert list(rec.keys()) == COLUMNS
    assert rec["Status"] == "Pending"
    assert rec["Ekegusii"] == ""
    assert rec["Metadata"].startswith("{")

    df = pd.DataFrame([
        new_record(domain="Health", english="Keep your surroundings clean today."),
        new_record(domain="Education", english="Schools reopen on Monday next week."),
    ])
    df = assign_ids(df)
    assert list(df["PSA_ID"]) == ["PSA000001", "PSA000002"]
    assert validate_schema(df) == []

    bad = pd.DataFrame([
        new_record(domain="Space", english="This domain is not allowed here."),
    ])
    problems = validate_schema(bad)
    assert any("invalid domains" in p for p in problems), problems

    missing = pd.DataFrame({"English": ["hello there"]})
    assert any("missing columns" in p for p in validate_schema(missing))
    print("ok  test_schema")


def test_clean():
    """Duplicate, too-short, and French-posing-as-English rows are removed."""
    df = pd.DataFrame([
        new_record(domain="Health",
                   english="Wash your hands with soap and clean water regularly."),
        new_record(domain="Health",
                   english="Wash your hands with soap and clean water regularly."),
        new_record(domain="Health", english="Too short."),                      # < MIN_WORDS
        new_record(domain="Health",                                             # French, >= 10 words
                   english="Veuillez vous laver les mains régulièrement avec "
                          "du savon et de l'eau propre chaque jour."),
        new_record(domain="Health",
                   english="Seek medical care immediately if you develop a high fever."),
    ])
    cleaned, stats = clean(df)
    assert stats["input"] == 5
    assert len(cleaned) == 2, f"expected 2 survivors, got {len(cleaned)}"
    assert stats["removed_total"] == 3
    assert stats["after_empty"] == 5       # no empty English rows in fixture
    assert stats["after_length"] == 4      # too-short row dropped
    assert stats["after_dedupe"] == 3      # duplicate dropped
    assert stats["output"] == 2            # French row dropped by lang check
    print("ok  test_clean")


def test_full_build_and_report(tmp_out=None):
    """Full pipeline: build with tico pointed at the fixture, then report."""
    tmp_out = Path(tmp_out or tempfile.mkdtemp(prefix="psa_test_"))
    csv_path = tmp_out / "psa_parallel_week1.csv"
    stats_path = tmp_out / "build_stats.json"
    report_path = tmp_out / "week1_report.md"

    # Redirect outputs into a temp dir; point the TMX importer at the fixture.
    orig_csv, orig_stats = build_dataset.DATASET_CSV, build_dataset.STATS_JSON
    orig_import = build_dataset.import_tico19
    orig_report_out = report_mod.REPORTS_DIR
    build_dataset.DATASET_CSV = csv_path
    build_dataset.STATS_JSON = stats_path
    build_dataset.import_tico19 = lambda **kw: orig_import(
        tmx_path=FIXTURE_TMX, verbose=False, **{k: v for k, v in kw.items()
                                                if k != "verbose"})
    report_mod.REPORTS_DIR = tmp_out
    try:
        out = build_dataset.build(scrape=False, use_tatoeba=False, verbose=False)
        assert out == csv_path and csv_path.exists()

        df = pd.read_csv(csv_path, dtype=str).fillna("")
        assert list(df.columns) == COLUMNS, list(df.columns)
        assert len(df) == 2, f"expected 2 fixture pairs, got {len(df)}"
        assert list(df["PSA_ID"]) == ["PSA000001", "PSA000002"]
        assert (df["Kiswahili"] != "").all()
        assert (df["Ekegusii"] == "").all()
        assert stats_path.exists()

        rep = report_mod.generate_report(csv_path=csv_path,
                                         out_path=report_path)
        assert rep.exists()
        text = rep.read_text(encoding="utf-8")
        assert "Domain" in text and "Challenges" in text
    finally:
        build_dataset.DATASET_CSV = orig_csv
        build_dataset.STATS_JSON = orig_stats
        build_dataset.import_tico19 = orig_import
        report_mod.REPORTS_DIR = orig_report_out
    print("ok  test_full_build_and_report")


def main():
    """Run all smoke tests; exit non-zero on the first failure."""
    test_parse_tmx()
    test_schema()
    test_clean()
    test_full_build_and_report()
    test_collector_verify_ssl()
    test_pagination_expansion()
    test_collector_caps_and_caption_filter()
    print("\nALL SMOKE TESTS PASSED")
    return 0




def test_collector_verify_ssl():
    """Mocked scrape: verify_ssl defaults True and False passes through (no KeyError)."""
    import SRC.collectors.base as base
    from bs4 import BeautifulSoup
    from SRC.collectors.sites import SITES

    calls = []
    def fake_get_soup(url, verify=True):
        calls.append(verify)
        if "category/news" in url or "example.go.ke/news" in url:
            return BeautifulSoup(
                '<a href="https://redcross.or.ke/some-article/">a</a>'
                '<a href="https://example.go.ke/node/123">b</a>', "lxml")
        return BeautifulSoup(
            "<article><p>Wash your hands with soap and clean water regularly.</p></article>",
            "lxml")

    redcross = dict([s for s in SITES if s["name"] == "redcross_news"][0])
    synthetic = {
        "name": "synthetic_nossl", "domain": "Health", "source": "Test",
        "start_urls": ["https://example.go.ke/news"],
        "link_patterns": [r"example\.go\.ke/node/\d+"],
        "content_selectors": ["article p", "p"],
        "verify_ssl": False,
    }

    orig = base.get_soup
    base.get_soup = fake_get_soup
    try:
        for cfg, expected in ((redcross, True), (synthetic, False)):
            recs = base.SiteCollector(cfg).collect(verbose=False)
            assert recs, f"{cfg['name']}: no records from mocked scrape"
            assert set(calls) == {expected}, f"{cfg['name']}: verify flags {set(calls)}"
            calls.clear()
    finally:
        base.get_soup = orig
    print("ok  test_collector_verify_ssl")




def test_pagination_expansion():
    """Pagination templates are expanded and appended to start_urls."""
    from SRC.collectors.base import SiteCollector
    cfg = {
        "name": "pag_test", "domain": "Health", "source": "T",
        "start_urls": ["https://x.go.ke/list"],
        "link_patterns": [],
        "pagination": {"template": "https://x.go.ke/list?page={n}", "start": 1, "pages": 3},
    }
    urls = SiteCollector(cfg)._page_urls(verbose=False)
    assert urls == ["https://x.go.ke/list",
                    "https://x.go.ke/list?page=1",
                    "https://x.go.ke/list?page=2",
                    "https://x.go.ke/list?page=3"], urls
    print("ok  test_pagination_expansion")




def test_collector_caps_and_caption_filter():
    """max_records stops early; min_words and circled-C caption filter apply."""
    import SRC.collectors.base as base
    from bs4 import BeautifulSoup
    calls = []
    def fake_soup(url, verify=True):
        calls.append(url)
        return BeautifulSoup(
            "<article>"
            "<p>Real size: 6 to 15 mm long. Ⓒ A.M.</p>"
            "<p>Short cap .</p>"
            "<p>Wash your hands with soap and clean water regularly.</p>"
            "<p>Report any suspicious activity to the nearest police station today.</p>"
            "</article>", "lxml")
    orig = base.get_soup
    base.get_soup = fake_soup
    try:
        cfg = {"name": "t", "domain": "Health", "source": "T",
               "start_urls": ["https://x.ke/a", "https://x.ke/b"],
               "link_patterns": [], "min_words": 7, "max_records": 2}
        recs = base.SiteCollector(cfg).collect(verbose=False)
        assert len(recs) == 2
        assert all(("Wash" in r["English"]) or ("Report" in r["English"]) for r in recs)
        assert calls == ["https://x.ke/a"]
    finally:
        base.get_soup = orig
    print("ok  test_collector_caps_and_caption_filter")


if __name__ == "__main__":
    raise SystemExit(main())
