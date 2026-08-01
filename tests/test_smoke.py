#!/usr/bin/env python3
"""Smoke tests for the Week 1 package. Plain asserts, no pytest required.

Run from the project root:  python tests/test_smoke.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Make the project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import build_dataset, report as report_mod  # noqa: E402
from src.cleaning import clean  # noqa: E402
from src.corpora.tico19 import parse_tmx  # noqa: E402
from src.schema import COLUMNS, assign_ids, new_record, validate_schema  # noqa: E402

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
    import src.cleaning as _cleaning
    if _cleaning._ld_detect is None:
        print("skip test_clean (langdetect not installed; language filter "
              "degrades gracefully — install requirements to run fully)")
        return
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
    orig_manual = build_dataset.import_manual
    orig_report_out = report_mod.REPORTS_DIR
    orig_audit_out = build_dataset.REPORTS_DIR
    build_dataset.DATASET_CSV = csv_path
    build_dataset.STATS_JSON = stats_path
    build_dataset.REPORTS_DIR = tmp_out
    build_dataset.import_tico19 = lambda **kw: orig_import(
        tmx_path=FIXTURE_TMX, verbose=False, **{k: v for k, v in kw.items()
                                                if k != "verbose"})
    # Isolate from any real files a developer has dropped into data/manual/
    build_dataset.import_manual = lambda verbose=True: []
    report_mod.REPORTS_DIR = tmp_out
    try:
        out = build_dataset.build(scrape=False, use_tatoeba=False,
                                  use_lecturer=False, verbose=False)
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
        build_dataset.REPORTS_DIR = orig_audit_out
        build_dataset.import_tico19 = orig_import
        build_dataset.import_manual = orig_manual
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
    # Week 2
    test_preprocessing()
    test_splits()
    test_eda_smoke()
    test_manual_import()
    test_run_week2_cli()
    # Week 3 — discovered modules (each exposes run(); self-skipping when
    # optional deps like torch/transformers/datasets are unavailable)
    week3_dir = Path(__file__).parent
    for mod_path in sorted(week3_dir.glob("test_week3_*.py")):
        spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run()
    print("\nALL SMOKE TESTS PASSED")
    return 0




# ---------------------------------------------------------------------------
# Week 2 tests
# ---------------------------------------------------------------------------

def test_preprocessing():
    """normalize_deep, tokenize counts, code-switch detection, glossary hits."""
    from src.preprocessing import (codeswitch_ratio, glossary_hits,
                                   is_codeswitched, load_glossary,
                                   normalize_deep, tokenize, word_tokens)

    # curly quotes/apostrophes -> ASCII, en/em dashes -> '-', whitespace collapsed
    messy = "  “Don’t”  wash – hands — now  "
    fixed = normalize_deep(messy)
    assert fixed == '"Don\'t" wash - hands - now', fixed

    toks = tokenize("Wash your hands, please!")
    assert toks == ["wash", "your", "hands", ",", "please", "!"], toks
    assert len(word_tokens("Wash your hands, please!")) == 4
    assert tokenize("Don't") == ["don't"]

    # mixed EN/SW sentence is detected; pure English is not
    mixed = "We must all report the crime and the suspects kwa polisi sasa."
    pure_en = "Please wash your hands with soap and clean water every day."
    assert is_codeswitched(mixed)
    assert codeswitch_ratio(mixed) > 0.15
    assert not is_codeswitched(pure_en)
    assert codeswitch_ratio("") == 0.0

    glossary = load_glossary()
    assert "harambee" in glossary, "glossary.json missing 'harambee'"
    hits = glossary_hits("Join the Harambee fundraiser at the baraza today.",
                         glossary)
    assert "harambee" in hits and "baraza" in hits, hits
    assert glossary_hits("Wash your hands with clean water.", glossary) == []

    # preprocess_dataframe adds columns without touching originals
    from src.preprocessing import preprocess_dataframe
    df = pd.DataFrame([
        new_record(domain="Health",
                   english="Attend the harambee at the chief's baraza today."),
        new_record(domain="Education",
                   english="Enrol your child in school before the term begins.",
                   kiswahili="Mwingize mtoto wako shuleni kabla ya muhula."),
    ])
    out = preprocess_dataframe(df, glossary=glossary)
    for col in ("English_norm", "Kiswahili_norm", "tokens_en", "tokens_sw",
                "codeswitch", "glossary_terms"):
        assert col in out.columns, col
    assert out.loc[0, "glossary_terms"]  # harambee + baraza
    assert int(out.loc[1, "tokens_sw"]) > 0 and int(out.loc[0, "tokens_sw"]) == 0
    assert list(out["English"]) == list(df["English"])  # originals untouched
    print("ok  test_preprocessing")


def test_splits():
    """100 rows / 4 domains -> exact 90/5/5, stratified, no leakage, seeded."""
    from src.splits import group_key, make_splits

    domains = ["Health", "Education", "Agriculture", "Security"]
    df = pd.DataFrame([
        new_record(domain=domains[i % 4],
                   english=f"Sample announcement number {i} for public awareness.")
        for i in range(100)
    ])
    train, dev, test = make_splits(df, seed=42)
    assert (len(train), len(dev), len(test)) == (90, 5, 5), \
        (len(train), len(dev), len(test))
    # stratified: every domain present in train
    assert set(train["Domain"].unique()) == set(domains)
    # zero group-key overlap between any pair of splits
    keys = {name: set(split["English"].map(group_key))
            for name, split in (("tr", train), ("dv", dev), ("te", test))}
    assert not (keys["tr"] & keys["dv"] or keys["tr"] & keys["te"]
                or keys["dv"] & keys["te"])
    # reproducible with the same seed (identical English rows per split)
    train2, dev2, test2 = make_splits(df, seed=42)
    for a, b in ((train, train2), (dev, dev2), (test, test2)):
        assert list(a["English"]) == list(b["English"])
    print("ok  test_splits")


def test_eda_smoke(tmp_out=None):
    """compute_eda keys, six PNGs written, report contains 'Key observations'."""
    from src.eda import compute_eda, make_figures, write_eda_report

    tmp_out = Path(tmp_out or tempfile.mkdtemp(prefix="psa_eda_"))
    domains = ["Health", "Education", "Agriculture", "Security"]
    df = pd.DataFrame([
        new_record(domain=domains[i % 4],
                   english=f"Public notice number {i} for every citizen today.",
                   kiswahili=("Osha mikono yako kwa maji safi kila siku."
                              if i % 3 == 0 else ""),
                   source="WHO" if i % 2 else "NTSA")
        for i in range(12)
    ])
    stats = compute_eda(df)
    for key in ("rows_total", "per_domain", "per_source", "paired", "unpaired",
                "length_en", "length_sw", "vocab_en", "vocab_sw",
                "type_token_ratio_en", "type_token_ratio_sw",
                "codeswitched_rows", "glossary_rows",
                "missing_translation_share", "per_domain_mean_length"):
        assert key in stats, key
    assert stats["rows_total"] == 12 and stats["paired"] == 4
    assert set(stats["length_en"]) == {"mean", "median", "min", "max"}

    figures = make_figures(df, out_dir=tmp_out / "figures")
    assert len(figures) == 6 and all(p.exists() and p.suffix == ".png"
                                     for p in figures), figures

    report = write_eda_report(stats, figures, out_path=tmp_out / "eda.md")
    text = report.read_text(encoding="utf-8")
    assert "Key observations" in text
    assert "figures/domain_bar.png" in text  # relative figure embedding
    print("ok  test_eda_smoke")


def test_manual_import():
    """Tiny CSV in a temp manual dir -> schema-valid Team-written records.

    Uses a temp dir so real submissions in data/manual/ never affect the
    count. Separately, any real submissions present must import cleanly.
    """
    from src.corpora.manual import import_manual
    from src import config

    manual_dir = Path(tempfile.mkdtemp(prefix="psa_manual_"))
    probe = manual_dir / "zz_test_probe.csv"
    probe.write_text(
        "Domain,English,Kiswahili,Ekegusii,Notes\n"
        "Health,\"Wash your hands with soap before eating.\",,,probe row\n"
        "Education,\"Enrol your child in the nearest public school today.\",,,probe row\n"
        "Space,\"Invalid domain rows are skipped entirely.\",,,bad row\n",
        encoding="utf-8")
    records = import_manual(manual_dir=manual_dir, verbose=False)
    assert len(records) == 2, f"expected 2 records, got {len(records)}"
    assert all(r["Source"] == "Team-written" for r in records)
    assert all(r["Status"] == "Pending" for r in records)
    df = assign_ids(pd.DataFrame(records))
    assert validate_schema(df) == [], validate_schema(df)
    # missing directory -> warn + []
    assert import_manual(manual_dir=manual_dir / "nope",
                         verbose=False) == []

    # Real submissions (if the developer has dropped any into data/manual/)
    # must import and validate cleanly too.
    real_dir = Path(config.EXTERNAL_DIR).parent / "manual"
    real_records = import_manual(manual_dir=real_dir, verbose=False)
    if real_records:
        real_df = assign_ids(pd.DataFrame(real_records))
        assert validate_schema(real_df) == [], validate_schema(real_df)
    print("ok  test_manual_import")


def test_run_week2_cli(tmp_out=None):
    """Full run_week2 pipeline on a small synthetic CSV with redirected paths."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_week2
    from src import config

    tmp_out = Path(tmp_out or tempfile.mkdtemp(prefix="psa_week2_"))
    input_csv = tmp_out / "input.csv"
    domains = ["Health", "Education", "Agriculture", "Security"]
    df = pd.DataFrame([
        new_record(domain=domains[i % 4],
                   english=f"Public notice number {i} for every citizen today.",
                   kiswahili=("Osha mikono yako kwa maji safi kila siku."
                              if i % 3 == 0 else ""),
                   source="WHO" if i % 2 else "NTSA")
        for i in range(40)
    ])
    df = assign_ids(df)
    df.to_csv(input_csv, index=False, encoding="utf-8")

    orig = (config.PROCESSED_DIR, config.REPORTS_DIR, config.BASE_DIR)
    config.PROCESSED_DIR = tmp_out / "processed"
    config.REPORTS_DIR = tmp_out / "reports"
    config.BASE_DIR = tmp_out
    try:
        rc = run_week2.main(["--input", str(input_csv), "--val-size", "12"])
        assert rc == 0
        assert (config.PROCESSED_DIR / "psa_preprocessed.csv").exists()
        assert (config.PROCESSED_DIR / "splits" / "train.csv").exists()
        assert (config.PROCESSED_DIR / "splits" / "dev.csv").exists()
        assert (config.PROCESSED_DIR / "splits" / "test.csv").exists()
        assert (config.PROCESSED_DIR / "splits" / "split_stats.json").exists()
        assert (config.REPORTS_DIR / "week2_eda_report.md").exists()
        figs = list((config.REPORTS_DIR / "figures").glob("*.png"))
        assert len(figs) == 6, figs
        val = config.BASE_DIR / "data" / "validation" / "validation_subset.csv"
        assert val.exists()
        val_df = pd.read_csv(val, dtype=str).fillna("")
        assert len(val_df) == 12
        for col in ("Reviewer", "Fluency_1to5", "Adequacy_1to5", "Issues",
                    "Notes"):
            assert col in val_df.columns, col
        assert (config.BASE_DIR / "docs" / "validation_guide.md").exists()
    finally:
        config.PROCESSED_DIR, config.REPORTS_DIR, config.BASE_DIR = orig
    print("ok  test_run_week2_cli")




def test_collector_verify_ssl():
    """Mocked scrape: verify_ssl defaults True and False passes through (no KeyError)."""
    import src.collectors.base as base
    from bs4 import BeautifulSoup
    from src.collectors.sites import SITES

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
    from src.collectors.base import SiteCollector
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
    import src.collectors.base as base
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
