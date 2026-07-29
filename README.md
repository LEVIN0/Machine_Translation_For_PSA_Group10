# Machine Translation of Public Service Announcements (PSAs) — Group 10

**DSA4020A Natural Language Processing — Semester Project (Summer 2026)**

A proof-of-concept multilingual machine translation system for Public Service
Announcements in Kenya. The system translates between **English / Kiswahili** and
**Ekegusii**, an under-resourced indigenous language, making government information
more accessible. The final deliverable will be a deployable web app demonstrating
few-shot cross-lingual transfer learning on our curated PSA dataset.

## Team

| Member | Role |
|--------|------|
| Claire Mwarari | Scraping, site configuration & pipeline development |
| Levin Ekuam | Corpus curation (TICO-19, Tatoeba) & cleaning/validation |
| Paul | Data collection support, documentation & repo management |

## Project timeline

| Week | Focus | Status |
|------|-------|--------|
| 1 | Data collection & curation | ✅ Complete |
| 2 | Preprocessing & EDA | ✅ Complete |
| 3 | Modeling with transfer learning (mT5 / NLLB / mBART) | ⬜ Next |
| 4 | Evaluation, deployment & documentation | ⬜ |

## Dataset at a glance

**13,519 rows** across the 5 brief domains, built from **18 data sources**
(16 scraping sources currently producing + the TICO-19 corpus + team-written
PSAs; 21 site configs in total — see `docs/sources.md` for the full registry
including blocked/dry sources).

| Domain | Rows | Share |
|--------|-----:|------:|
| Health | 8,655 | 64.0% |
| Agriculture | 2,409 | 17.8% |
| Governance | 1,386 | 10.3% |
| Security | 654 | 4.8% |
| Education | 415 | 3.1% |

- **3,137 rows (23.2%) are genuine EN–SW parallel pairs** (TICO-19 + 150
  team-written pairs); the rest are English-only pending translation work.
- Train/dev/test splits (90/5/5, stratified by domain, seed 42, zero text
  leakage): **12,167 / 676 / 676** rows in `data/processed/splits/`.
- A 500-row native-speaker validation subset is prepared in
  `data/validation/` with guidelines in `docs/validation_guide.md`.
- Ekegusii column is an intentional placeholder for Week 3 few-shot transfer.

## Repository structure

```
├── SRC/
│   ├── config.py            # paths, domains, languages (en / sw / guz), scraping settings
│   ├── schema.py            # the dataset schema — single source of truth
│   ├── scraper.py           # robots.txt handling + polite (rate-limited) fetching
│   ├── cleaning.py          # dedup, language detection, fragment & relevance filtering
│   ├── build_dataset.py     # merge all sources → clean → assign IDs → CSV + stats
│   ├── report.py            # auto-generates the dataset stats report
│   ├── preprocessing.py     # tokenization, normalization, code-switch & glossary tagging
│   ├── eda.py               # dataset statistics + 6 EDA figures + auto EDA report
│   ├── splits.py            # leakage-safe stratified train/dev/test splits
│   ├── collectors/
│   │   ├── base.py          # config-driven site collector (one engine for all sites)
│   │   └── sites.py         # 21 per-source configs: URLs, link patterns, selectors
│   └── corpora/
│       ├── tico19.py        # downloads + parses the TICO-19 EN–SW translation memory
│       ├── tatoeba.py       # optional Tatoeba EN–SW pairs (manual download)
│       └── manual.py        # imports team-written PSAs from data/manual/
├── scripts/
│   ├── run_week1.py         # Week 1 pipeline: scrape + corpora → dataset
│   ├── scrape_more.py       # incremental: append new sites / team-written PSAs
│   ├── reclean.py           # re-apply cleaning to the existing CSV (no rescrape)
│   └── run_week2.py         # Week 2 pipeline: preprocess → EDA → splits → validation subset
├── tests/
│   ├── fixtures/            # small TMX + synthetic CSV fixtures for offline tests
│   └── test_smoke.py        # test suite (run before committing)
├── data/
│   ├── raw/                 # untouched raw exports
│   ├── processed/           # dataset CSV, build stats, preprocessed CSV, splits/
│   ├── external/            # downloaded corpora (TICO-19 TMX, Tatoeba files)
│   ├── manual/              # team-written PSA submissions (+ template)
│   ├── validation/          # 500-row native-speaker validation subset
│   └── glossary.json        # Kenyan cultural/institutional term glossary (24 terms)
├── docs/
│   ├── sources.md           # documented source registry (incl. blocked/dry sources)
│   ├── ETHICS.md            # scraping ethics: robots.txt, rate limits, licensing
│   ├── team_written_psa_kit.md  # how the team-written PSAs were produced
│   └── validation_guide.md  # native-speaker validation guidelines
├── reports/
│   ├── week1_report.md      # Week 1 report (data collection & curation)
│   ├── week2_report.md      # Week 2 report (preprocessing & EDA)
│   ├── week2_eda_report.md  # auto-generated EDA stats report
│   └── figures/             # 6 EDA figures (domain, length, pairing, sources)
└── requirements.txt
```

## Dataset schema

| Column | Description |
|--------|-------------|
| PSA_ID | Sequential ID assigned after cleaning (PSA000001…) |
| Domain | Health / Education / Agriculture / Security / Governance |
| English | English text (source language) |
| Kiswahili | Kiswahili translation where available (target language 1) |
| Ekegusii | Placeholder for Week 3 few-shot transfer (target language 2) |
| Source | Publishing organisation |
| Date | Publication or collection date (ISO) |
| URL | Page the text was collected from |
| Metadata | JSON provenance: type (scraped/corpus/manual), tool, license |
| Status | Validation status ("Pending" until native-speaker review) |

## Setup (Windows / VS Code)

```powershell
# 1. Create and activate a virtual environment
py -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser   # once, if blocked
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the installation
python tests/test_smoke.py        # expect: ALL SMOKE TESTS PASSED
```

In VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv` so every
new terminal activates automatically.

## Usage

```powershell
# --- Week 1: build the dataset ---

# Full run: scrape all configured sources + TICO-19, then clean and report
# (takes ~15–25 minutes because of the polite 2s delay between requests)
python scripts/run_week1.py

# Fast offline run: TICO-19 only, no scraping
python scripts/run_week1.py --no-scrape

# Scrape only specific sources (handy when tuning one site)
python scripts/run_week1.py --sites redcross_news,amref_kenya --max-pages 5

# Incrementally append new sources and/or team-written PSAs to the
# EXISTING dataset (does not rescrape everything):
python scripts/scrape_more.py --sites nacada_drug_prev,dci_units
python scripts/scrape_more.py                 # no --sites = ingest data/manual/ only

# Re-apply cleaning rules to the existing dataset WITHOUT re-scraping
python scripts/reclean.py

# --- Week 2: preprocess, EDA, splits, validation subset ---
python scripts/run_week2.py

# --- Any time: run the test suite ---
python tests/test_smoke.py
```

Outputs:

- `data/processed/psa_parallel_week1.csv` — the dataset
- `data/processed/build_stats.json` — build/cleaning statistics
- `data/processed/psa_preprocessed.csv` — tokenized/normalized/tagged dataset
- `data/processed/splits/` — train.csv, dev.csv, test.csv + split_stats.json
- `reports/week2_eda_report.md` + `reports/figures/` — auto EDA report + 6 figures
- `reports/week1_report_auto.md` — auto stats report (git-ignored, regenerable)

## Data strategy (three tiers)

1. **Automated scraping** from Kenyan government, NGO and UN sources
   (BeautifulSoup + requests, config-driven collectors; 21 sites configured).
2. **Open parallel corpora** for genuine EN–SW pairs — currently TICO-19
   (CC BY 4.0, 3,004 rows). Tatoeba (CC BY 2.0) is supported but optional:
   download the exports from https://tatoeba.org/en/downloads into
   `data/external/tatoeba/`.
3. **Team-written PSAs** for sub-topics with no scrapeable web presence —
   **done**: 150 original EN–SW pairs covering Education (school access,
   vocational, civic education, resources, safety), Security (public safety,
   crime prevention, national security, GBV, cybersecurity) and Governance
   (anti-corruption, public participation, elections, service delivery,
   devolution), 10 pairs per sub-topic. Process documented in
   `docs/team_written_psa_kit.md`; submissions live in `data/manual/`.

> **FLORES-200 is NOT used as training data.** It is an evaluation benchmark;
> training on it would inflate our Week 4 metrics. It is reserved for evaluation.

## Adding a new scraping source

Each source is a dict in `SRC/collectors/sites.py`:

```python
{
    "name": "example_site",
    "domain": "Health",                 # one of the 5 brief domains
    "source": "Example Organisation",
    "start_urls": ["https://example.go.ke/news"],   # listing page(s)
    "link_patterns": [r"example\.go\.ke/\d+/"],     # regex for article URLs; [] = start_urls are the content
    "content_selectors": ["article p", "main p", "p"],  # tried in order, "p" is fallback
    "max_pages": 20,
    "verify_ssl": True,                 # False only for broken .go.ke cert chains
    # optional: "pagination": {"template": "...?page={n}", "start": 1, "pages": 3},
    # optional: "min_words": 7, "max_records": 2500,
}
```

Then test with `python scripts/scrape_more.py --sites example_site` — the new
rows are appended to the existing dataset and re-cleaned automatically.

## Ethics and licensing

- robots.txt is checked before **every** request; genuine disallow rules are
  respected (e.g. UNICEF — zero rows collected). If robots.txt is unreachable,
  we allow with a logged warning — an unreachable file is not a disallow rule.
- Minimum 2-second delay (plus jitter) between requests to any one site.
- Every row records its source URL, collection date and license metadata.
- TICO-19: CC BY 4.0 — Translation Initiative for COVID-19 (TICO-19 consortium,
  including Translators without Borders). Attribution required and given.
- Tatoeba: CC BY 2.0 FR. Website content is collected for academic research only.
- Team-written PSAs are original work by the group (license: original work).
- No personal data is collected. See `docs/ETHICS.md` for details.

## Known limitations

- Domain imbalance remains: Health is 64% of the dataset (down from ~85%
  mid-pipeline). Education and Security were boosted with new scraping
  sources and team-written pairs, and remain the priority for further growth.
- Only 23.2% of rows are currently EN–SW parallel; English-only rows need
  translation (Week 3 back-translation) or exclusion from supervised training.
- Ekegusii is an empty placeholder until Week 3 few-shot transfer.
- Code-switching is rare in this corpus (1 row flagged in 13,519), so the
  Week 2 code-switch handling is built and tested but barely exercised.
- Some Kenyan government sites publish advisories only as PDFs/images (e.g.
  Ministry of Health) and are not scrapable as text; others (NTSA, National
  Police Service, Kilimo) yielded no usable text with the current selectors.
