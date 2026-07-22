# Machine Translation of Public Service Announcements (PSAs) — Group 10

**DSA4020A Natural Language Processing — Semester Project (Summer 2026)**

A proof-of-concept multilingual machine translation system for Public Service
Announcements in Kenya. The system translates between **English / Kiswahili** and
**Ekegusii**, an under-resourced indigenous language, making government information
more accessible. The final deliverable will be a deployable web app demonstrating
few-shot cross-lingual transfer learning on our curated PSA dataset.

## Team

| Member | Role (Week 1) |
|--------|---------------|
| Claire Mwarari | Scraping, site configuration & pipeline development |
| Levin Ekuam | Corpus curation (TICO-19, Tatoeba) & cleaning/validation |
| Paul | Scraping support, documentation & repo management |

## Project timeline

| Week | Focus | Status |
|------|-------|--------|
| 1 | Data collection & curation | ✅ Complete - 12,705 rows, 13 sources |
| 2 | Preprocessing & EDA | ⬜ |
| 3 | Modeling with transfer learning (mT5 / NLLB / mBART) | ⬜ |
| 4 | Evaluation, deployment & documentation | ⬜ |

## Repository structure

```
├── SRC/
│   ├── config.py            # paths, domains, languages (en / sw / guz), scraping settings
│   ├── schema.py            # the dataset schema — single source of truth
│   ├── scraper.py           # robots.txt handling + polite (rate-limited) fetching
│   ├── cleaning.py          # dedup, language detection, fragment & relevance filtering
│   ├── build_dataset.py     # merge all sources → clean → assign IDs → CSV + stats
│   ├── report.py            # auto-generates the Week 1 stats report
│   ├── collectors/
│   │   ├── base.py          # config-driven site collector (one engine for all sites)
│   │   └── sites.py         # per-source configs: URLs, link patterns, selectors
│   └── corpora/
│       ├── tico19.py        # downloads + parses the TICO-19 EN–SW translation memory
│       └── tatoeba.py       # optional Tatoeba EN–SW pairs (manual download)
├── scripts/
│   ├── run_week1.py         # main pipeline entry point
│   └── reclean.py           # re-apply cleaning to the existing CSV (no rescrape)
├── tests/
│   ├── fixtures/            # small TMX fixture for offline tests
│   └── test_smoke.py        # test suite (run before committing)
├── data/
│   ├── raw/                 # untouched raw exports
│   ├── processed/           # psa_parallel_week1.csv + build_stats.json
│   └── external/            # downloaded corpora (TICO-19 TMX, Tatoeba files)
├── docs/
│   ├── sources.md           # documented source registry
│   └── ETHICS.md            # scraping ethics: robots.txt, rate limits, licensing
├── reports/
│   └── week1_report.md      # Week 1 report
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
| Metadata | JSON provenance: type (scraped/corpus), tool, license |
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
# Full run: scrape all configured sources + TICO-19, then clean and report
# (takes ~15–25 minutes because of the polite 2s delay between requests)
python scripts/run_week1.py

# Fast offline run: TICO-19 only, no scraping
python scripts/run_week1.py --no-scrape

# Scrape only specific sources (handy when tuning one site)
python scripts/run_week1.py --sites redcross_news,amref_kenya --max-pages 5

# Re-apply cleaning rules to the existing dataset WITHOUT re-scraping
python scripts/reclean.py
```

Outputs:

- `data/processed/psa_parallel_week1.csv` — the dataset
- `data/processed/build_stats.json` — cleaning statistics
- `reports/week1_report.md` — auto-generated stats report

## Data strategy (three tiers)

1. **Automated scraping** from Kenyan government, NGO and UN sources
   (BeautifulSoup + requests, config-driven collectors).
2. **Open parallel corpora** for genuine EN–SW pairs — currently TICO-19
   (CC BY 4.0). Tatoeba (CC BY 2.0) is supported but optional: download the
   exports from https://tatoeba.org/en/downloads into `data/external/tatoeba/`.
3. **Team-written PSAs** for sub-topics with no scrapeable web presence
   (mental health, GBV, school safety, voter education, etc.).

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

Then run `python scripts/run_week1.py --sites example_site --max-pages 3` to test.

## Ethics and licensing

- robots.txt is checked before **every** request; genuine disallow rules are
  respected (e.g. UNICEF). If robots.txt is unreachable, we allow with a logged
  warning — an unreachable file is not a disallow rule.
- Minimum 2-second delay (plus jitter) between requests to any one site.
- Every row records its source URL, collection date and license metadata.
- TICO-19: CC BY 4.0 — Translation Initiative for COVID-19 (TICO-19 consortium,
  including Translators without Borders). Attribution required and given.
- Tatoeba: CC BY 2.0 FR. Website content is collected for academic research only.
- No personal data is collected. See `docs/ETHICS.md` for details.

## Known limitations (Week 1)

- Domain imbalance: Health dominates (~68%); Education and Security are thin
  (addressed with team-written PSAs).
- Kiswahili is currently only present for TICO-19 rows; most scraped rows are
  English-only pending translation/parallel-source work in later weeks.
- Ekegusii is an empty placeholder until Week 3 few-shot transfer.
- Some Kenyan government sites publish advisories only as PDFs/images (e.g.
  Ministry of Health) and are not scrapable as text yet.
