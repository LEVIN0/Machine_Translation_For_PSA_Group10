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
| 3 | Modeling with transfer learning (mT5-small / NLLB-200) | ✅ Complete |
| 4 | Evaluation, deployment & documentation | ⬜ Next |

## Dataset at a glance

**6,823 rows** across the 5 brief domains, **audited against the lecturer's
PSA Framework** (`reports/framework_audit.md`). Composition: lecturer gold
dataset 2,852 rows (with Ekegusii + Dholuo + Somali translations) · TICO-19
corpus 3,004 · audited scraped PSAs 817 · team-written 150.

| Domain | Rows | Share |
|--------|-----:|------:|
| Health | 3,944 | 57.8% |
| Security | 952 | 14.0% |
| Education | 828 | 12.1% |
| Agriculture | 559 | 8.2% |
| Governance | 540 | 7.9% |

- **5,972 rows (87.5%) are EN–SW parallel**; **2,848 rows (41.7%) have
  Ekegusii** — our transfer target now has real PSA training pairs.
- Every row carries an auditable `psa_class` in its Metadata (PSA /
  PressRelease / Legal / Informational); 9,548 scraped rows that failed the
  framework audit were deleted (see `reports/framework_audit.md`).
- Train/dev/test splits (90/5/5, stratified by domain, seed 42, zero text
  leakage): **6,141 / 341 / 341** rows in `data/processed/splits/`.
- A 500-row native-speaker validation subset is prepared in
  `data/validation/` with guidelines in `docs/validation_guide.md`.
- The Ekegusii benchmark (`data/external/guz_benchmark/guz_test.tsv`, 138
  eng–guz pairs from the held-out test split) is reserved purely for
  evaluation. FLORES-200 was evaluated and dropped: it contains no Ekegusii.

## Repository structure

```
├── src/
│   ├── config.py            # paths, domains, languages (en / sw / guz), scraping settings
│   ├── schema.py            # the dataset schema — single source of truth
│   ├── scraper.py           # robots.txt handling + polite (rate-limited) fetching
│   ├── cleaning.py          # text normalization, dedup, language detection, filtering
│   ├── psa_classify.py      # lecturer-calibrated PSA framework classifier
│   ├── audit.py             # framework audit: classify, stamp, drop scraped non-PSA
│   ├── build_dataset.py     # collect → audit → merge gold → clean → CSV + stats
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
│       ├── lecturer.py      # imports the lecturer gold dataset (PSA_KE_Final.csv)
│       └── manual.py        # imports team-written PSAs from data/manual/
├── training/
│   ├── config.py            # model zoo (mT5-small, NLLB-200-600M) + TrainConfig
│   ├── data.py              # splits → paired HF datasets (benchmark never trains)
│   ├── augment.py           # back-translation of English-only rows
│   ├── trainer.py           # Seq2SeqTrainer wiring: W&B, freezing, best-checkpoint
│   ├── evaluate.py          # sacreBLEU + chrF on PSA dev/test + guz benchmark
│   ├── inference.py         # MTTranslator (EN/SW/guz) + demo PSAs
│   └── ablate.py            # ablation matrix + auto results table
├── scripts/
│   ├── run_week1.py         # Week 1 pipeline: scrape + corpora → audit → gold → dataset
│   ├── run_week2.py         # Week 2 pipeline: preprocess → EDA → splits → validation subset
│   ├── build_guz_benchmark.py # build held-out Ekegusii benchmark from the test split
│   ├── run_training.py      # Week 3: one training run with a named config
│   ├── run_ablations.py     # Week 3: run the ablation matrix
│   ├── run_eval.py          # Week 3: evaluate checkpoint(s)
│   └── translate.py         # Week 3: translation demo CLI (success criterion)
├── notebooks/
│   └── week3_colab.ipynb    # one-click Colab runbook (GPU check → train → demo)
├── tests/
│   ├── fixtures/            # small TMX + synthetic CSV + benchmark TSV fixtures
│   ├── test_smoke.py        # test suite (run before committing)
│   └── test_week3_*.py      # Week 3 modules (auto-discovered by test_smoke.py)
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
│   ├── validation_guide.md  # native-speaker validation guidelines
│   └── week3_colab_guide.md # Colab training guide + GPU troubleshooting
├── reports/
│   ├── week1_report.md      # Week 1 report (data collection & curation)
│   ├── week2_report.md      # Week 2 report (preprocessing & EDA)
│   ├── week2_eda_report.md  # auto-generated EDA stats report
│   ├── framework_audit.md   # PSA framework audit: per-source kept/dropped, method
│   ├── week3_report.md      # Week 3 report (modeling with transfer learning)
│   ├── week3_results.md     # auto-generated ablation results table
│   └── figures/             # 6 EDA figures (domain, length, pairing, sources)
├── requirements.txt         # Weeks 1–2 dependencies
└── requirements-training.txt  # Week 3 ML stack (torch, transformers, wandb…)
```

## Dataset schema

| Column | Description |
|--------|-------------|
| PSA_ID | Sequential ID assigned after cleaning (PSA000001…) |
| Domain | Health / Education / Agriculture / Security / Governance |
| English | English text (source language) |
| Kiswahili | Kiswahili translation where available (target language 1) |
| Ekegusii | Ekegusii translation where available (target language 2; 2,848 rows from the lecturer gold dataset) |
| Source | Publishing organisation |
| Date | Publication or collection date (ISO) |
| URL | Page the text was collected from |
| Metadata | JSON provenance: type (scraped/corpus/manual/gold), tool, license, psa_class |
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

# Full run: scrape all configured sources + TICO-19 + team-written PSAs,
# audit every row against the PSA framework, merge the lecturer gold
# dataset, then clean and report
# (takes ~15–25 minutes because of the polite 2s delay between requests)
python scripts/run_week1.py

# Fast offline run: corpora + gold only, no scraping
python scripts/run_week1.py --no-scrape

# Scrape only specific sources (handy when tuning one site)
python scripts/run_week1.py --sites redcross_news,amref_kenya --max-pages 5

# --- Week 2: preprocess, EDA, splits, validation subset ---
python scripts/run_week2.py

# --- Week 3: training (GPU needed — use notebooks/week3_colab.ipynb on Colab) ---
python scripts/build_guz_benchmark.py           # one-time Ekegusii benchmark build
python scripts/run_training.py --model nllb_600m --run-name ft_nllb_base
python scripts/run_ablations.py --matrix quick  # or: standard
python scripts/run_eval.py --checkpoint runs/ft_nllb_base/checkpoint-best
python scripts/translate.py --demo              # translation demo (8 sample PSAs)

# --- Any time: run the test suite ---
python tests/test_smoke.py
```

For Week 3 training we use **Google Colab (free GPU)** — open
`notebooks/week3_colab.ipynb` and run top to bottom; the guide with expected
runtimes and GPU troubleshooting is `docs/week3_colab_guide.md`. Experiments
are tracked in **Weights & Biases** (project `psa-mt-group10`) with JSON-log
fallback, so no run is ever lost. See `reports/week3_report.md` for
hyperparameters, ablations and preliminary results.

Outputs:

- `data/processed/psa_parallel_week1.csv` — the dataset
- `data/processed/build_stats.json` — build/cleaning statistics
- `data/processed/psa_preprocessed.csv` — tokenized/normalized/tagged dataset
- `data/processed/splits/` — train.csv, dev.csv, test.csv + split_stats.json
- `reports/week2_eda_report.md` + `reports/figures/` — auto EDA report + 6 figures
- `reports/week1_report_auto.md` — auto stats report (git-ignored, regenerable)

## Data strategy (four tiers)

0. **Lecturer gold dataset** (`PSA_KE_Final.csv`) — 2,852 framework-validated
   PSAs with EN/SW/**Ekegusii**/Dholuo/Somali text; merged verbatim, marked
   `type:"gold"` in Metadata, `Status="Validated"`.
1. **Automated scraping** from Kenyan government, NGO and UN sources
   (BeautifulSoup + requests, config-driven collectors; 21 sites configured),
   **filtered by the framework audit** to high-confidence PSA rows.
2. **Open parallel corpora** for genuine EN–SW pairs — currently TICO-19
   (CC BY 4.0, 3,004 rows, kept whole as corpus data). Tatoeba (CC BY 2.0) is
   supported but optional: download the exports from
   https://tatoeba.org/en/downloads into `data/external/tatoeba/`.
3. **Team-written PSAs** for sub-topics with no scrapeable web presence —
   **done**: 150 original EN–SW pairs covering Education (school access,
   vocational, civic education, resources, safety), Security (public safety,
   crime prevention, national security, GBV, cybersecurity) and Governance
   (anti-corruption, public participation, elections, service delivery,
   devolution), 10 pairs per sub-topic. Process documented in
   `docs/team_written_psa_kit.md`; submissions live in `data/manual/`.

> **Framework audit.** Applying the lecturer's PSA Framework, every
> collected row is classified (`src/psa_classify.py`, lecturer-calibrated
> rules) as part of the Week 1 build, and scraped rows that fail the audit
> are deleted — 9,548 in the canonical build. Corpus, team-written and
> lecturer gold rows are exempt and kept whole. Full methodology and
> per-source kept/dropped counts: `reports/framework_audit.md`.

> **The Ekegusii benchmark is NOT training data.** `guz_test.tsv` is built
> from the held-out test split and is evaluation-only; training on it would
> inflate our Week 4 metrics. (FLORES-200 was considered for this role and
> dropped — the archive has 204 languages, none of them Ekegusii, and
> NLLB-200's tokenizer has no `guz_Latn` token. See `docs/SPEC_WEEK3.md` §8.)

## Adding a new scraping source

Each source is a dict in `src/collectors/sites.py`:

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

Then test with `python scripts/run_week1.py --sites example_site --max-pages 5`
— new rows go through the same audit and cleaning as every other source.

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

- Health is still the largest domain (57.8%), though far better balanced than
  at Week 1 (was 85% mid-pipeline); Week 3 uses a domain-balanced training view.
- 851 rows (12.5%) remain English-only (mostly TICO-19); Week 3
  back-translation will create synthetic pairs from vetted rows.
- The PSA framework classifier is a documented heuristic — per-row decisions
  are auditable via `psa_class` in Metadata, but borderline cases exist.
- Some Kenyan government sites publish advisories only as PDFs/images (e.g.
  Ministry of Health) and are not scrapable as text; others (NTSA, National
  Police Service, Kilimo) yielded no usable text with the current selectors.
