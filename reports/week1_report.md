# Week 1 Report — Machine Translation of Public Service Announcements (Kenya)

**Course:** DSA4020A Natural Language Processing
**Group:** 10
**Week:** 1 — Data Collection & Curation
**Date:** 23–29 July 2026

---

## 1. Overview

This week we built the data foundation for the project: a parallel dataset of
Public Service Announcements (PSAs) in English and Kiswahili, with Ekegusii as
our second target language. The final Week 1 dataset is **6,823 rows across the
five brief domains**, collected through automated web scraping, an existing
parallel corpus, the lecturer's gold dataset and team-written PSAs — with every
row classified against the lecturer's PSA Framework as part of the build.

All code and data are on our GitHub repo:
`https://github.com/LEVIN0/Machine_Translation_For_PSA_Group10`

## 2. What we set out to do

From the project brief, Week 1 required:

- Identify and document at least 10 reliable sources
- Implement a hybrid scraping pipeline (manual + automated), respecting robots.txt
  and rate limits
- Collect PSAs across the five domains (Health, Education, Agriculture, Security,
  Governance)
- Create a structured dataset with the agreed columns
- Initial cleaning: deduplication, language detection, relevance filtering
- Reach at least 5,000 parallel sentences (or equivalent per pair)
- Submit this report with summary stats, sample entries and challenges faced
- Hold a team meeting to assign roles and review ethical scraping practices

## 3. Sources documented

We documented 17 scraping configurations in `src/collectors/sites.py`, of which
13 produced data. The rest were evaluated and dropped or blocked for the reasons
explained in the Challenges section.

| # | Source | Domain | Result |
|---|--------|--------|--------|
| 1 | WHO COVID-19 Q&A | Health | 81 rows |
| 2 | WHO Fact Sheets | Health | 5,041 rows (with fact sheets under one listing page) |
| 3 | UNICEF Parenting | Health | 0 — blocked by robots.txt (respected) |
| 4 | Kenya Red Cross | Health | 409 rows |
| 5 | Amref Health Africa (Kenya) | Health | 421 rows |
| 6 | TICO-19 corpus | Health | 3,033 EN–SW pairs |
| 7 | NTSA | Security | 0 — page moved (404) |
| 8 | Kenya Meteorological Department | Security | 28 rows |
| 9 | Ministry of Agriculture (kilimo.go.ke) | Agriculture | 0 — page moved (404) |
| 10 | Ministry of Education | Education | 15 rows |
| 11 | Kenya Revenue Authority | Governance | 1,017 rows (4 listing pages) |
| 12 | eCitizen Kenya | Governance | 2 rows |
| 13 | National Police Service | Security | 0 — page moved (404) |
| 14 | Infonet-Biovision | Agriculture | 2,418 rows (capped deliberately) |
| 15 | NEMA | Security | 248 rows |
| 16 | Communications Authority of Kenya | Security | 9 rows |
| 17 | EACC | Governance | 376 rows |
| + | KICD (added late) | Education | 16 rows |

Three further sources complete the dataset (see §5): the lecturer's gold
dataset (`PSA_KE_Final.csv`, 2,852 rows with Ekegusii translations), the
TICO-19 parallel corpus, and 150 team-written PSAs for sub-topics with no
scrapeable web presence.

We also evaluated the Ministry of Health website (health.go.ke) and dropped it
as a scraping source — see Challenges.

## 4. The PSA Framework and data quality

The lecturer's **PSA Framework** (a 4-step classification guide: core intent →
style/tone → audience → channel) defines what counts as a PSA for this
project. We apply it inside the build pipeline as a **framework audit** step
(`src/audit.py` + the frozen, lecturer-calibrated classifier in
`src/psa_classify.py`): every collected row is scored, each row's `psa_class`
(PSA / PressRelease / Legal / Informational) is stamped into its Metadata for
auditability, and scraped rows that fail the audit are deleted from the
dataset. TICO-19 (corpus), team-written (manual) and lecturer gold rows are
exempt and kept whole.

A key calibration insight: the lecturer's gold file itself applies a *broader*
standard than the framework text suggests — service announcements like
"[KUCCPS Portal] KUCCPS portal will open in March 2025…" are labelled PSA — so
we calibrated the classifier to his applied standard: directive/advisory
language, imperatives, audience markers and service/deadline announcements
keep a row; press-release, legal-notice, encyclopedic and
orphaned-continuation signals drop it.

Applying the framework strictly is what turns a raw web harvest into a PSA
dataset: WHO fact-sheet definitions ("COVID-19 is the disease caused by…"),
Infonet encyclopedic farming descriptions and press-release narrative about
officials and events do not qualify, and sentence-splitting during scraping
had stripped action context from many advisory paragraphs. In the canonical
build the audit kept **817 of 10,365 scraped rows** (9,548 deleted); full
methodology and per-source kept/dropped counts are in
`reports/framework_audit.md`.

## 5. How we collected the data (four tiers)

No realistic amount of scraping alone would produce 5,000 *bilingual* PSA
sentences in one week, so the dataset combines four tiers:

0. **Lecturer gold dataset** (`data/external/PSA_KE_Final.csv`) — 2,852
   framework-validated PSAs (2,895 imported, less internal duplicates and
   cleaning losses) with EN/SW/**Ekegusii**/Dholuo/Somali text; merged
   verbatim, marked `type:"gold"` in Metadata, `Status="Validated"`. Text is
   normalized on import (encoding repair, whitespace); nothing is reworded.
1. **Automated scraping** (BeautifulSoup + requests). A config-driven
   collector: each source has listing URLs, a regex for article links, CSS
   selectors for the body text, and optional pagination. Every request goes
   through a robots.txt check and a 2-second-plus-jitter delay. Each source
   fails independently so one bad site cannot kill the whole run. Scraped
   rows are then filtered to high-confidence PSAs by the framework audit (§4).
2. **Existing parallel corpora.** TICO-19 (COVID-19 crisis communication,
   human translated, CC BY 4.0) contributes 3,004 genuine English–Kiswahili
   pairs, kept whole as corpus data (flagged in Metadata).
3. **Team-written PSAs.** 150 original EN–SW pairs covering the sub-topics
   that barely exist as scrapeable English text (school access, public
   safety, anti-corruption, voter education…), 10 pairs per sub-topic;
   process documented in `docs/team_written_psa_kit.md`.

## 6. Dataset summary

**File:** `data/processed/psa_parallel_week1.csv`
**Total rows:** 6,823

### Pipeline summary (from `data/processed/build_stats.json`)

| Stage | Rows |
|------|-----:|
| Collected (scraped 10,365 + TICO-19 3,004 + team-written 150) | 13,519 |
| After framework audit (9,548 scraped non-PSA rows deleted) | 3,971 |
| After lecturer gold merge (+2,895) and cleaning/dedupe | **6,823** |

### Rows per domain

| Domain | Rows | Share |
|--------|------|-------|
| Health | 3,944 | 57.8% |
| Security | 952 | 14.0% |
| Education | 828 | 12.1% |
| Agriculture | 559 | 8.2% |
| Governance | 540 | 7.9% |

### Composition and pairing

- Composition: lecturer gold 2,852 · TICO-19 corpus 3,004 · audited scraped
  PSA 817 · team-written 150.
- **5,972 rows (87.5%) are EN–SW parallel**; 851 rows (12.5%) are
  English-only (mostly TICO-19).
- **2,848 rows (41.7%) have Ekegusii** (from the gold dataset), which also
  fixed the initial domain imbalance — the gold data is Security/Education-heavy.

### Schema

| Column | Content |
|--------|---------|
| PSA_ID | Sequential ID assigned after cleaning (PSA000001…) |
| Domain | One of the five brief domains |
| English | English text |
| Kiswahili | Kiswahili text where available (target language 1) |
| Ekegusii | Ekegusii text where available (target language 2; 2,848 rows) |
| Source | Publishing organisation |
| Date | Publication or collection date |
| URL | Page the text was collected from |
| Metadata | JSON provenance: type (scraped/corpus/manual/gold), tool, license, psa_class |
| Status | Validation status ("Pending" until native-speaker review in Week 2) |

### Sample entries

| PSA_ID | Domain | Type | English (truncated) |
|--------|--------|------|---------------------|
| PSA000001 | Health | scraped | If possible, call your health care provider, hotline or health facility first, so you can be directed to the r… |
| PSA000771 | Health | corpus | about how long have these symptoms been going on? |
| PSA003822 | Education | manual | The Ministry of Education reminds parents that enrolment for Grade One is open at all public primary scho… |
| PSA004731 | Health | gold | We are advising all health workers to ensure they are wearing masks and PPEs at all times. Please take al… |

## 7. Challenges faced

This was honestly where most of the week's work went.

**a) Antivirus TLS inspection broke SSL on most .go.ke sites.**
Requests to NTSA, Kilimo, education.go.ke, meteo.go.ke and others failed with
`CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`. The
sites worked fine in a browser and from other networks — our local antivirus
was intercepting HTTPS and re-signing certificates, which Python (correctly)
refused to trust. We added a per-site `verify_ssl: False` option for the
affected public, read-only pages instead of disabling verification globally.

**b) A false robots.txt "block" caused by the same SSL issue.**
For several sites the robots.txt file itself could not be fetched because of
the SSL problem, and Python's `robotparser` treats "couldn't read the file" as
"disallow everything" — so our own pipeline blocked sites like Kenya Met and
the Ministry of Education even though the log said "allowing by default" one
line earlier. WHO was being blocked the same way. We rewrote the robots
handling to fetch robots.txt through our own request code (honouring each
site's SSL setting) and to allow when the file is unreachable, since an
unreachable file is not a disallow rule. After the fix, WHO turned out to
explicitly allow our paths and became our biggest scraped source. UNICEF, on
the other hand, genuinely disallows the path we wanted, and the pipeline
respected that — zero records.

**c) Government content published as PDFs and images, not text.**
The Ministry of Health site was our most wanted source, but its press
statements and disease alert pages contain almost no HTML text — the
advisories are posted as PDFs and poster images. We dropped MoH as a scraping
source and replaced it with Amref Health Africa. Extracting text from those
PDFs is something we may revisit later.

**d) Moved or dead URLs.**
Several documented URLs from our initial source list no longer exist:
redcross.or.ke/news (the real listing is /category/news),
health.go.ke/press-release (real: /press-statements), NTSA /news,
kilimo.go.ke/category/news and nationalpolice.go.ke/news (all 404). KALRO and
FAO Kenya render their news with JavaScript, so plain HTTP scraping gets
nothing usable. We fixed what we could and documented the rest.

**e) Anti-bot blocking.**
eCitizen returns 403 to non-browser clients and KRA's Swahili site section
(`/sw/news-center/public-notices`) returned a 402 error. On the positive side,
that error confirmed KRA runs a Swahili version of its site — potential
parallel English–Kiswahili government content.

**f) Content-style mismatch (our biggest scraping data-quality problem).**
Infonet-Biovision initially produced 22,357 rows — more than everything else
combined — but most of it was photo captions and encyclopedia-style pest
descriptions ("Real size: 6 to 15 mm long. Ⓒ A.M."), not PSA-style text. Two
things made it worse: the copyright sign on that site is Ⓒ (a circled letter),
which our boilerplate filter didn't recognise, and even after adding it,
Python's `.lower()` converts Ⓒ to a *different* character (ⓒ), so the filter
still missed it. We fixed the filter, set a 7-word minimum for that source and
capped it at 2,500 records to keep the dataset balanced. (The framework audit
in §4 is the systematic version of this lesson: most of what the web calls
"content" is not a PSA.)

**g) Sentence-splitting artifacts.**
Splitting scraped lists into sentences left bullet markers on some rows
("• The Leadership and Integrity Act, 2012.") and dangling fragments where a
list introduction lost its list ("The levy shall be collected as follows:
-"). We added cleaning rules to strip leading bullets and drop
colon-dangling fragments.

**h) Language detection is non-deterministic by default.**
`langdetect` gives different results on different runs unless seeded. We
pinned the seed so our cleaning results are reproducible.

**i) Domain imbalance.**
Early in the week Education (28 rows) and Security (277 rows) were far behind
Health (~8,700 scraped rows). These sub-topics mostly don't exist as
scrapeable English web text in Kenya — the lecturer gold dataset (Security/
Education-heavy) and our 150 team-written pairs brought them to 952 and 828
rows in the final dataset.

## 8. Team roles and ethical scraping review

We held our Week 1 meeting and agreed on:

- **Roles:** scraping, site configuration & pipeline development (Claire
  Mwarari); corpus curation & cleaning/validation (Levin Ekuam); data
  collection support, documentation & repo management (Paul).
- **Ethics:** we check robots.txt before every request and respect genuine
  disallows (UNICEF being the proof); we wait 2+ seconds between requests to
  any one site; every row carries its source URL and license metadata;
  TICO-19 is CC BY 4.0 and is attributed; team-written PSAs are original
  work; we collect no personal data; everything is for academic use.

## 9. Next steps (Week 2)

- Preprocessing pipeline: tokenization, normalization, code-switching handling,
  glossary for cultural terms
- Full EDA: domain distribution, text length histograms, vocabulary sizes,
  language-pair statistics
- Native-speaker validation subset (~500 sentences)
- Train/dev/test splits (reserving the held-out test split strictly for
  evaluation)
