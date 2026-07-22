# Week 1 Report — Machine Translation of Public Service Announcements (Kenya)

**Course:** DSA4020A Natural Language Processing
**Group:** 10
**Week:** 1 — Data Collection & Curation
**Date:** 23 July 2026

---

## 1. Overview

This week we built the data foundation for the project: a parallel dataset of Public
Service Announcements (PSAs) in English and Kiswahili, with a placeholder column for
Ekegusii, our second target language. By the end of the week we had **13,033 cleaned
rows** from **13 distinct sources**, collected through a mix of automated web scraping
and existing parallel corpora, which is well above the 5,000-sentence target.

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

We documented 17 source configurations in `SRC/collectors/sites.py`, of which 13
produced data this week. The rest were evaluated and dropped or blocked for the
reasons explained in the Challenges section.

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

We also evaluated the Ministry of Health website (health.go.ke) and dropped it as a
scraping source — see Challenges.

## 4. Dataset summary

**File:** `data/processed/psa_parallel_week1.csv`
**Total rows after cleaning:** 13,033

### Rows per domain

| Domain | Rows | Share |
|--------|------|-------|
| Health | 8,904 | 68.3% |
| Agriculture | 2,418 | 18.5% |
| Governance | 1,395 | 10.7% |
| Security | 285 | 2.2% |
| Education | 31 | 0.2% |

### Schema

| Column | Content |
|--------|---------|
| PSA_ID | Sequential ID assigned after cleaning (PSA000001…) |
| Domain | One of the five brief domains |
| English | English text |
| Kiswahili | Kiswahili text (filled for TICO-19 pairs; empty for most scraped rows) |
| Ekegusii | Empty placeholder — few-shot transfer target in Week 3 |
| Source | Publishing organisation |
| Date | Publication or collection date |
| URL | Page the text was collected from |
| Metadata | JSON with provenance (type: scraped/corpus, tool, license) |
| Status | Validation status ("Pending" until native-speaker review in Week 2) |

### Cleaning summary (from `build_stats.json`)

| Step | Rows remaining |
|------|----------------|
| Raw collected | 13,283 |
| After empty-field filter | 13,283 |
| After length filter (4–80 words) | 13,268 |
| After deduplication (exact + near-duplicates) | 13,116 |
| After language detection (langdetect on EN and SW columns) | 13,033 |

### Sample entries

| PSA_ID | Domain | English | Source |
|--------|--------|---------|--------|
| 2411 | Health | Ultrasound can be used to screen for Down syndrome and major structural abnormalities during the first trimester, and for severe fetal anomalies during the second trimester. | WHO |
| 8212 | Agriculture | Avoid growing brassica crops in the same field for a period of at least three seasons. | Infonet-Biovision |
| 6842 | Governance | The levy shall be collected as follows: - | KRA *(this type of fragment is exactly what our later cleaning pass targets — see challenges)* |
| 10553 | Health | Pneumonia is one of the most severe symptoms and can progress rapidly to acute respiratory distress syndrome. | TICO-19 (paired with Kiswahili) |
| 9622 | Governance | • The Leadership and Integrity Act, 2012. | EACC *(bullet marker stripped in the later cleaning pass)* |

## 5. How we collected the data

We used a three-tier approach, because it became clear early on that no realistic
amount of scraping alone would produce 5,000 *bilingual* PSA sentences in one week:

1. **Automated scraping** (BeautifulSoup + requests). We built a config-driven
   collector: each source has listing URLs, a regex for article links, CSS selectors
   for the body text, and optional pagination. Every request goes through a robots.txt
   check and a 2-second-plus-jitter delay. Each source fails independently so one bad
   site cannot kill the whole run.
2. **Existing parallel corpora.** TICO-19 (COVID-19 crisis communication, human
   translated, CC BY 4.0) gave us 3,033 genuine English–Kiswahili pairs, which are
   currently our only large source of Kiswahili text. We deliberately did **not** use
   FLORES-200 as training data — it is an evaluation benchmark, and training on it
   would make our Week 4 metrics meaningless. We are keeping it aside for evaluation.
3. **Team-written PSAs (planned).** To cover the sub-topics that barely exist as
   scrapeable English text (mental health, GBV, school safety, voter education), we
   plan to write a few hundred PSAs ourselves in announcement style. This is our main
   plan for fixing the Education and Security imbalance.

For Ekegusii, we could not find any usable parallel corpora this week. The column
exists in the schema as a placeholder, and the plan is few-shot cross-lingual transfer
(English/Kiswahili → Ekegusii) in Week 3, plus native-speaker help for a seed set.

## 6. Challenges faced

This was honestly where most of the week's work went.

**a) Antivirus TLS inspection broke SSL on most .go.ke sites.**
Requests to NTSA, Kilimo, education.go.ke, meteo.go.ke and others failed with
`CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`. The sites
worked fine in a browser and from other networks — our local antivirus was
intercepting HTTPS and re-signing certificates, which Python (correctly) refused to
trust. We added a per-site `verify_ssl: False` option for the affected public,
read-only pages instead of disabling verification globally.

**b) A false robots.txt "block" caused by the same SSL issue.**
For several sites the robots.txt file itself could not be fetched because of the SSL
problem, and Python's `robotparser` treats "couldn't read the file" as "disallow
everything" — so our own pipeline blocked sites like Kenya Met and the Ministry of
Education even though the log said "allowing by default" one line earlier. WHO was
being blocked the same way. We rewrote the robots handling to fetch robots.txt through
our own request code (honouring each site's SSL setting) and to allow when the file is
unreachable, since an unreachable file is not a disallow rule. After the fix, WHO
turned out to explicitly allow our paths and became our biggest scraped source
(5,122 rows). UNICEF, on the other hand, genuinely disallows the path we wanted, and
the pipeline respected that — zero records.

**c) Government content published as PDFs and images, not text.**
The Ministry of Health site was our most wanted source, but its press statements and
disease alert pages contain almost no HTML text — the advisories are posted as PDFs
and poster images. We dropped MoH as a scraping source and replaced it with Amref
Health Africa. Extracting text from those PDFs is something we may revisit in Week 2.

**d) Moved or dead URLs.**
Several documented URLs from our initial source list no longer exist:
redcross.or.ke/news (the real listing is /category/news), health.go.ke/press-release
(real: /press-statements), NTSA /news, kilimo.go.ke/category/news and
nationalpolice.go.ke/news (all 404). KALRO and FAO Kenya render their news with
JavaScript, so plain HTTP scraping gets nothing usable. We fixed what we could and
documented the rest.

**e) Anti-bot blocking.**
eCitizen returns 403 to non-browser clients and KRA's Swahili site section
(`/sw/news-center/public-notices`) returned a 402 error. On the positive side, that
error confirmed KRA runs a Swahili version of its site — potential parallel
English–Kiswahili government content we want to explore in Week 2.

**f) Content-style mismatch (our biggest data-quality problem).**
Infonet-Biovision initially produced 22,357 rows — more than everything else
combined — but most of it was photo captions and encyclopedia-style pest descriptions
("Real size: 6 to 15 mm long. Ⓒ A.M."), not PSA-style text. Two things made it worse:
the copyright sign on that site is Ⓒ (a circled letter), which our boilerplate filter
didn't recognise, and even after adding it, Python's `.lower()` converts Ⓒ to a
*different* character (ⓒ), so the filter still missed it. We fixed the filter, set a
7-word minimum for that source and capped it at 2,500 records to keep the dataset
balanced.

**g) Sentence-splitting artifacts.**
Splitting scraped lists into sentences left bullet markers on some rows
("• The Leadership and Integrity Act, 2012.") and dangling fragments where a list
introduction lost its list ("The levy shall be collected as follows: -"). We added
cleaning rules to strip leading bullets and drop colon-dangling fragments, and made
the cleaning step re-runnable without re-scraping (`scripts/reclean.py`).

**h) Language detection is non-deterministic by default.**
`langdetect` gives different results on different runs unless seeded. We pinned the
seed so our cleaning results are reproducible.

**i) Domain imbalance.**
Despite 17 sources, Education (~31 rows) and Security (~285 rows) are far behind
Health (~8,900). These sub-topics mostly don't exist as scrapeable English web text in
Kenya, which is why we plan team-written PSAs for them.

## 7. Team roles and ethical scraping review

We held our Week 1 meeting and agreed on:

- **Roles:** since we are a group of three, we split the work as: scraping, site
  configuration & pipeline development (1 member), corpus curation & cleaning/
  validation (1 member), scraping support, documentation & repo management
  (1 member). *(Names to be filled in the final report.)*
- **Ethics:** we check robots.txt before every request and respect genuine disallows
  (UNICEF being the proof); we wait 2+ seconds between requests to any one site; every
  row carries its source URL and license metadata; TICO-19 is CC BY 4.0 and Tatoeba
  (if we add it) is CC BY 2.0 — both will be attributed; we collect no personal data;
  everything is for academic use.

## 8. Next steps (Week 2)

- Preprocessing pipeline: tokenization, normalization, code-switching handling,
  glossary for cultural terms
- Full EDA: domain distribution, text length histograms, vocabulary sizes,
  language-pair statistics
- Native-speaker validation subset (~500 sentences)
- Fix domain imbalance with team-written PSAs (Education and Security)
- Investigate KRA's Swahili site section and Infonet's per-article Swahili versions
  as sources of genuine parallel text
- Train/dev/test splits (reserving FLORES-200 and/or TICO-19 test split strictly
  for evaluation)
