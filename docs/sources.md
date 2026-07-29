# Data Sources — full registry

The brief requires ≥10 documented sources. After the framework audit and the
lecturer gold merge, the dataset has four tiers: the **lecturer gold dataset**,
**21 configured scraping sources**, the **TICO-19 parallel corpus**, and
**team-written PSAs**. Row counts below are the CURRENT dataset (6,823 rows,
post-audit — see `reports/framework_audit.md` for pre-audit yields and
per-source kept/dropped figures).

## Gold dataset (lecturer-provided)

| # | Source | Class | Rows | Notes |
|---|--------|-------|-----:|-------|
| 0 | Lecturer dataset (`data/external/PSA_KE_Final.csv`) | PSA (pre-classified) | 2,852 | 2,903 raw rows minus 8 internal duplicates and cleaning losses; EN/SW/**Ekegusii**/Dholuo/Somali; merged verbatim (`type:"gold"`, `Status="Validated"`); Dholuo/Somali preserved in Metadata. 195 rows (6.8%) carry mojibake from the source file (kept verbatim, flagged to lecturer). |

## Web sources (scraped, then framework-audited)

21 sources are configured in `SRC/collectors/sites.py`. The framework audit
(`SRC/psa_classify.py`) kept only high-confidence PSA rows; deleted counts are
in `reports/framework_audit.md`.

| # | Config name | Source | Domain | URL | Status | Rows |
|---|-------------|--------|--------|-----|--------|-----:|
| 1 | who_covid_qna + who_fact_sheets | WHO | Health | https://www.who.int/news-room/fact-sheets | ✅ Producing (post-audit PSA rows) | 401 |
| 2 | infonet_agri | Infonet-Biovision (Biovision Africa Trust) | Agriculture | https://infonet-biovision.org/crops-fruits-vegetables | ✅ Producing | 210 |
| 3 | kra_notices | Kenya Revenue Authority | Governance | https://www.kra.go.ke/news-center/public-notices | ✅ Producing | 78 |
| 4 | amref_kenya | Amref Health Africa | Health | https://amref.org/kenya/news/ | ✅ Producing | 29 |
| 5 | redcross_news | Kenya Red Cross | Health | https://www.redcross.or.ke/category/news | ✅ Producing | 22 |
| 6 | nema_insights | National Environment Management Authority | Security | https://nema.go.ke/category/insights/article/ | ✅ Producing | 15 |
| 7 | dci_units | Directorate of Criminal Investigations | Security | https://www.dci.go.ke/anti-narcotics | ✅ Producing | 15 |
| 8 | eacc_news | Ethics and Anti-Corruption Commission | Governance | https://eacc.go.ke/default/news/ | ✅ Producing | 13 |
| 9 | kuccps_news | KUCCPS | Education | https://www.kuccps.net/ | ✅ Producing | 12 |
| 10 | education_ke | Ministry of Education | Education | https://www.education.go.ke | ✅ Producing | 7 |
| 11 | nacada_drug_prev | NACADA | Security | https://www.nacada.go.ke | ✅ Producing | 7 |
| 12 | covaw_news | COVAW (Coalition on Violence Against Women) | Security | https://covaw.or.ke/category/news-press-releases/ | ✅ Producing | 6 |
| 13 | ecitizen | eCitizen Kenya | Governance | https://www.ecitizen.go.ke | ✅ Producing (thin — JS-heavy site) | 1 |
| 14 | kenya_met | Kenya Meteorological Department | Security | https://meteo.go.ke/news | ✅ Producing | 1 |
| 15 | ca_cyber | Communications Authority of Kenya | Security | https://www.ca.go.ke/cyber-security | ⚠️ Scraped rows all failed the PSA audit (informational) | 0 |
| 16 | kicd_news | Kenya Institute of Curriculum Development | Education | https://kicd.ac.ke/news | ⚠️ Scraped rows all failed the PSA audit | 0 |
| 17 | unicef_parenting | UNICEF Parenting | Health | https://www.unicef.org/parenting/health | ⛔ Blocked by robots.txt — respected, zero rows collected | 0 |
| 18 | ntsa | NTSA | Security | https://www.ntsa.go.ke/news | ⚠️ Configured, no usable text yield (cert chain + thin pages) | 0 |
| 19 | nps_kenya | National Police Service | Security | https://www.nationalpolice.go.ke/news | ⚠️ Configured, no usable text yield | 0 |
| 20 | kilimo | Ministry of Agriculture | Agriculture | https://kilimo.go.ke/category/news/ | ⚠️ Configured, no usable text yield | 0 |

Evaluated and dropped during Week 1: **Ministry of Health**
(https://www.health.go.ke) — advisories are published as PDFs/images, not
scrapable text. Replaced by Amref Health Africa (#4).

## Parallel corpora (bulk EN–SW volume)

| Source | Domain | URL | Languages | Status | Rows |
|--------|--------|-----|-----------|--------|-----:|
| TICO-19 | Health | https://tico-19.github.io/ | EN–SW (human-translated) | ✅ **CC BY 4.0**; kept whole as corpus data (`type:"corpus"`) — crisis/health domain, highly PSA-adjacent; its per-row `psa_class` is recorded in Metadata | 3,004 |
| Tatoeba | Mixed (keyword heuristic) | https://tatoeba.org/en/downloads | EN–SW | ⬜ Optional, not yet used (**CC BY 2.0 FR**); rows matching no PSA-domain keyword would be dropped by the relevance filter | 0 |

## Team-written PSAs (tier 3)

| Source | Domains | Status | Rows |
|--------|---------|--------|-----:|
| Team-written (`data/manual/team_psas_generated.csv`) | Education, Security, Governance | ✅ 150 original EN–SW pairs, 10 per sub-topic × 15 sub-topics (see `docs/team_written_psa_kit.md`) | 150 |

## robots.txt & rate-limit policy (summary; full text in docs/ETHICS.md)

- robots.txt is fetched (with the site's own SSL settings) and consulted
  before **every** request; disallowed pages are skipped, never circumvented.
  UNICEF (#17) is the concrete example: genuinely disallowed → zero rows.
  An *unreachable* robots.txt is treated as allow-with-logged-warning, not as
  a disallow rule.
- Every request waits **2 s + random 0–1 s jitter**, sends an identifying
  User-Agent (`PSA-Research-Bot/1.0 (university NLP student project)`), and
  uses a 30 s timeout. Per-site page caps bound total load; per-site record
  caps and word-count floors keep any single source from swamping the dataset.
- Failures warn and continue; the pipeline never retries aggressively.
- Several `.go.ke` sites serve broken certificate chains (made worse by local
  antivirus TLS interception); those configs carry `verify_ssl: False`, and
  the exception is documented per site rather than applied globally.
