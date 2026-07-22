# Data Sources — Week 1 (≥10 documented sources required by the brief)

All 12 web sources below are configured in `SRC/collectors/sites.py`; the two
corpora are imported by `SRC/corpora/`. URLs are starting points — every fetch
failure is non-fatal by design.

## Web sources (scraped)

| # | Source | Domain | URL | Languages | Notes |
|---|---|---|---|---|---|
| 1 | WHO (COVID-19 Q&A) | Health | https://www.who.int/news-room/questions-and-answers/item/coronavirus-disease-covid-19 | EN | PSA-style public-health guidance; check robots before fetch (enforced in code). |
| 2 | WHO (Fact Sheets) | Health | https://www.who.int/news-room/fact-sheets | EN | Link pattern `/news-room/fact-sheets/detail/`; academic use with attribution. |
| 3 | UNICEF Parenting | Health | https://www.unicef.org/parenting/health | EN | Child-health advisory text; attribution per row. |
| 4 | Kenya Red Cross | Health | https://www.redcross.or.ke/news | EN/SW (some) | Disaster/health announcements; small excerpts for research. |
| 5 | Ministry of Health Kenya | Health | https://www.health.go.ke/press-release | EN/SW (some) | Official government PSAs; public information. |
| 6 | NTSA | Security | https://www.ntsa.go.ke/news | EN | Road-safety announcements ("Road Safety" → Security per DOMAIN_ALIAS). |
| 7 | Kenya Meteorological Department | Security | https://meteo.go.ke/news | EN | Weather/flood warnings; public safety alerts. |
| 8 | Ministry of Agriculture (Kilimo) | Agriculture | https://kilimo.go.ke/category/news/ | EN | Farming advisories; WordPress category pagination. |
| 9 | Ministry of Education | Education | https://www.education.go.ke/news | EN | School/exam announcements. |
| 10 | Kenya Revenue Authority | Governance | https://www.kra.go.ke/news-center/public-notices | EN | Tax deadline notices; canonical PSAs. |
| 11 | eCitizen Kenya | Governance | https://www.ecitizen.go.ke | EN | Government service announcements; JS-heavy, low yield expected. |
| 12 | National Police Service | Security | https://www.nationalpolice.go.ke/news | EN | Public security advisories. |

## Parallel corpora (bulk EN–SW volume)

| Source | Domain | URL | Languages | Notes |
|---|---|---|---|---|
| TICO-19 | Health | https://tico-19.github.io/ | EN–SW (human-translated) | **CC BY 4.0**; crisis/health domain — highly PSA-adjacent. Split reservation (dev/test) is a Week 2 decision; never train on the evaluation split. |
| Tatoeba | Mixed (keyword heuristic) | https://tatoeba.org/en/downloads | EN–SW | **CC BY 2.0 FR**; general-domain — rows matching no PSA-domain keyword are **dropped** (relevance filter). |

## robots.txt & rate-limit policy (summary; full text in docs/ETHICS.md)

- `robots.txt` is consulted via `urllib.robotparser` before **every** request,
  with one parser cached per origin; disallowed pages are skipped, never
  circumvented.
- Every request waits **2 s + random 0–1 s jitter**, sends an identifying
  User-Agent (`PSA-Research-Bot/1.0 (university NLP student project)`), and
  uses a 30 s timeout. Per-site page caps (15–25) bound total load.
- Failures warn and continue; the pipeline never retries aggressively.
