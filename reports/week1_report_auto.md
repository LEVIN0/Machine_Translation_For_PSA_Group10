# Week 1 Report — Machine Translation of Public Service Announcements (Kenya)

_Generated: 2026-07-29 — DSA4020A, Group 10_

## Overview

- **Total rows:** 13519
- **Paired EN–SW rows (Kiswahili filled):** 3137
- **Distinct sources:** 18
- **Dataset file:** `psa_parallel_week1.csv`

## Rows per Domain

| Domain | Rows |
|---|---|
| Health | 8655 |
| Agriculture | 2409 |
| Governance | 1386 |
| Security | 654 |
| Education | 415 |

## Rows per Source

| Source | Rows |
|---|---|
| WHO | 4832 |
| TICO-19 (Translation Initiative for COVID-19) | 3004 |
| Infonet-Biovision (Biovision Africa Trust) | 2409 |
| Kenya Revenue Authority | 999 |
| Amref Health Africa | 416 |
| Kenya Red Cross | 403 |
| Ethics and Anti-Corruption Commission | 335 |
| Ministry of Education | 256 |
| National Environment Management Authority | 241 |
| Team-written | 150 |
| COVAW (Coalition on Violence Against Women) | 128 |
| Directorate of Criminal Investigations | 115 |
| KUCCPS | 96 |
| NACADA | 84 |
| Kenya Meteorological Department | 28 |
| Kenya Institute of Curriculum Development | 13 |
| Communications Authority of Kenya | 8 |
| eCitizen Kenya | 2 |

## Sentence length (words)

| Language | Mean | Median |
|---|---|---|
| English | 19.4 | 17 |
| Kiswahili | 23.7 | 21 |

## Sample rows (first 5)

| PSA_ID | Domain | English | Kiswahili | Source |
|---|---|---|---|---|
| PSA000001 | Health | WHO is continuously monitoring and responding to this pandemic. |  | WHO |
| PSA000002 | Health | This questions and answers page will be updated as more is known about COVID-19, how it spreads and how it is affecting… |  | WHO |
| PSA000003 | Health | For more information, regularly check the WHO coronavirus pages. https://www.who.int/covid-19 |  | WHO |
| PSA000004 | Health | COVID-19 is the disease caused by a coronavirus called SARS-CoV-2. |  | WHO |
| PSA000005 | Health | WHO first learned of this new virus on 31 December 2019, following a report of a cluster of cases of so-called viral pn… |  | WHO |

## Challenges faced

<!-- Editable: add/adjust bullets as the week progresses. -->
- Scarcity of truly bilingual government pages — most Kenyan PSA sites publish in English only, so Kiswahili must come from corpora (TICO-19) or be translated in later weeks.
- JS-heavy sites render content client-side; requests+BeautifulSoup only sees the static shell (Selenium is an optional fallback, not used here).
- robots.txt blocks on some origins — respected by design (see docs/ETHICS.md), reducing yield from those sites.
- Mixed-language pages (English/Kiswahili code-switching) require language-detection filtering in the cleaning stage.
- Tatoeba domain noise — general-domain sentences; only rows matching a PSA domain keyword heuristic are kept, the rest are dropped.

## Next steps (Week 2)

- Native-speaker validation of scraped and corpus rows (Status: Pending -> Validated/Rejected).
- Decide the TICO-19 dev/test split reservation — never train on the evaluation split.
- Scale up Tatoeba/OPUS EN-SW volume with the relevance heuristic.
- Begin team-written PSA expansion (target ~100-200 authentic-style rows).
- First baseline MT experiments (English -> Kiswahili).
