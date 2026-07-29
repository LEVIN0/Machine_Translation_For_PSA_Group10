# Framework audit — PSA classification remediation

_Generated 2026-07-29T08:43:17+00:00 by `scripts/remediate_dataset.py`_

## Methodology

Every dataset row was audited against the lecturer's 4-step PSA framework (PSA FRAMEWORK.pdf) using the frozen, lecturer-calibrated scoring classifier in `SRC/psa_classify.py`. The classifier is a documented heuristic: directive/imperative signals, audience markers and call-to-action phrases add score; press-release, legal-notice, encyclopedic and connective-continuation signals subtract. Rows scoring >= 2 are PSA. Per the locked team decision, only rows with Metadata.type == "scraped" are eligible for deletion; TICO-19 (corpus), team-written (manual) and lecturer gold rows are exempt and kept whole, with their psa_class recorded in Metadata for analysis. Deleted scraped rows are removed entirely from the dataset.

## Kept / dropped per source (Metadata.type, audit phase)

| source type | before | dropped | kept |
|---|---:|---:|---:|
| corpus | 3004 | 0 | 3004 |
| manual | 150 | 0 | 150 |
| scraped | 10365 | 9548 | 817 |

## Kept rows per domain

| domain | rows |
|---|---:|
| Agriculture | 559 |
| Education | 828 |
| Governance | 540 |
| Health | 3944 |
| Security | 952 |

## Sample of DELETED rows (up to 10)

| Source | psa_class | score | English (truncated) |
|---|---|---:|---|
| WHO | Informational | 0 | WHO is continuously monitoring and responding to this pandemic. |
| WHO | Informational | 0 | This questions and answers page will be updated as more is known about COVID-19, how it spreads and… |
| WHO | Informational | 0 | For more information, regularly check the WHO coronavirus pages. https://www.who.int/covid-19 |
| WHO | Informational | 0 | COVID-19 is the disease caused by a coronavirus called SARS-CoV-2. |
| WHO | Informational | 0 | WHO first learned of this new virus on 31 December 2019, following a report of a cluster of cases o… |
| WHO | Informational | -2 | The most common symptoms of COVID-19 are |
| WHO | Informational | 0 | These include, but are not limited to: those taking immunosuppressive medication; those with chroni… |
| WHO | Informational | 0 | As testing rates fall, it is more difficult to know how many people have COVID-19 and do not seek a… |
| WHO | Informational | 0 | At the start of the pandemic, 15% of people were thought to become seriously unwell and require hos… |
| WHO | Informational | 0 | More recent estimates suggest that hospitalization is required in around 3% of people with COVID-19. |

## Totals

- Rows before: **13519**
- Rows after: **6823**
- Deleted (scraped non-PSA): **9548**
- Class distribution (all rows, pre-deletion): `{"Informational": 12243, "PSA": 1168, "PressRelease": 93, "Legal": 15}`

## Lecturer gold merge

- Gold rows imported: **2895** (from `PSA_KE_Final.csv`)
- Domains: Agriculture (368), Education (768), Governance (403), Health (490), Security (866)
- Translation coverage: Kiswahili 2892/2895, Ekegusii 2891/2895, Dholuo 2889/2895, Somali 2893/2895
