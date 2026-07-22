# Ethics & Compliance — PSA Data Collection (Week 1)

This project collects public web text for a university NLP course (DSA4020A).
All collection is governed by the rules below; the code in `SRC/scraper.py`
enforces them mechanically, not just by convention.

## 1. robots.txt compliance

- Every URL is checked against its origin's `robots.txt` **before** fetching,
  via `urllib.robotparser.RobotFileParser` (`robots_allowed()` in
  `SRC/scraper.py`).
- One `RobotFileParser` is **cached per origin** so we never hammer a site
  with repeated robots.txt downloads.
- If a page is disallowed, the request is skipped and the block is printed
  (`Blocked by robots.txt: <url>`). We do not work around blocks.
- If `robots.txt` itself cannot be fetched (network error), the robotparser
  default applies (allow) and a warning is logged.

## 2. Rate limiting

- `polite_get()` sleeps **2.0 seconds plus a random 0–1 s jitter** before
  every single request (`REQUEST_DELAY = 2.0` in `SRC/config.py`).
- Requests carry a descriptive User-Agent:
  `PSA-Research-Bot/1.0 (university NLP student project)`.
- 30-second timeouts; all network failures are non-fatal warnings — the
  pipeline retries nothing aggressively and never loops on errors.
- `max_pages` caps (15–25 per site) bound total load per origin.

## 3. Licensing & attribution

| Source | License / terms | How we comply |
|---|---|---|
| TICO-19 | CC BY 4.0 | Attribution stored in each row's Metadata and printed on download. |
| Tatoeba | CC BY 2.0 FR | Attribution stored in each row's Metadata; downloads are manual. |
| WHO / UNICEF / Kenyan government sites | Site-specific terms | Small excerpts of **public service** text, collected for non-commercial academic research with full source URL per row. |

- Every record carries its `Source`, `URL`, and a JSON `Metadata` blob with
  the license where one applies — provenance is never detached from data.
- We do **not** use FLORES-200/FLORES+ or the TICO-19 test split as training
  data; evaluation benchmarks are reserved for evaluation only.

## 4. No personal data

- We collect only public informational/announcement text. No forms, no
  logins, no comment sections, no user-generated personal data, no scraping
  behind authentication.
- Boilerplate (cookie banners, subscription prompts) is filtered out, and
  near-duplicate re-scrapes are removed in cleaning.

## 5. Educational use

- All output is for a supervised university course project; nothing is
  redistributed commercially. If any site owner objects, the corresponding
  source config in `SRC/collectors/sites.py` will be removed on request.
