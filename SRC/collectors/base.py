"""Generic site collector driven by a per-site config dict.

A site config (see SRC/collectors/sites.py) provides:
    name               short identifier, e.g. "who_covid_qna"
    domain             one of config.DOMAINS (an alias like "Road Safety" is
                       mapped via DOMAIN_ALIAS, e.g. "Road Safety" -> "Security")
    source             human-readable source name, e.g. "WHO"
    start_urls         list of URLs to begin from
    link_patterns      list of regex strings matched against absolute hrefs;
                       empty list = treat start_urls themselves as article pages
    content_selectors  CSS selectors tried in order; fallback "p"
    max_pages          cap on article pages per run (default 25)
    split_sentences    split extracted blocks into sentences (default True)
    verify_ssl         verify TLS certificates (default True; set False only
                       for sites with broken certificate chains)

All network errors are non-fatal: warn and continue; collect() returns [] on
total failure.
"""

import re

from ..config import MAX_WORDS, MIN_WORDS
from ..schema import new_record
from ..scraper import absolutize, get_soup

# Domain aliases applied before emitting records ("Road Safety" -> "Security").
DOMAIN_ALIAS = {"Road Safety": "Security"}

# Case-insensitive boilerplate markers; blocks containing these are dropped.
# NOTE: "Ⓒ" (U+24B8, circled C) is a different character from "©" (U+00A9)
# and is used by Infonet-Biovision for photo-credit captions.
BOILERPLATE = (
    # NOTE: text is lowercased before matching, and "Ⓒ".lower() == "ⓒ" —
    # both circled-C forms must be listed (photo-credit captions).
    "cookie", "subscribe", "all rights reserved", "©", "Ⓒ", "ⓒ", "(c)",
    "sign up", "log in", "share this", "javascript",
)

# Sentence split: after .!? followed by whitespace and an uppercase/digit/quote.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'])")


class SiteCollector:
    """Collect PSA-style sentences from one configured website."""

    def __init__(self, cfg):
        """Store config and apply defaults for optional settings."""
        self.cfg = dict(cfg)
        self.cfg.setdefault("max_pages", 25)
        self.cfg.setdefault("split_sentences", True)
        self.cfg.setdefault("verify_ssl", True)
        self.cfg.setdefault("min_words", MIN_WORDS)
        # max_records: optional cap; collect() stops once reached (None = no cap)
        self.cfg.setdefault("max_records", None)

    def collect(self, verbose=True):
        """Fetch pages, extract text blocks, filter, and emit schema records."""
        records = []
        seen = set()
        cap = self.cfg.get("max_records")
        for page_url in self._page_urls(verbose=verbose):
            if cap is not None and len(records) >= cap:
                break
            soup = get_soup(page_url, verify=self.cfg["verify_ssl"])
            if soup is None:
                continue  # network/parse error already warned; keep going
            for block in self._extract_blocks(soup):
                if cap is not None and len(records) >= cap:
                    break
                for text in self._units(block):
                    if cap is not None and len(records) >= cap:
                        break
                    text = self._clean_unit(text)
                    if not text or not self._keep(text):
                        continue
                    key = " ".join(text.lower().split())
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(new_record(
                        domain=self._domain(),
                        english=text,
                        source=self.cfg.get("source", ""),
                        url=page_url,
                        metadata={
                            "type": "scraped",
                            "tool": "requests+bs4",
                            "site": self.cfg["name"],
                        },
                    ))
        if verbose:
            print(f"[collect] {self.cfg['name']}: {len(records)} records")
        return records

    # -- internal helpers -----------------------------------------------------

    def _domain(self):
        """Return the configured domain, resolving aliases (Road Safety->Security)."""
        raw = self.cfg.get("domain", "")
        return DOMAIN_ALIAS.get(raw, raw)

    def _page_urls(self, verbose=True):
        """Resolve start_urls to the final list of article page URLs."""
        start_urls = list(self.cfg.get("start_urls", []))
        pag = self.cfg.get("pagination")
        if pag:
            start = pag.get("start", 1)
            for n in range(start, start + pag.get("pages", 3)):
                start_urls.append(pag["template"].format(n=n))
        link_patterns = self.cfg.get("link_patterns", [])
        max_pages = self.cfg.get("max_pages", 25)
        if not link_patterns:
            return list(start_urls)
        patterns = [re.compile(p) for p in link_patterns]
        pages = []
        seen = set()
        for start in start_urls:
            soup = get_soup(start, verify=self.cfg["verify_ssl"])
            if soup is None:
                continue  # non-fatal
            for a in soup.find_all("a", href=True):
                href = absolutize(start, a["href"])
                if href in seen:
                    continue
                if any(p.search(href) for p in patterns):
                    seen.add(href)
                    pages.append(href)
                    if len(pages) >= max_pages:
                        break
            if len(pages) >= max_pages:
                break
        if verbose:
            print(f"[collect] {self.cfg['name']}: {len(pages)} page(s) discovered")
        return pages

    def _extract_blocks(self, soup):
        """Try content_selectors in order; first selector yielding >=1 block wins."""
        selectors = list(self.cfg.get("content_selectors", [])) or ["p"]
        if "p" not in selectors:
            selectors.append("p")  # fallback
        for sel in selectors:
            try:
                els = soup.select(sel)
            except Exception:
                continue  # invalid selector for this parser; try next
            if els:
                return [el.get_text(" ", strip=True) for el in els]
        return []

    def _units(self, block):
        """Split a text block into sentence units if split_sentences is on."""
        if self.cfg.get("split_sentences", True):
            return SENTENCE_RE.split(block)
        return [block]

    @staticmethod
    def _clean_unit(text):
        """Collapse internal whitespace and strip."""
        return " ".join((text or "").split()).strip()

    def _keep(self, text):
        """Apply word-count bounds and boilerplate filters."""
        n_words = len(text.split())
        if n_words < self.cfg["min_words"] or n_words > MAX_WORDS:
            return False
        low = text.lower()
        return not any(marker in low for marker in BOILERPLATE)
