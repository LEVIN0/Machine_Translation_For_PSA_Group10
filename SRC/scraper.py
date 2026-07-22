"""Polite HTTP scraping helpers: robots.txt compliance, rate limiting, parsing.

Ethics policy (see docs/ETHICS.md):
- robots.txt is checked via urllib.robotparser before every request, with one
  cached RobotFileParser per origin. If robots.txt cannot be fetched, the
  robotparser default applies (allow) and a warning is logged.
- Every request waits REQUEST_DELAY seconds plus a random 0-1s jitter, uses a
  descriptive User-Agent, and a 30s timeout.
- All network failures are non-fatal: warn and return None.
"""

import random
import time
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import REQUEST_DELAY, REQUEST_TIMEOUT, USER_AGENT

# One cached RobotFileParser per (origin, verify) pair.
_ROBOTS_CACHE = {}


def robots_allowed(url, user_agent=USER_AGENT, verify=True):
    """Return True if `url` may be fetched per the origin's robots.txt.

    robots.txt is fetched with `requests` (honouring `verify`, so sites with
    broken certificate chains can still have their robots.txt read) and parsed
    manually via RobotFileParser.parse(). Policy on fetch failure: log a
    warning and ALLOW — an unreachable robots.txt is not a disallow rule.
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    key = (origin, verify)
    rp = _ROBOTS_CACHE.get(key)
    if rp is None:
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{origin}/robots.txt")
        try:
            resp = requests.get(
                f"{origin}/robots.txt",
                headers={"User-Agent": user_agent},
                timeout=REQUEST_TIMEOUT,
                verify=verify,
            )
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            elif resp.status_code in (401, 403):
                rp.disallow_all = True  # per robotparser convention
            else:  # 404 etc.: no robots.txt -> everything allowed
                rp.allow_all = True
        except Exception as exc:
            print(f"[robots] WARNING: could not read robots.txt for {origin} "
                  f"({exc}); allowing by default")
            rp.allow_all = True
        _ROBOTS_CACHE[key] = rp
    try:
        return rp.can_fetch(user_agent, url)
    except Exception as exc:
        print(f"[robots] WARNING: can_fetch failed for {url} ({exc}); allowing")
        return True


def polite_get(url, delay=REQUEST_DELAY, verify=True):
    """Fetch `url` politely; return a requests.Response or None.

    Checks robots.txt first (prints and returns None if blocked), then sleeps
    `delay` seconds plus a random 0-1s jitter before issuing the request with
    the project User-Agent and a timeout. Any exception -> warn, return None.

    `verify=False` disables TLS certificate verification. Use ONLY for sites
    whose certificate chain is broken (e.g. some .go.ke servers) or when local
    antivirus TLS-inspection breaks verification; log a warning when disabled.
    """
    if not robots_allowed(url, verify=verify):
        print(f"[scraper] Blocked by robots.txt: {url}")
        return None
    time.sleep(delay + random.uniform(0, 1))
    if not verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            verify=verify,
        )
        resp.raise_for_status()
        return resp
    except Exception as exc:
        print(f"[scraper] WARNING: failed to fetch {url} ({exc})")
        return None


def get_soup(url, verify=True):
    """Fetch `url` and return a BeautifulSoup (lxml parser), or None on failure."""
    resp = polite_get(url, verify=verify)
    if resp is None:
        return None
    try:
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        print(f"[scraper] WARNING: failed to parse {url} ({exc})")
        return None


def absolutize(base_url, href):
    """Resolve a possibly-relative href against a base URL."""
    return urljoin(base_url, href)
