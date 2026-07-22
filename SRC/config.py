"""Central configuration for the PSA Week 1 data collection package.

Paths are defined relative to the project root (parent of this SRC package),
so the package is portable — no hard-coded absolute paths.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EXTERNAL_DIR = BASE_DIR / "data" / "external"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_CSV = PROCESSED_DIR / "psa_parallel_week1.csv"
STATS_JSON = PROCESSED_DIR / "build_stats.json"


def ensure_dirs() -> None:
    """Create all working directories (called by pipeline functions, not at import)."""
    for d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- Dataset definition ------------------------------------------------------
DOMAINS = ["Health", "Education", "Agriculture", "Security", "Governance"]

LANGUAGES = {"English": "en", "Kiswahili": "sw", "Ekegusii": "guz"}

# --- Scraping politeness -------------------------------------------------------
USER_AGENT = "PSA-Research-Bot/1.0 (university NLP student project)"
REQUEST_DELAY = 2.0          # seconds between requests (+ random 0-1s jitter)
REQUEST_TIMEOUT = 30         # seconds

# --- Text filtering bounds -----------------------------------------------------
MIN_WORDS = 4
MAX_WORDS = 80
