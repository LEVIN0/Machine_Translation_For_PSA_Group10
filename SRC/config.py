"""
Project Configuration
---------------------
This file stores configuration variables used throughout the project.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
RAW_DATA = BASE_DIR / "data" / "raw"
PROCESSED_DATA = BASE_DIR / "data" / "processed"

# Output file
RAW_DATASET = RAW_DATA / "raw_psa.csv"

# PSA Categories
DOMAINS = [
    "Health",
    "Education",
    "Agriculture",
    "Security",
    "Governance"
]