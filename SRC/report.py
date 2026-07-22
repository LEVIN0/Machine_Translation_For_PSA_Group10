"""Week 1 report generator: reads the built CSV and writes a markdown report."""

from datetime import date
from pathlib import Path

import pandas as pd

from .config import DATASET_CSV, REPORTS_DIR, ensure_dirs

_CHALLENGES = [
    "Scarcity of truly bilingual government pages — most Kenyan PSA sites "
    "publish in English only, so Kiswahili must come from corpora (TICO-19) "
    "or be translated in later weeks.",
    "JS-heavy sites render content client-side; requests+BeautifulSoup only "
    "sees the static shell (Selenium is an optional fallback, not used here).",
    "robots.txt blocks on some origins — respected by design (see "
    "docs/ETHICS.md), reducing yield from those sites.",
    "Mixed-language pages (English/Kiswahili code-switching) require "
    "language-detection filtering in the cleaning stage.",
    "Tatoeba domain noise — general-domain sentences; only rows matching a "
    "PSA domain keyword heuristic are kept, the rest are dropped.",
]

_NEXT_STEPS = [
    "Native-speaker validation of scraped and corpus rows (Status: Pending "
    "-> Validated/Rejected).",
    "Decide the TICO-19 dev/test split reservation — never train on the "
    "evaluation split.",
    "Scale up Tatoeba/OPUS EN-SW volume with the relevance heuristic.",
    "Begin team-written PSA expansion (target ~100-200 authentic-style rows).",
    "First baseline MT experiments (English -> Kiswahili).",
]


def _truncate(text, width=120):
    """Truncate text for the sample table."""
    text = str(text)
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _md_escape(text):
    """Escape pipe characters so markdown tables don't break."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def generate_report(csv_path=DATASET_CSV,
                    out_path=REPORTS_DIR / "week1_report.md"):
    """Generate the Week 1 markdown report from the built dataset CSV."""
    ensure_dirs()
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    total = len(df)
    paired = int(((df["Kiswahili"].str.strip() != "")).sum())
    n_sources = int(df["Source"].replace("", pd.NA).dropna().nunique())

    lines = []
    lines.append("# Week 1 Report — Machine Translation of Public Service Announcements (Kenya)")
    lines.append("")
    lines.append(f"_Generated: {date.today().isoformat()} — DSA4020A, Group 10_")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total rows:** {total}")
    lines.append(f"- **Paired EN–SW rows (Kiswahili filled):** {paired}")
    lines.append(f"- **Distinct sources:** {n_sources}")
    lines.append(f"- **Dataset file:** `{csv_path.name}`")
    lines.append("")

    # Rows per Domain
    lines.append("## Rows per Domain")
    lines.append("")
    lines.append("| Domain | Rows |")
    lines.append("|---|---|")
    for domain, count in df["Domain"].value_counts().items():
        lines.append(f"| {domain} | {count} |")
    lines.append("")

    # Rows per Source
    lines.append("## Rows per Source")
    lines.append("")
    lines.append("| Source | Rows |")
    lines.append("|---|---|")
    for source, count in df["Source"].value_counts().items():
        lines.append(f"| {_md_escape(source)} | {count} |")
    lines.append("")

    # Word counts
    lines.append("## Sentence length (words)")
    lines.append("")
    lines.append("| Language | Mean | Median |")
    lines.append("|---|---|---|")
    en_wc = df["English"].map(lambda s: len(str(s).split()))
    sw_wc = df.loc[df["Kiswahili"].str.strip() != "", "Kiswahili"].map(
        lambda s: len(str(s).split()))
    if total:
        lines.append(f"| English | {en_wc.mean():.1f} | {en_wc.median():.0f} |")
    else:
        lines.append("| English | – | – |")
    if len(sw_wc):
        lines.append(f"| Kiswahili | {sw_wc.mean():.1f} | {sw_wc.median():.0f} |")
    else:
        lines.append("| Kiswahili | – (no paired rows yet) | – |")
    lines.append("")

    # Sample rows
    lines.append("## Sample rows (first 5)")
    lines.append("")
    lines.append("| PSA_ID | Domain | English | Kiswahili | Source |")
    lines.append("|---|---|---|---|---|")
    for _, row in df.head(5).iterrows():
        lines.append(
            f"| {row['PSA_ID']} | {row['Domain']} "
            f"| {_md_escape(_truncate(row['English']))} "
            f"| {_md_escape(_truncate(row['Kiswahili']))} "
            f"| {_md_escape(row['Source'])} |"
        )
    lines.append("")

    # Challenges
    lines.append("## Challenges faced")
    lines.append("")
    lines.append("<!-- Editable: add/adjust bullets as the week progresses. -->")
    for item in _CHALLENGES:
        lines.append(f"- {item}")
    lines.append("")

    # Next steps
    lines.append("## Next steps (Week 2)")
    lines.append("")
    for item in _NEXT_STEPS:
        lines.append(f"- {item}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out_path}")
    return out_path
