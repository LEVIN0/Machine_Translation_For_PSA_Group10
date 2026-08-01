"""Week 2 exploratory data analysis: statistics, figures, and a markdown
EDA report for the PSA parallel dataset.

All functions accept a plain schema dataframe (the Week 1 CSV); preprocessing
columns are recomputed internally where needed, so this module works with or
without `preprocess_dataframe` having run first.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless servers have no display — MUST precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .preprocessing import (  # noqa: E402
    is_codeswitched,
    load_glossary,
    glossary_hits,
    word_tokens,
)

_DPI = 150


def _word_counts(series):
    """Word-token counts for a text series."""
    return series.fillna("").map(lambda s: len(word_tokens(s)))


def _length_stats(counts):
    """Mean/median/min/max of a word-count series (zeros for empty input)."""
    if len(counts) == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "min": int(counts.min()),
        "max": int(counts.max()),
    }


def compute_eda(df):
    """Compute the Week 2 EDA statistics dict from a schema dataframe.

    Covers: row totals, per-domain and per-source counts, paired (EN+SW) vs
    unpaired rows, EN/SW word-length mean/median/min/max, vocabulary sizes
    (unique lowercased word tokens) and type-token ratios for EN and SW,
    code-switched row count (English column), glossary coverage, missing-
    translation share, and per-domain mean English length.
    """
    total = len(df)
    sw = df["Kiswahili"].fillna("")
    paired_mask = sw.str.strip() != ""
    paired = int(paired_mask.sum())

    en_counts = _word_counts(df["English"])
    sw_counts = _word_counts(sw[paired_mask])

    en_tokens = [t for s in df["English"].fillna("") for t in word_tokens(s)]
    sw_tokens = [t for s in sw[paired_mask] for t in word_tokens(s)]

    glossary = load_glossary()
    glossary_rows = int(df["English"].fillna("").map(
        lambda s: bool(glossary_hits(s, glossary))).sum()) if glossary else 0
    codeswitched = int(df["English"].fillna("").map(is_codeswitched).sum())

    per_domain_mean = (
        df.assign(_wc=en_counts).groupby("Domain")["_wc"].mean().to_dict()
        if total else {}
    )

    return {
        "rows_total": total,
        "per_domain": df["Domain"].value_counts().to_dict() if total else {},
        "per_source": df["Source"].value_counts().to_dict() if total else {},
        "paired": paired,
        "unpaired": total - paired,
        "paired_share": paired / total if total else 0.0,
        "length_en": _length_stats(en_counts),
        "length_sw": _length_stats(sw_counts),
        "vocab_en": len(set(en_tokens)),
        "vocab_sw": len(set(sw_tokens)),
        "type_token_ratio_en": (len(set(en_tokens)) / len(en_tokens)
                                if en_tokens else 0.0),
        "type_token_ratio_sw": (len(set(sw_tokens)) / len(sw_tokens)
                                if sw_tokens else 0.0),
        "codeswitched_rows": codeswitched,
        "glossary_rows": glossary_rows,
        "missing_translation_share": (total - paired) / total if total else 0.0,
        "per_domain_mean_length": per_domain_mean,
    }


def _save(fig, out_dir, name, paths):
    """Save one figure at 150 dpi and record its path."""
    path = out_dir / name
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)


def _empty_fig(title, out_dir, name, paths):
    """Write a placeholder figure for empty data (keeps the 6-PNG contract)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
    ax.set_title(title)
    ax.set_axis_off()
    _save(fig, out_dir, name, paths)


def make_figures(df, out_dir=None):
    """Write the six Week 2 EDA figures as PNGs; return their paths.

    Figures (150 dpi, matplotlib default style, no seaborn, titles + axis
    labels): domain_bar, domain_pie, length_hist_english,
    length_hist_kiswahili (paired rows only), source_bar, paired_vs_unpaired.
    `out_dir` defaults to REPORTS_DIR/"figures" (resolved at call time).
    """
    out_dir = Path(out_dir or (config.REPORTS_DIR / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    sw = df["Kiswahili"].fillna("")
    paired_mask = sw.str.strip() != ""
    en_counts = _word_counts(df["English"])
    sw_counts = _word_counts(sw[paired_mask])

    # 1. Domain distribution — bar
    domain_counts = df["Domain"].value_counts()
    if len(domain_counts):
        fig, ax = plt.subplots(figsize=(7, 4))
        domain_counts.plot.bar(ax=ax, color="#4C78A8")
        ax.set_title("Rows per Domain")
        ax.set_xlabel("Domain")
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, out_dir, "domain_bar.png", paths)
    else:
        _empty_fig("Rows per Domain", out_dir, "domain_bar.png", paths)

    # 2. Domain distribution — pie
    if len(domain_counts):
        fig, ax = plt.subplots(figsize=(6, 6))
        domain_counts.plot.pie(ax=ax, autopct="%1.1f%%", startangle=90)
        ax.set_title("Domain Share")
        ax.set_ylabel("")
        _save(fig, out_dir, "domain_pie.png", paths)
    else:
        _empty_fig("Domain Share", out_dir, "domain_pie.png", paths)

    # 3. English text-length histogram (all rows)
    if len(en_counts):
        fig, ax = plt.subplots(figsize=(7, 4))
        en_counts.plot.hist(ax=ax, bins=30, color="#4C78A8")
        ax.set_title("English Sentence Length Distribution")
        ax.set_xlabel("Words per row")
        ax.set_ylabel("Rows")
        _save(fig, out_dir, "length_hist_english.png", paths)
    else:
        _empty_fig("English Sentence Length Distribution", out_dir,
                   "length_hist_english.png", paths)

    # 4. Kiswahili text-length histogram (paired rows only)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(sw_counts):
        sw_counts.plot.hist(ax=ax, bins=30, color="#59A14F")
    ax.set_title("Kiswahili Sentence Length Distribution (paired rows only)")
    ax.set_xlabel("Words per row")
    ax.set_ylabel("Rows")
    _save(fig, out_dir, "length_hist_kiswahili.png", paths)

    # 5. Rows per source — horizontal bar
    source_counts = df["Source"].value_counts()
    if len(source_counts):
        fig, ax = plt.subplots(
            figsize=(7, max(3, 0.35 * len(source_counts) + 1)))
        source_counts.sort_values().plot.barh(ax=ax, color="#F28E2B")
        ax.set_title("Rows per Source")
        ax.set_xlabel("Rows")
        ax.set_ylabel("Source")
        _save(fig, out_dir, "source_bar.png", paths)
    else:
        _empty_fig("Rows per Source", out_dir, "source_bar.png", paths)

    # 6. Paired (EN+SW) vs unpaired rows
    fig, ax = plt.subplots(figsize=(5, 4))
    paired_counts = pd.Series(
        {"Paired (EN+SW)": int(paired_mask.sum()),
         "Unpaired (EN only)": int((~paired_mask).sum())})
    paired_counts.plot.bar(ax=ax, color=["#59A14F", "#E15759"])
    ax.set_title("Paired vs Unpaired Rows")
    ax.set_xlabel("Row type")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out_dir, "paired_vs_unpaired.png", paths)

    print(f"[eda] wrote {len(paths)} figures -> {out_dir}")
    return paths


def _observations(stats):
    """Compute the 'Key observations' bullets from the numbers."""
    total = stats["rows_total"]
    bullets = []
    if not total:
        return ["Dataset is empty — nothing to observe yet."]

    top_domain, top_n = max(stats["per_domain"].items(), key=lambda kv: kv[1])
    top_share = top_n / total * 100
    bullets.append(
        f"**{top_domain}** dominates with {top_n:,} rows ({top_share:.1f}% of "
        "the dataset) — plan downsampling or targeted collection in the other "
        "domains before training.")

    paired_share = stats["paired_share"] * 100
    bullets.append(
        f"Only {stats['paired']:,} of {total:,} rows ({paired_share:.1f}%) "
        "have a Kiswahili translation — English-only rows need translation "
        "(team-written kit + Week 3 back-translation) or exclusion from "
        "supervised training.")

    missing = stats["missing_translation_share"] * 100
    bullets.append(
        f"Missing-translation share is {missing:.1f}%; Ekegusii is 100% empty "
        "and remains the Week 3 few-shot transfer target.")

    if stats["codeswitched_rows"]:
        bullets.append(
            f"{stats['codeswitched_rows']:,} rows look code-switched "
            "(mixed EN/SW stopwords) — review before training; they may need "
            "re-translation or a dedicated handling rule.")
    else:
        bullets.append("No code-switched rows detected by the stopword heuristic.")

    if stats["glossary_rows"]:
        share = stats["glossary_rows"] / total * 100
        bullets.append(
            f"Glossary terms appear in {stats['glossary_rows']:,} rows "
            f"({share:.1f}%) — these cultural/institutional terms need "
            "consistent handling (see data/glossary.json).")

    len_en = stats["length_en"]
    bullets.append(
        f"English rows average {len_en['mean']:.1f} words "
        f"(median {len_en['median']:.0f}, range {len_en['min']}–"
        f"{len_en['max']}); EN vocabulary is {stats['vocab_en']:,} types "
        f"(type-token ratio {stats['type_token_ratio_en']:.3f}).")

    longest_domain = max(stats["per_domain_mean_length"].items(),
                         key=lambda kv: kv[1])
    bullets.append(
        f"Longest sentences come from **{longest_domain[0]}** "
        f"(mean {longest_domain[1]:.1f} words) — check the Week 3 "
        "subword-segmentation budget against this domain.")
    return bullets


def write_eda_report(stats, figures, out_path=None):
    """Write the Week 2 EDA markdown report; returns the report path.

    Includes an overview, embedded figures (relative `figures/x.png` paths),
    stats tables, and a 'Key observations' section computed from the numbers.
    `out_path` defaults to REPORTS_DIR/"week2_eda_report.md".
    """
    out_path = Path(out_path or (config.REPORTS_DIR / "week2_eda_report.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = stats["rows_total"]
    lines = []
    lines.append("# Week 2 EDA Report — PSA Parallel Dataset (EN–SW)")
    lines.append("")
    lines.append("_DSA4020A, Group 10 — generated by `src/eda.py`_")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total rows:** {total:,}")
    lines.append(f"- **Paired rows (EN+SW):** {stats['paired']:,} "
                 f"({stats['paired_share'] * 100:.1f}%)")
    lines.append(f"- **Unpaired rows (EN only):** {stats['unpaired']:,}")
    lines.append(f"- **Code-switched rows (EN column):** "
                 f"{stats['codeswitched_rows']:,}")
    lines.append(f"- **Rows containing glossary terms:** "
                 f"{stats['glossary_rows']:,}")
    lines.append(f"- **Vocabulary (EN / SW):** {stats['vocab_en']:,} / "
                 f"{stats['vocab_sw']:,} unique word types")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for path in figures:
        name = Path(path).name
        lines.append(f"![{Path(name).stem}](figures/{name})")
        lines.append("")

    lines.append("## Domain distribution")
    lines.append("")
    lines.append("| Domain | Rows | Share | Mean EN words |")
    lines.append("|---|---|---|---|")
    for domain, count in sorted(stats["per_domain"].items(),
                                key=lambda kv: kv[1], reverse=True):
        share = count / total * 100 if total else 0.0
        mean_len = stats["per_domain_mean_length"].get(domain, 0.0)
        lines.append(f"| {domain} | {count:,} | {share:.1f}% | {mean_len:.1f} |")
    lines.append("")

    lines.append("## Source distribution")
    lines.append("")
    lines.append("| Source | Rows |")
    lines.append("|---|---|")
    for source, count in sorted(stats["per_source"].items(),
                                key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {source} | {count:,} |")
    lines.append("")

    lines.append("## Sentence length (word tokens)")
    lines.append("")
    lines.append("| Language | Mean | Median | Min | Max |")
    lines.append("|---|---|---|---|---|")
    for lang, key in (("English", "length_en"),
                      ("Kiswahili (paired rows)", "length_sw")):
        s = stats[key]
        lines.append(f"| {lang} | {s['mean']:.1f} | {s['median']:.0f} "
                     f"| {s['min']} | {s['max']} |")
    lines.append("")

    lines.append("## Vocabulary")
    lines.append("")
    lines.append("| Language | Unique types | Type-token ratio |")
    lines.append("|---|---|---|")
    lines.append(f"| English | {stats['vocab_en']:,} "
                 f"| {stats['type_token_ratio_en']:.3f} |")
    lines.append(f"| Kiswahili | {stats['vocab_sw']:,} "
                 f"| {stats['type_token_ratio_sw']:.3f} |")
    lines.append("")

    lines.append("## Key observations")
    lines.append("")
    for bullet in _observations(stats):
        lines.append(f"- {bullet}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eda] wrote report -> {out_path}")
    return out_path
