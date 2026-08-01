"""Week 2 train/dev/test splitting with domain stratification and
leakage-safe grouping.

Design decisions:
- Stratified by Domain: per-domain quotas via largest-remainder rounding so
  split sizes hit the requested ratios exactly when group sizes allow.
- Grouped by a normalized English key (lowercase, alphanumeric-only): all
  rows sharing a key (duplicates / near-identical re-scrapes) go to the SAME
  split, so they cannot leak across train/dev/test.
- Reproducible: all shuffling uses random.Random(seed).
"""

import json
import math
import random
import re

import pandas as pd

from . import config

_KEY_RE = re.compile(r"[^a-z0-9]+")


def group_key(text):
    """Normalized grouping key: lowercase, alphanumeric characters only."""
    return _KEY_RE.sub("", str(text or "").lower())


def _largest_remainder(exact, target):
    """Round {key: float} to integers summing exactly to `target`
    (largest fractional parts rounded up first, ties by key order)."""
    floors = {k: int(math.floor(v)) for k, v in exact.items()}
    remainder = target - sum(floors.values())
    if remainder > 0:
        order = sorted(exact, key=lambda k: (exact[k] - floors[k]),
                       reverse=True)
        for k in order[:remainder]:
            floors[k] += 1
    return floors


def make_splits(df, train=0.90, dev=0.05, test=0.05, seed=42):
    """Split df into (train, dev, test) DataFrames.

    Stratified by Domain; grouped by group_key(English) so rows with
    near-identical English text never straddle splits. Ratios must sum to 1.
    Returns three fresh DataFrames (reset index, original column order).
    """
    if not math.isclose(train + dev + test, 1.0, abs_tol=1e-9):
        raise ValueError(f"ratios must sum to 1.0, got {train + dev + test}")

    df = df.reset_index(drop=True)
    keys = df["English"].map(group_key)

    # Build groups: key -> {"domain": ..., "indices": [...]}
    groups = {}
    for idx, (key, domain) in enumerate(zip(keys, df["Domain"])):
        grp = groups.setdefault(key, {"domain": domain, "indices": []})
        grp["indices"].append(idx)

    by_domain = {}
    for key, grp in groups.items():
        by_domain.setdefault(grp["domain"], []).append((key, len(grp["indices"])))

    n_total = len(df)
    n_test = int(round(n_total * test))
    n_dev = int(round(n_total * dev))

    domain_rows = {d: sum(size for _, size in gs) for d, gs in by_domain.items()}
    test_quota = _largest_remainder(
        {d: n * test for d, n in domain_rows.items()}, n_test)
    dev_quota = _largest_remainder(
        {d: n * dev for d, n in domain_rows.items()}, n_dev)

    rng = random.Random(seed)
    assignment = {}  # group key -> split label
    for domain in sorted(by_domain):
        groups_d = by_domain[domain]
        rng.shuffle(groups_d)
        remaining_test = test_quota.get(domain, 0)
        remaining_dev = dev_quota.get(domain, 0)
        for key, size in groups_d:
            if remaining_test > 0:
                assignment[key] = "test"
                remaining_test -= size
            elif remaining_dev > 0:
                assignment[key] = "dev"
                remaining_dev -= size
            else:
                assignment[key] = "train"

    labels = keys.map(assignment)
    train_df = df[labels == "train"].reset_index(drop=True)
    dev_df = df[labels == "dev"].reset_index(drop=True)
    test_df = df[labels == "test"].reset_index(drop=True)
    return train_df, dev_df, test_df


def save_splits(train, dev, test, out_dir=None, seed=42, ratios=(0.90, 0.05, 0.05)):
    """Write train/dev/test CSVs plus split_stats.json to out_dir.

    split_stats.json records sizes, per-domain counts per split, the seed and
    ratios, and a leakage check (overlapping group keys between splits,
    asserted to be zero). `out_dir` defaults to PROCESSED_DIR/"splits"
    (resolved at call time so tests can redirect config paths).
    """
    out_dir = out_dir or (config.PROCESSED_DIR / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, split_df in (("train", train), ("dev", dev), ("test", test)):
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False, encoding="utf-8")
        written[name] = path

    # Leakage check: group keys must be pairwise disjoint across splits.
    key_sets = {
        name: set(split_df["English"].map(group_key))
        for name, split_df in (("train", train), ("dev", dev), ("test", test))
    }
    overlap = (key_sets["train"] & key_sets["dev"]
               | key_sets["train"] & key_sets["test"]
               | key_sets["dev"] & key_sets["test"])
    assert not overlap, f"leakage across splits: {len(overlap)} shared keys"

    stats = {
        "seed": seed,
        "ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "per_domain": {
            name: split_df["Domain"].value_counts().to_dict()
            for name, split_df in (("train", train), ("dev", dev), ("test", test))
        },
        "leakage_overlapping_keys": 0,
    }
    stats_path = out_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[splits] train={len(train)} dev={len(dev)} test={len(test)} "
          f"-> {out_dir}")
    return stats_path
