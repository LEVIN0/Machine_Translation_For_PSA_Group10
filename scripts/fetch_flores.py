#!/usr/bin/env python3
"""Download FLORES-200 and emit the canonical Ekegusii TSVs (SPEC §2.2).

Downloads https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz,
extracts ONLY the four needed files
    dev/eng_Latn.dev, dev/guz_Latn.dev,
    devtest/eng_Latn.devtest, devtest/guz_Latn.devtest
and writes, under data/external/flores200/:
    guz_dev.tsv      (997 pairs)  — few-shot SEED source (trainable)
    guz_devtest.tsv  (1012 pairs) — evaluation ONLY, never train
Both TSVs: header ``eng<TAB>guz``, UTF-8, one sentence pair per line.

Prints the archive SHA256 and pair counts. Network failure -> clear error
message, exit 1. pathlib + UTF-8 everywhere; Windows-compatible.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# Archive member suffix -> local extract name (matched by suffix so the
# layout holds whether or not the tarball wraps files in a top-level dir).
NEEDED_MEMBERS = {
    "dev/eng_Latn.dev": "eng_dev",
    "dev/guz_Latn.dev": "guz_dev",
    "devtest/eng_Latn.devtest": "eng_devtest",
    "devtest/guz_Latn.devtest": "guz_devtest",
}

EXPECTED_COUNTS = {"guz_dev.tsv": 997, "guz_devtest.tsv": 1012}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "external" / "flores200"


def _download(url: str, dest: Path) -> str:
    """Download url -> dest; return the archive SHA256. Exit 1 on failure."""
    print(f"fetch_flores: downloading {url}")
    print("              (this is ~3 GB on first run; please be patient)")
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, \
                dest.open("wb") as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                sha.update(block)
                fh.write(block)
    except (urllib.error.URLError, OSError) as exc:
        print(f"fetch_flores: ERROR downloading FLORES-200: {exc}",
              file=sys.stderr)
        print("fetch_flores: check your network connection and retry; the "
              "dataset URL is " + url, file=sys.stderr)
        sys.exit(1)
    return sha.hexdigest()


def _extract_needed(archive: Path, work_dir: Path) -> dict[str, Path]:
    """Extract only the four needed members; return name -> local path."""
    found: dict[str, Path] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            norm = member.name.lstrip("./")
            for suffix, key in NEEDED_MEMBERS.items():
                if norm.endswith(suffix) and key not in found:
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    dest = work_dir / key
                    dest.write_bytes(src.read())
                    found[key] = dest
    missing = sorted(set(NEEDED_MEMBERS.values()) - set(found))
    if missing:
        print(f"fetch_flores: ERROR archive is missing members: {missing}",
              file=sys.stderr)
        sys.exit(1)
    return found


def _read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.strip()]


def _write_tsv(eng_path: Path, guz_path: Path, out_path: Path) -> int:
    """Join eng/guz line-by-line into a canonical TSV; assert equal counts."""
    eng, guz = _read_lines(eng_path), _read_lines(guz_path)
    if len(eng) != len(guz):
        print(f"fetch_flores: ERROR line-count mismatch for {out_path.name}: "
              f"eng={len(eng)} guz={len(guz)}", file=sys.stderr)
        sys.exit(1)
    lines = ["eng\tguz"] + [f"{e}\t{g}" for e, g in zip(eng, guz)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(eng)


def main(out_dir: Path = DEFAULT_OUT_DIR) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flores200_") as tmp:
        tmp_dir = Path(tmp)
        archive = tmp_dir / "flores200_dataset.tar.gz"
        sha256 = _download(FLORES_URL, archive)
        extracted = _extract_needed(archive, tmp_dir)
        counts = {
            "guz_dev.tsv": _write_tsv(extracted["eng_dev"], extracted["guz_dev"],
                                      out_dir / "guz_dev.tsv"),
            "guz_devtest.tsv": _write_tsv(extracted["eng_devtest"],
                                          extracted["guz_devtest"],
                                          out_dir / "guz_devtest.tsv"),
        }
    for name, expected in EXPECTED_COUNTS.items():
        got = counts[name]
        if got != expected:
            print(f"fetch_flores: ERROR expected {expected} pairs in {name}, "
                  f"got {got}", file=sys.stderr)
            sys.exit(1)
    print(f"fetch_flores: archive SHA256: {sha256}")
    for name in EXPECTED_COUNTS:
        print(f"fetch_flores: {name}: {counts[name]} pairs -> {out_dir / name}")
    print("fetch_flores: done. Reminder: guz_devtest.tsv is EVALUATION-ONLY "
          "(never used for training).")
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    sys.exit(main(out))
