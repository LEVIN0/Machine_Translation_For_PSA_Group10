"""TICO-19 corpus importer (Translation Initiative for COVID-19).

TICO-19 provides human-translated English<->Kiswahili crisis/health segments
(license CC BY 4.0). Two access modes:

- split="tmx_all": parse the full translation-memory TMX
  (https://tico-19.github.io/data/TM/all.en-sw.tmx.zip).
- split="dev"|"test": read the official benchmark text files
  (TICO19.{split}.en / .sw line pairs) if they were extracted under
  EXTERNAL_DIR/tico19/bench/ (from https://tico-19.github.io/data/tico19-testset.zip).

SPLIT POLICY (also stated in README and the Week 1 report): the dev/test split
reservation is a Week 2 decision. Never train on the evaluation split.
"""

import zipfile
from pathlib import Path
from xml.etree.ElementTree import iterparse

import requests

from ..config import EXTERNAL_DIR, REQUEST_TIMEOUT, USER_AGENT
from ..schema import new_record

TMX_URL = "https://tico-19.github.io/data/TM/all.en-sw.tmx.zip"
TESTSET_URL = "https://tico-19.github.io/data/tico19-testset.zip"

_SOURCE = "TICO-19 (Translation Initiative for COVID-19)"
_URL = "https://tico-19.github.io/"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _local_name(tag):
    """Strip any XML namespace from a tag, returning its local name."""
    return tag.rsplit("}", 1)[-1]


def download_tmx(dest_dir=None):
    """Stream-download the TMX zip, extract it, and return the .tmx path.

    Skips the download if a .tmx is already extracted under `dest_dir`.
    TICO-19 is licensed CC BY 4.0 — attribution is required.
    """
    dest_dir = Path(dest_dir or EXTERNAL_DIR / "tico19")
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest_dir.rglob("*.tmx"))
    if existing:
        print(f"[tico19] Using existing TMX: {existing[0]}")
        return existing[0]

    zip_path = dest_dir / "all.en-sw.tmx.zip"
    print(f"[tico19] Downloading {TMX_URL} ...")
    resp = requests.get(
        TMX_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()
    with open(zip_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    tmx_files = sorted(dest_dir.rglob("*.tmx"))
    if not tmx_files:
        raise FileNotFoundError(f"No .tmx found after extracting {zip_path}")
    print("[tico19] Attribution reminder: TICO-19, CC BY 4.0, "
          "https://tico-19.github.io/")
    return tmx_files[0]


def parse_tmx(path):
    """Parse a TMX file into a list of (english, kiswahili) segment pairs.

    Uses ElementTree.iterparse with local-name tag matching (namespace-safe).
    For each <tu>, the <seg> texts of <tuv xml:lang="en"> and
    <tuv xml:lang="sw"> are paired; tus missing either side are skipped.
    """
    pairs = []
    context = iterparse(str(path), events=("end",))
    for _, elem in context:
        if _local_name(elem.tag) != "tu":
            continue
        segs = {}
        for tuv in elem.iter():
            if _local_name(tuv.tag) != "tuv":
                continue
            lang = tuv.get(_XML_LANG) or tuv.get("lang") or ""
            lang = lang.lower()
            if lang.startswith("en"):
                key = "en"
            elif lang.startswith("sw"):
                key = "sw"
            else:
                continue
            for child in tuv:
                if _local_name(child.tag) == "seg":
                    text = "".join(child.itertext()).strip()
                    if text:
                        segs[key] = " ".join(text.split())
                    break
        if "en" in segs and "sw" in segs:
            pairs.append((segs["en"], segs["sw"]))
        elem.clear()
    return pairs


def _read_bench_pairs(split, bench_dir):
    """Read official TICO19.{split}.en/.sw line pairs from the bench directory."""
    en_path = Path(bench_dir) / f"TICO19.{split}.en"
    sw_path = Path(bench_dir) / f"TICO19.{split}.sw"
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    sw_lines = sw_path.read_text(encoding="utf-8").splitlines()
    return list(zip(en_lines, sw_lines))


def import_tico19(split="tmx_all", max_pairs=None, tmx_path=None, bench_dir=None,
                  verbose=True):
    """Import TICO-19 records in the dataset schema.

    split="tmx_all" parses the full TMX (downloading it first unless tmx_path
    is given). split="dev"|"test" reads official benchmark line pairs from
    EXTERNAL_DIR/tico19/bench/ if present. Every record has English and
    Kiswahili filled, domain "Health", and corpus metadata (CC BY 4.0).
    """
    pairs = []
    if split in ("dev", "test"):
        bench_dir = Path(bench_dir or EXTERNAL_DIR / "tico19" / "bench")
        en_file = bench_dir / f"TICO19.{split}.en"
        sw_file = bench_dir / f"TICO19.{split}.sw"
        if en_file.exists() and sw_file.exists():
            pairs = _read_bench_pairs(split, bench_dir)
        elif verbose:
            print(f"[tico19] Official {split} files not found under {bench_dir}; "
                  f"falling back to split='tmx_all'. Download: {TESTSET_URL}")
            split = "tmx_all"

    if split == "tmx_all":
        path = Path(tmx_path) if tmx_path else download_tmx()
        pairs = parse_tmx(path)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    records = [
        new_record(
            domain="Health",
            english=en,
            kiswahili=sw,
            source=_SOURCE,
            url=_URL,
            metadata={
                "type": "corpus",
                "corpus": "TICO-19",
                "license": "CC BY 4.0",
                "split": split,
            },
        )
        for en, sw in pairs
        if en.strip() and sw.strip()
    ]
    if verbose:
        print(f"[tico19] imported {len(records)} EN-SW pairs (split={split})")
    return records
