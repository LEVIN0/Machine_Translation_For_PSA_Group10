#!/usr/bin/env python3
"""Translate text / run the PSA demo (SPEC_WEEK3.md §2.9) — the Week 3
success criterion.

Run from the project root:
    python scripts/translate.py --demo
    python scripts/translate.py --checkpoint runs/ft_nllb_base/checkpoint-best \
        --text "Wash your hands with soap." --src eng --tgt swa
    python scripts/translate.py --checkpoint ... --interactive

With --demo and no --checkpoint, the newest runs/*/checkpoint-best is
auto-discovered.
"""

import argparse
import sys
import textwrap
from pathlib import Path

# Make the project root importable when run as a script from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import LANGS  # noqa: E402
from training.inference import DEMO_PSAS  # noqa: E402


def discover_checkpoint(runs_root: Path = Path("runs")) -> Path | None:
    """Newest runs/*/checkpoint-best by modification time."""
    if not runs_root.is_dir():
        return None
    candidates = [p for p in runs_root.glob("*/checkpoint-best") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def print_demo_table(rows: list[dict]) -> None:
    """rows: {domain, src, source, tgt, translation}; clean tabular output."""
    w_src, w_tgt = 46, 46
    header = (f"{'#':<3}{'Domain':<13}{'Dir':<9}"
              f"{'Source':<{w_src + 2}}{'Translation':<{w_tgt}}")
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, 1):
        direction = f"{row['src']}->{row['tgt']}"
        src_lines = _wrap(row["source"], w_src)
        tgt_lines = _wrap(row["translation"], w_tgt)
        for j in range(max(len(src_lines), len(tgt_lines))):
            s = src_lines[j] if j < len(src_lines) else ""
            t = tgt_lines[j] if j < len(tgt_lines) else ""
            if j == 0:
                print(f"{i:<3}{row['domain']:<13}{direction:<9}"
                      f"{s:<{w_src + 2}}{t:<{w_tgt}}")
            else:
                print(f"{'':<3}{'':<13}{'':<9}{s:<{w_src + 2}}{t:<{w_tgt}}")
        print("-" * len(header))


def run_demo(translator) -> None:
    rows = []
    for psa in DEMO_PSAS:
        src = psa["src"]
        # eng<->swa plus ->guz for every demo line (SPEC_WEEK3.md §2.9).
        targets = (["swa", "guz"] if src == "eng" else ["eng", "guz"])
        for tgt in targets:
            try:
                translation = translator.translate([psa["text"]],
                                                   src=src, tgt=tgt)[0]
            except Exception as exc:  # demo must never die on one line
                translation = f"<error: {exc}>"
            rows.append({"domain": psa["domain"], "src": src, "tgt": tgt,
                         "source": psa["text"], "translation": translation})
    print_demo_table(rows)


def run_interactive(translator, src: str, tgt: str,
                    max_length: int, num_beams: int) -> None:
    print(f"interactive mode ({src}->{tgt}). Empty line or Ctrl-D to quit.")
    while True:
        try:
            text = input(f"{src}> ").strip()
        except EOFError:
            break
        if not text:
            break
        print(f"{tgt}> " + translator.translate(
            [text], src=src, tgt=tgt,
            max_length=max_length, num_beams=num_beams)[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Translate with a fine-tuned checkpoint, or run the "
                    "PSA demo table.")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint dir/hub id; with --demo it is "
                             "auto-discovered from newest runs/*/checkpoint-best")
    parser.add_argument("--text", default=None, help="text to translate")
    parser.add_argument("--src", default="eng", choices=sorted(LANGS))
    parser.add_argument("--tgt", default="swa", choices=sorted(LANGS))
    parser.add_argument("--demo", action="store_true",
                        help="translate the 8 demo PSAs into all applicable "
                             "targets and print a table")
    parser.add_argument("--interactive", action="store_true",
                        help="read-translate loop on stdin")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args(argv)

    if not (args.demo or args.interactive or args.text):
        parser.error("nothing to do: pass --demo, --interactive or --text")

    ckpt = args.checkpoint
    if ckpt is None:
        found = discover_checkpoint()
        if found is None:
            parser.error("no --checkpoint given and no runs/*/checkpoint-best "
                         "found — train a model first (scripts/run_training.py; "
                         "see docs/week3_kinesis_guide.md)")
        ckpt = str(found)
        print(f"[translate] auto-discovered checkpoint: {ckpt}")

    from training.inference import MTTranslator  # noqa: lazy (torch inside)

    translator = MTTranslator(ckpt)
    print(f"[translate] model: {translator.model_key} "
          f"({translator.family}) on {translator.device}")

    if args.demo:
        print()
        run_demo(translator)
        print()
    if args.text:
        out = translator.translate([args.text], src=args.src, tgt=args.tgt,
                                   max_length=args.max_length,
                                   num_beams=args.num_beams)[0]
        if not args.demo:
            print(f"{args.src}> {args.text}")
            print(f"{args.tgt}> {out}")
    if args.interactive:
        run_interactive(translator, args.src, args.tgt,
                        args.max_length, args.num_beams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
