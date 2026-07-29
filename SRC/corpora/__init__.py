"""Parallel-corpus importers (TICO-19, Tatoeba) + team-written manual PSAs
+ lecturer gold dataset."""

from .tico19 import import_tico19, parse_tmx
from .tatoeba import import_tatoeba
from .manual import import_manual
from .lecturer import load_lecturer

__all__ = ["import_tico19", "parse_tmx", "import_tatoeba", "import_manual",
           "load_lecturer"]
