"""Parallel-corpus importers (TICO-19, Tatoeba)."""

from .tico19 import import_tico19, parse_tmx
from .tatoeba import import_tatoeba

__all__ = ["import_tico19", "parse_tmx", "import_tatoeba"]
