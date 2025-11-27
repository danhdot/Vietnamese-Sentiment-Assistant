"""Utility helpers for lightweight Vietnamese text normalization."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_TEENCODE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("hok", "không"),
    ("hong", "không"),
    ("ko", "không"),
    ("k", "không"),
    ("hem", "không"),
    ("kg", "không"),
    ("hok biet", "không biết"),
    ("bít", "biết"),
    ("đc", "được"),
    ("dc", "được"),
    ("okela", "ổn"),
    ("oge", "ổn"),
    ("zui", "vui"),
    ("qa", "quá"),
    ("wa", "quá"),
    ("wá", "quá"),
    ("dep", "đẹp"),
)

_MULTI_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\u200b\ufeff]")


@lru_cache(maxsize=2048)
def strip_accents(text: str) -> str:
    """Return a lower-cased string without diacritics for lexicon lookups."""

    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def normalize_text(text: str) -> str:
    """Lightweight normalization that preserves user intent."""

    lowered = text.strip()
    lowered = _PUNCT_RE.sub("", lowered)
    lowered = _MULTI_SPACE_RE.sub(" ", lowered)
    lowered = lowered.lower()
    for needle, replacement in _TEENCODE_REPLACEMENTS:
        lowered = lowered.replace(needle, replacement)
    return lowered
