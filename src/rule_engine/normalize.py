"""Input normalisation with an exact offset map back to the raw text (section 37).

The rule engine never destroys the original comment: :func:`normalize` returns a
:class:`NormalizedText` holding the untouched ``raw_text``, the cleaned
``normalized_text`` and an ``index_map`` that maps every character of the
normalized text to the position it came from in the raw text. That is what lets
the engine report "this span of the original comment matched".

Steps (all configurable in :class:`~src.rule_engine.config.NormalizeConfig`):

1. Unicode normalisation (NFC by default) applied per base+combining cluster so
   the offset map stays exact even for decomposed Vietnamese input.
2. Invisible / zero-width / bidi characters removed.
3. Whitespace, newline and tab runs collapsed to a single space.
4. Runs of the same character shortened to ``max_repeat_char``.
5. Lower-casing (Unicode aware, keeps Vietnamese diacritics).

Section 37 forbids deleting diacritics in the main text, so that is *not* done
here - accent-less matching lives in the separate auxiliary views built by
:mod:`src.rule_engine.obfuscation`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

from .config import NormalizeConfig

#: Zero-width / bidi / soft-hyphen characters that carry no linguistic signal.
INVISIBLE_CHARS = frozenset(
    "\u00ad"
    "\u200b\u200c\u200d"
    "\u2060"
    "\ufeff"
    "\u200e\u200f"
    "\u202a\u202b\u202c\u202d\u202e"
    "\u2066\u2067\u2068\u2069"
)


@dataclass
class NormalizedText:
    """Raw text, normalized text and the character-level offset map."""

    raw_text: str
    normalized_text: str
    index_map: List[int]
    truncated: bool = False

    def raw_span(self, start: int, end: int) -> Tuple[int, int]:
        """Map a ``[start, end)`` span of the normalized text onto the raw text."""
        if start < 0 or end > len(self.index_map) or start >= end:
            return (0, 0)
        return (self.index_map[start], self.index_map[end - 1] + 1)

    def raw_slice(self, start: int, end: int) -> str:
        raw_start, raw_end = self.raw_span(start, end)
        return self.raw_text[raw_start:raw_end]

    def normalized_slice(self, start: int, end: int) -> str:
        return self.normalized_text[start:end]


def _clusters_with_nfc(text: str, form: str) -> Tuple[str, List[int]]:
    """NFC-normalise base+combining clusters while tracking raw positions."""
    out_chars: List[str] = []
    out_map: List[int] = []
    cluster_start = 0
    cluster: List[str] = []
    for index, char in enumerate(text):
        combining = unicodedata.category(char) == "Mn"
        if cluster and not combining:
            piece = unicodedata.normalize(form, "".join(cluster))
            out_chars.append(piece)
            out_map.extend([cluster_start] * len(piece))
            cluster = [char]
            cluster_start = index
        else:
            cluster.append(char)
    if cluster:
        piece = unicodedata.normalize(form, "".join(cluster))
        out_chars.append(piece)
        out_map.extend([cluster_start] * len(piece))
    return "".join(out_chars), out_map


def _is_space(char: str) -> bool:
    return char.isspace() or unicodedata.category(char) in {"Zs", "Zl", "Zp"}


def normalize(raw_text: object, cfg: NormalizeConfig | None = None) -> NormalizedText:
    """Normalise ``raw_text`` and return it together with its offset map."""
    cfg = cfg or NormalizeConfig()
    text = "" if raw_text is None else str(raw_text)
    truncated = False
    if cfg.max_text_chars and len(text) > cfg.max_text_chars:
        text = text[: cfg.max_text_chars]
        truncated = True

    # 1. Unicode normalisation, keeping raw offsets -------------------------
    working, working_map = _clusters_with_nfc(text, cfg.unicode_form) if cfg.unicode_form else (text, list(range(len(text))))

    # 2. character-level cleaning ------------------------------------------
    out_chars: List[str] = []
    out_map: List[int] = []
    last_char = ""
    run_length = 0

    for index, char in enumerate(working):
        raw_index = working_map[index]

        if cfg.remove_invisible and char in INVISIBLE_CHARS:
            continue

        if _is_space(char):
            if out_chars and out_chars[-1] != " ":
                out_chars.append(" ")
                out_map.append(raw_index)
            last_char, run_length = " ", 1
            continue

        lowered = char.lower() if cfg.lowercase else char
        if not lowered:
            continue

        for piece in lowered:
            if cfg.max_repeat_char and piece == last_char and piece.isalnum():
                run_length += 1
                if run_length > cfg.max_repeat_char:
                    continue
            else:
                run_length = 1
            out_chars.append(piece)
            out_map.append(raw_index)
            last_char = piece

    # collapse to a single space is already handled; strip the edges --------
    while out_chars and out_chars[0] == " ":
        out_chars.pop(0)
        out_map.pop(0)
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_map.pop()

    return NormalizedText(
        raw_text=text,
        normalized_text="".join(out_chars),
        index_map=out_map,
        truncated=truncated,
    )
