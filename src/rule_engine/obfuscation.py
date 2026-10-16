"""Anti-obfuscation matching views (section 38).

Writing tricks the engine fights here: inserting spaces/punctuation between
characters (``n g u``), dropping Vietnamese diacritics (``con cho``), leet
substitutions (``c0n ch0``), stretching letters (``nguuu``) and deliberate
word splitting.

Two auxiliary views are built next to the faithful ``base`` view:

* ``plain``  - accent-less + leet + repeat-collapsed; separators are kept, so
  word boundaries still work;
* ``compact`` - ``plain`` with spaces/punctuation removed, used only for
  spaced-out obfuscation (the matcher re-checks that the original span really
  contained separators and was flanked by them).

Every view carries an ``index_map`` mapping its characters back to the *base*
normalized text, which the engine then maps onto the raw comment. The base view
itself is never modified for the AI model: views are matching-only, exactly as
section 37 requires.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ObfuscationConfig
from .normalize import NormalizedText

#: Characters NFD cannot decompose (đ/Đ) mapped explicitly to their base letter.
ACCENTLESS_SPECIAL = {"đ": "d", "Đ": "d"}

#: Common leet substitutions (section 38: letter/number/symbol swaps).
#: Punctuation that also acts as a word separator (``!``, ``|``, ``+``) is
#: deliberately excluded: mapping it to a letter would merge it with the next
#: token and break the word-boundary check used by the ``plain`` view.
LEET_MAP = {
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
}


@dataclass
class View:
    """A matching view plus its map back to base-normalized characters."""

    name: str
    text: str
    index_map: List[int]

    def base_span(self, start: int, end: int) -> Tuple[int, int]:
        if start < 0 or end > len(self.index_map) or start >= end:
            return (0, 0)
        return (self.index_map[start], self.index_map[end - 1] + 1)


# --------------------------------------------------------------------------- #
# Character transforms (shared by views and head-word preparation)
# --------------------------------------------------------------------------- #
def _accentless_char(char: str) -> str:
    if char in ACCENTLESS_SPECIAL:
        return ACCENTLESS_SPECIAL[char]
    decomposed = unicodedata.normalize("NFD", char)
    base = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return base or char


def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics (auxiliary signal only, section 37)."""
    return "".join(_accentless_char(char) for char in text)


def to_leet(text: str) -> str:
    return "".join(LEET_MAP.get(char, char) for char in text)


def collapse_repeats(text: str) -> str:
    out: List[str] = []
    for char in text:
        if out and out[-1] == char:
            continue
        out.append(char)
    return "".join(out)


def is_separator(char: str) -> bool:
    return char.isspace() or unicodedata.category(char)[0] in ("P", "S")


def strip_separators(text: str) -> str:
    return "".join(char for char in text if not is_separator(char))


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def _apply_char_transform(
    text: str, index_map: Sequence[int], transform
) -> Tuple[str, List[int]]:
    out_chars: List[str] = []
    out_map: List[int] = []
    for index, char in enumerate(text):
        piece = transform(char)
        if not piece:
            continue
        for sub in piece:
            out_chars.append(sub)
            out_map.append(index_map[index])
    return "".join(out_chars), out_map


def _collapse_view(text: str, index_map: Sequence[int]) -> Tuple[str, List[int]]:
    out_chars: List[str] = []
    out_map: List[int] = []
    for index, char in enumerate(text):
        if out_chars and out_chars[-1] == char:
            continue
        out_chars.append(char)
        out_map.append(index_map[index])
    return "".join(out_chars), out_map


def build_views(base: NormalizedText, cfg: ObfuscationConfig | None = None) -> Dict[str, View]:
    """Build the base / plain / compact matching views for one normalized text."""
    cfg = cfg or ObfuscationConfig()
    views: Dict[str, View] = {"base": View("base", base.normalized_text, list(range(len(base.normalized_text))))}
    if not cfg.enabled:
        return views

    text = base.normalized_text
    index_map: List[int] = list(range(len(text)))

    if cfg.accentless:
        text, index_map = _apply_char_transform(text, index_map, _accentless_char)
    if cfg.leet:
        text, index_map = _apply_char_transform(text, index_map, lambda ch: LEET_MAP.get(ch, ch))
    if cfg.collapse_repeats:
        text, index_map = _collapse_view(text, index_map)
    if cfg.max_variant_chars and len(text) > cfg.max_variant_chars:
        text, index_map = text[: cfg.max_variant_chars], index_map[: cfg.max_variant_chars]

    views["plain"] = View("plain", text, index_map)

    if cfg.separator_stripping:
        compact_text, compact_map = _apply_char_transform(
            text, index_map, lambda ch: "" if is_separator(ch) else ch
        )
        views["compact"] = View("compact", compact_text, compact_map)
    return views


# --------------------------------------------------------------------------- #
# Head-word preparation and transformation classification
# --------------------------------------------------------------------------- #
def plain_form(term: str, cfg: ObfuscationConfig | None = None) -> str:
    """Transform a dictionary term the way the ``plain`` view transforms text."""
    cfg = cfg or ObfuscationConfig()
    text = term
    if cfg.accentless:
        text = strip_accents(text)
    if cfg.leet:
        text = to_leet(text)
    if cfg.collapse_repeats:
        text = collapse_repeats(text)
    return text


def compact_form(term: str, cfg: ObfuscationConfig | None = None) -> str:
    """Transform a dictionary term the way the ``compact`` view transforms text."""
    cfg = cfg or ObfuscationConfig()
    text = plain_form(term, cfg)
    return strip_separators(text) if cfg.separator_stripping else text


#: Single-step transform -> match_type name.
_SINGLE_TRANSFORM_TYPE = {
    ("accentless",): "accentless_alias",
    ("leet",): "leet",
    ("collapse",): "repeat_collapsed",
    ("separators",): "separator_inserted",
}


def _pipeline(cfg: ObfuscationConfig, compact: bool) -> List[Tuple[str, object]]:
    steps: List[Tuple[str, object]] = []
    if cfg.accentless:
        steps.append(("accentless", strip_accents))
    if cfg.leet:
        steps.append(("leet", to_leet))
    if cfg.collapse_repeats:
        steps.append(("collapse", collapse_repeats))
    if compact and cfg.separator_stripping:
        steps.append(("separators", strip_separators))
    return steps


def _reduction_steps(text: str, target: str, cfg: ObfuscationConfig, compact: bool) -> List[str]:
    """Transforms that reduce ``text`` towards the view form ``target``."""
    steps: List[str] = []
    current = text
    for name, transform in _pipeline(cfg, compact):
        if current == target:
            break
        candidate = transform(current)  # type: ignore[operator]
        if candidate != current:
            steps.append(name)
            current = candidate
    return steps if current == target else []


def classify_transformation(
    base_span: str, term: str, cfg: ObfuscationConfig, compact: bool = False
) -> Optional[Tuple[str, str]]:
    """Return ``(match_type, transformation_trace)`` or ``None`` when unexplainable.

    ``target`` is the form the corresponding view compares against (plain or
    compact); the trace lists the pipeline steps needed to move the observed
    span onto it. When the span already equals the view form, the steps that
    reduce the dictionary term to it explain the difference (e.g. dropped
    diacritics or a collapsed repeated run).
    """
    if base_span == term:
        return ("exact", "none")

    target = compact_form(term, cfg) if compact else plain_form(term, cfg)
    steps = _reduction_steps(base_span, target, cfg, compact)
    if not steps:
        if base_span != target:
            return None
        steps = _reduction_steps(term, target, cfg, compact)

    # a term with diacritics matched by an accent-less span: also accentless
    if (
        cfg.accentless
        and "accentless" not in steps
        and strip_accents(term) != term
        and strip_accents(base_span) == base_span
    ):
        steps.insert(0, "accentless")

    if not steps:
        return None
    match_type = _SINGLE_TRANSFORM_TYPE.get(tuple(steps), "obfuscated_combined")
    return match_type, "+".join(steps)
