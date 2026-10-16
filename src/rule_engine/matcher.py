"""Dictionary matching with exact, alias, obfuscation and fuzzy strategies.

Section 38 asks for several matching levels and for every match to record its
``match_type``, transformation trace, offsets and confidence. This module
implements them on top of the views built by :mod:`src.rule_engine.obfuscation`:

===============  ==========================================================
``exact``         dictionary term found verbatim in the normalized text
``accentless_alias``  diacritics dropped (``con cho``)
``leet``          digits/symbols swapped for letters (``c0n``)
``repeat_collapsed``  stretched letters (``nguuu``)
``separator_inserted``  spaces/punctuation between characters (``n g u``)
``obfuscated_combined``  several transformations at once
``fuzzy``         bounded edit distance on long single-token terms
===============  ==========================================================

Section 38 also warns against arbitrary substring hits and double counting:
the compact view only fires when the original span really contained separators
and was flanked by separators, and matches are de-duplicated per span and label
keeping the strongest score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config import MatcherConfig, ObfuscationConfig
from .lexical_classifier import Lexicon, LexiconEntry
from .normalize import NormalizedText
from .obfuscation import (
    View,
    classify_transformation,
    compact_form,
    is_separator,
    plain_form,
)

WORD_CHAR = r"\w"


@dataclass
class Match:
    """One dictionary hit with raw offsets and full provenance."""

    entry_id: str
    term: str
    label: str
    match_type: str
    match_factor: float
    severity: float
    entry_confidence: float
    start: int
    end: int
    normalized_span: str
    raw_span: str
    transformation_trace: str
    context_required: bool = False
    ambiguous_alias: bool = False

    @property
    def base_score(self) -> float:
        return self.severity * self.entry_confidence * self.match_factor

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "term": self.term,
            "label": self.label,
            "match_type": self.match_type,
            "transformation_trace": self.transformation_trace,
            "match_factor": round(self.match_factor, 4),
            "severity": round(self.severity, 4),
            "entry_confidence": round(self.entry_confidence, 4),
            "start": self.start,
            "end": self.end,
            "raw_span": self.raw_span,
            "normalized_span": self.normalized_span,
            "context_required": self.context_required,
            "ambiguous_alias": self.ambiguous_alias,
        }


def _alternation(forms: Iterable[str]) -> Optional[re.Pattern[str]]:
    ordered = sorted({form for form in forms if form}, key=len, reverse=True)
    if not ordered:
        return None
    return re.compile("|".join(re.escape(form) for form in ordered))


def bounded_levenshtein(left: str, right: str, max_distance: int) -> int:
    """Levenshtein distance capped at ``max_distance`` (returns it + 1 when over)."""
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


class Matcher:
    """Pre-compiled dictionary matcher."""

    def __init__(
        self,
        lexicon: Lexicon,
        matcher_cfg: Optional[MatcherConfig] = None,
        obfuscation_cfg: Optional[ObfuscationConfig] = None,
        match_factors: Optional[Dict[str, float]] = None,
        severity_by_label: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lexicon = lexicon
        self.cfg = matcher_cfg or MatcherConfig()
        self.obf = obfuscation_cfg or ObfuscationConfig()
        self.factors = match_factors or {}
        self.severity_by_label = severity_by_label or {"HATE": 0.9, "OFFENSIVE": 0.6}
        self._build_indexes()

    # ------------------------------------------------------------------ #
    # indexes
    # ------------------------------------------------------------------ #
    def _eligible(self, entry: LexiconEntry) -> bool:
        return len(entry.term) >= self.cfg.min_term_len

    def _build_indexes(self) -> None:
        entries = [entry for entry in self.lexicon.entries if self._eligible(entry)]

        self.exact_re = _alternation(entry.term for entry in entries)

        plain_index: Dict[str, List[LexiconEntry]] = {}
        for entry in entries:
            form = plain_form(entry.term, self.obf)
            if len(form) < self.cfg.min_term_len:
                continue  # e.g. "ccc" collapsing to a single "c"
            plain_index.setdefault(form, []).append(entry)
        self.plain_index = plain_index
        self.plain_re = _alternation(plain_index) if self.obf.enabled else None

        compact_index: Dict[str, List[LexiconEntry]] = {}
        if self.obf.enabled and self.obf.separator_stripping:
            for entry in entries:
                if len(entry.term) < self.obf.min_compact_term_len:
                    continue
                form = compact_form(entry.term, self.obf)
                if len(form) < self.obf.min_compact_term_len:
                    continue
                compact_index.setdefault(form, []).append(entry)
        self.compact_index = compact_index
        self.compact_re = _alternation(compact_index)

        fuzzy_index: Dict[Tuple[str, int], List[LexiconEntry]] = {}
        if self.cfg.fuzzy_enabled:
            for entry in entries:
                if " " in entry.term or len(entry.term) < self.cfg.fuzzy_min_term_len:
                    continue
                fuzzy_index.setdefault((entry.term[0], len(entry.term)), []).append(entry)
        self.fuzzy_index = fuzzy_index
        self.token_re = re.compile(r"\w+", re.UNICODE)

    # ------------------------------------------------------------------ #
    # strategies
    # ------------------------------------------------------------------ #
    def _make_match(
        self,
        nt: NormalizedText,
        base: View,
        entry: LexiconEntry,
        label: str,
        match_type: str,
        trace: str,
        base_start: int,
        base_end: int,
        ambiguous: bool = False,
    ) -> Match:
        raw_start, raw_end = base.index_map[base_start], base.index_map[base_end - 1] + 1
        return Match(
            entry_id=entry.entry_id,
            term=entry.term,
            label=label,
            match_type=match_type,
            match_factor=float(self.factors.get(match_type, self.factors.get("obfuscated_combined", 0.6))),
            severity=float(self.severity_by_label.get(label, 0.5)),
            entry_confidence=float(entry.confidence),
            start=raw_start,
            end=raw_end,
            normalized_span=base.text[base_start:base_end],
            raw_span=nt.raw_text[raw_start:raw_end],
            transformation_trace=trace,
            context_required=bool(entry.context_required),
            ambiguous_alias=ambiguous,
        )

    def _match_exact(self, nt: NormalizedText, base: View) -> List[Match]:
        if self.exact_re is None:
            return []
        pattern = re.compile(
            r"(?<!\w)(?:" + self.exact_re.pattern + r")(?!\w)", re.UNICODE
        )
        matches: List[Match] = []
        for found in pattern.finditer(base.text):
            entry = self.lexicon.by_term.get(found.group(0))
            if entry is None:
                continue
            for label in entry.labels:
                matches.append(
                    self._make_match(nt, base, entry, label, "exact", "none", found.start(), found.end())
                )
        return matches

    def _match_plain(self, nt: NormalizedText, base: View, plain: View) -> List[Match]:
        if self.plain_re is None:
            return []
        pattern = re.compile(r"(?<!\w)(?:" + self.plain_re.pattern + r")(?!\w)", re.UNICODE)
        matches: List[Match] = []
        for found in pattern.finditer(plain.text):
            entries = self.plain_index.get(found.group(0), [])
            base_start, base_end = plain.index_map[found.start()], plain.index_map[found.end() - 1] + 1
            base_span = base.text[base_start:base_end]
            for entry in entries:
                if base_span == entry.term:
                    continue  # already produced by the exact strategy
                classified = classify_transformation(base_span, entry.term, self.obf)
                if classified is None:
                    continue
                match_type, trace = classified
                ambiguous = len(entries) > 1
                for label in entry.labels:
                    matches.append(
                        self._make_match(
                            nt, base, entry, label, match_type, trace, base_start, base_end, ambiguous
                        )
                    )
        return matches

    def _is_flanked_by_separator(self, base_text: str, start: int, end: int) -> bool:
        if start > 0 and not is_separator(base_text[start - 1]):
            return False
        if end < len(base_text) and not is_separator(base_text[end]):
            return False
        return True

    def _match_compact(self, nt: NormalizedText, base: View, compact: View) -> List[Match]:
        if self.compact_re is None:
            return []
        matches: List[Match] = []
        for found in self.compact_re.finditer(compact.text):
            entries = self.compact_index.get(found.group(0), [])
            base_start, base_end = compact.index_map[found.start()], compact.index_map[found.end() - 1] + 1
            base_span = base.text[base_start:base_end]
            if not any(is_separator(char) for char in base_span):
                continue  # contiguous text - the exact/plain strategies own it
            if not self._is_flanked_by_separator(base.text, base_start, base_end):
                continue
            for entry in entries:
                classified = classify_transformation(base_span, entry.term, self.obf, compact=True)
                if classified is None or "separators" not in classified[1]:
                    continue
                match_type, trace = classified
                ambiguous = len(entries) > 1
                for label in entry.labels:
                    matches.append(
                        self._make_match(
                            nt, base, entry, label, match_type, trace, base_start, base_end, ambiguous
                        )
                    )
        return matches

    def _match_fuzzy(self, nt: NormalizedText, base: View) -> List[Match]:
        if not self.cfg.fuzzy_enabled or not self.fuzzy_index:
            return []
        matches: List[Match] = []
        max_distance = self.cfg.fuzzy_max_distance
        for token in self.token_re.finditer(base.text):
            word = token.group(0)
            if len(word) < self.cfg.fuzzy_min_term_len:
                continue
            candidates: List[LexiconEntry] = []
            for length in (len(word) - 1, len(word), len(word) + 1):
                candidates.extend(self.fuzzy_index.get((word[0], length), []))
            for entry in candidates:
                if word == entry.term:
                    continue
                if bounded_levenshtein(word, entry.term, max_distance) > max_distance:
                    continue
                for label in entry.labels:
                    matches.append(
                        self._make_match(
                            nt, base, entry, label, "fuzzy", "edit_distance", token.start(), token.end()
                        )
                    )
        return matches

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def match(self, nt: NormalizedText, views: Dict[str, View]) -> List[Match]:
        """Run every strategy and return the de-duplicated, capped match list."""
        base = views["base"]
        matches: List[Match] = []
        matches.extend(self._match_exact(nt, base))
        if self.obf.enabled and "plain" in views:
            matches.extend(self._match_plain(nt, base, views["plain"]))
        if self.obf.enabled and "compact" in views:
            matches.extend(self._match_compact(nt, base, views["compact"]))
        matches.extend(self._match_fuzzy(nt, base))

        best: Dict[Tuple[int, int, str], Match] = {}
        for match in matches:
            key = (match.start, match.end, match.label)
            current = best.get(key)
            if current is None or match.base_score > current.base_score:
                best[key] = match
        ordered = sorted(best.values(), key=lambda m: (-m.base_score, m.start, m.end))
        return ordered[: self.cfg.max_matches]
