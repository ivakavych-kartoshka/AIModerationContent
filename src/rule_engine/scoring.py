"""Lexical label + rule scoring (section 39).

For every match the score is

    match_score = severity x entry_confidence x match_factor x context_factor

where ``context_factor`` accounts for negation and quotation signals around the
span (a slur inside quotes or after "không" contributes less). Matches of one
label are aggregated with the saturating rule

    rule_label_score = 1 - product(1 - match_score_i)

over at most ``aggregation_cap`` matches per label, after de-duplication. Per
section 39 this is a heuristic score, **not** a calibrated probability; the
engine exposes ``rule_hate_score`` and ``rule_offensive_score`` separately so
the fusion step can combine them per label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ScoringConfig
from .matcher import Match
from .normalize import NormalizedText

#: Tokens that flip the polarity of a nearby slur.
NEGATION_CUES = (
    "không",
    "khong",
    "ko",
    "k",
    "chẳng",
    "chang",
    "chả",
    "cha",
    "đâu có",
    "đéo",
    "phải chi",
    "nào phải",
)
QUOTE_CHARS = ('"', "'", "“", "”", "‘", "’", "«", "»")
NEGATION_WINDOW_TOKENS = 6


@dataclass
class ScoredMatch:
    match: Match
    context_factor: float
    score: float
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        data = self.match.to_dict()
        data["context_factor"] = round(self.context_factor, 4)
        data["signals"] = list(self.signals)
        data["score"] = round(self.score, 4)
        return data


@dataclass
class RuleResult:
    """Everything the engine reports for one comment (section 39 schema)."""

    rule_score: float
    rule_hate_score: float
    rule_offensive_score: float
    matches: List[ScoredMatch]
    dictionary_version: str
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_score": round(self.rule_score, 4),
            "rule_hate_score": round(self.rule_hate_score, 4),
            "rule_offensive_score": round(self.rule_offensive_score, 4),
            "matches": [scored.to_dict() for scored in self.matches],
            "dictionary_version": self.dictionary_version,
            "meta": self.meta,
        }


def _inside_quotes(text: str, position: int) -> Optional[str]:
    for quote in QUOTE_CHARS:
        if text[:position].count(quote) % 2 == 1:
            return quote
    return None


def _is_negated(text: str, start: int, window: int = NEGATION_WINDOW_TOKENS) -> bool:
    prefix = text[:start].split()
    tail = " ".join(prefix[-window:]).lower()
    return any(cue in tail for cue in NEGATION_CUES)


def compute_context_factor(
    normalized: NormalizedText, match: Match, cfg: ScoringConfig
) -> Tuple[float, List[str]]:
    """Return the context multiplier and the signals that produced it."""
    factor = 1.0
    signals: List[str] = []

    normalized_start = _raw_to_normalized(normalized, match.start)
    if normalized_start is None:
        if match.context_required:
            factor *= cfg.context_required_factor
            signals.append("context_required")
        return factor, signals

    quote = _inside_quotes(normalized.normalized_text, normalized_start)
    if quote is not None:
        factor *= cfg.quote_factor
        signals.append("quoted")

    if _is_negated(normalized.normalized_text, normalized_start):
        factor *= cfg.negation_factor
        signals.append("negated")

    if match.context_required and "context_required" not in signals:
        factor *= cfg.context_required_factor
        signals.append("context_required")
    return factor, signals


def _raw_to_normalized(normalized: NormalizedText, raw_position: int) -> Optional[int]:
    """Binary-search the inverse of the offset map."""
    low, high = 0, len(normalized.index_map) - 1
    if high < 0:
        return None
    while low < high:
        mid = (low + high) // 2
        if normalized.index_map[mid] < raw_position:
            low = mid + 1
        else:
            high = mid
    return low if normalized.index_map[low] == raw_position else None


def aggregate_label_scores(matches: Sequence[ScoredMatch], cap: int) -> Dict[str, float]:
    """Per-label saturating sum ``1 - product(1 - s_i)`` over the top matches."""
    by_label: Dict[str, List[float]] = {}
    for scored in matches:
        by_label.setdefault(scored.match.label, []).append(scored.score)
    result: Dict[str, float] = {}
    for label, scores in by_label.items():
        top = sorted(scores, reverse=True)[:cap]
        product = 1.0
        for score in top:
            product *= 1.0 - min(max(score, 0.0), 1.0)
        result[label] = 1.0 - product
    return result


def score_matches(
    normalized: NormalizedText,
    matches: Sequence[Match],
    cfg: ScoringConfig,
    dictionary_version: str,
) -> RuleResult:
    """Add context factors, aggregate per label and build the result object."""
    scored: List[ScoredMatch] = []
    for match in matches:
        factor, signals = compute_context_factor(normalized, match, cfg)
        score = match.base_score * factor
        scored.append(ScoredMatch(match=match, context_factor=factor, score=score, signals=signals))

    per_label = aggregate_label_scores(scored, cfg.aggregation_cap)

    # overall score: saturating sum over de-duplicated spans (best label wins)
    per_span: Dict[Tuple[int, int], float] = {}
    for item in scored:
        key = (item.match.start, item.match.end)
        per_span[key] = max(per_span.get(key, 0.0), item.score)
    top_spans = sorted(per_span.values(), reverse=True)[: cfg.aggregation_cap]
    product = 1.0
    for score in top_spans:
        product *= 1.0 - min(max(score, 0.0), 1.0)
    overall = 1.0 - product

    return RuleResult(
        rule_score=overall,
        rule_hate_score=per_label.get("HATE", 0.0),
        rule_offensive_score=per_label.get("OFFENSIVE", 0.0),
        matches=scored,
        dictionary_version=dictionary_version,
        meta={
            "n_matches": len(scored),
            "n_spans": len(per_span),
            "labels": sorted(per_label),
        },
    )
