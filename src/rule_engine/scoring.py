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
    rule_clean_score: float = 0.0
    matches: List[ScoredMatch] = field(default_factory=list)
    dictionary_version: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_score": round(float(self.rule_score), 4),
            "rule_hate_score": round(float(self.rule_hate_score), 4),
            "rule_offensive_score": round(float(self.rule_offensive_score), 4),
            "rule_clean_score": round(float(self.rule_clean_score), 4),
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
        # Clamp score to [0,1] to avoid numerical drift
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        scored.append(ScoredMatch(match=match, context_factor=factor, score=score, signals=signals))

    per_label = aggregate_label_scores(scored, cfg.aggregation_cap)

    hate_score = float(per_label.get("HATE", 0.0))
    off_score = float(per_label.get("OFFENSIVE", 0.0))
    if hate_score < 0.0:
        hate_score = 0.0
    if hate_score > 1.0:
        hate_score = 1.0
    if off_score < 0.0:
        off_score = 0.0
    if off_score > 1.0:
        off_score = 1.0

    # overall score: saturating sum over de-duplicated spans (best label wins)
    per_span: Dict[Tuple[int, int], float] = {}
    for item in scored:
        key = (item.match.start, item.match.end)
        if item.score > per_span.get(key, 0.0):
            per_span[key] = float(item.score)
    top_spans = sorted(per_span.values(), reverse=True)[: cfg.aggregation_cap]
    product = 1.0
    for s in top_spans:
        s_clamped = min(max(float(s), 0.0), 1.0)
        product *= 1.0 - s_clamped
    overall = 1.0 - product
    if overall < 0.0:
        overall = 0.0
    if overall > 1.0:
        overall = 1.0

    # Toxic evidence derived from actual matches/labels (no hardcoded 0.9/0.6)
    s_toxic = 1.0 - (1.0 - hate_score) * (1.0 - off_score)
    if s_toxic < 0.0:
        s_toxic = 0.0
    if s_toxic > 1.0:
        s_toxic = 1.0

    # Derive CLEAN from evidence certainty: reduce clean when there is
    # strong/verified/quoted-context evidence. Factors come from matches.
    rule_clean_score = 1.0 - s_toxic
    if scored:
        # Average match confidence weighted by match strength
        total_w = 0.0
        w_conf = 0.0
        for item in scored:
            w = float(item.score)
            if w < 0.0:
                w = 0.0
            total_w += w
            w_conf += w * float(getattr(item.match, "entry_confidence", 0.5))
        if total_w > 0.0:
            avg_conf = w_conf / total_w
        else:
            avg_conf = 0.0
        # If evidence exists but confidence/context is weak (e.g. negated/quoted),
        # keep more uncertainty -> higher clean probability.
        neg_q = 0.0
        for item in scored:
            if "negated" in item.signals:
                neg_q += 0.1
            if "quoted" in item.signals:
                neg_q += 0.05
            if item.match.context_required and "context_required" in item.signals:
                neg_q += 0.05
        neg_q = min(neg_q, 0.25)
        # High certainty toxic evidence -> push clean down; weak/uncertain -> keep higher
        rule_clean_score = (1.0 - s_toxic) * (1.0 - 0.5 * avg_conf) + (1.0 - s_toxic) * neg_q
        rule_clean_score = rule_clean_score + (1.0 - s_toxic) * 0.0  # placeholder for clarity
    # Ensure clean is complementary in a reasonable range and does not ignore evidence
    if rule_clean_score < 0.0:
        rule_clean_score = 0.0
    if rule_clean_score > 1.0:
        rule_clean_score = 1.0
    # If there is strong toxic evidence, clean should not dominate
    if s_toxic >= 0.5 and rule_clean_score > (1.0 - s_toxic):
        rule_clean_score = 1.0 - s_toxic

    return RuleResult(
        rule_score=overall,
        rule_hate_score=hate_score,
        rule_offensive_score=off_score,
        rule_clean_score=rule_clean_score,
        matches=scored,
        dictionary_version=dictionary_version,
        meta={
            "n_matches": len(scored),
            "n_spans": len(per_span),
            "labels": sorted(per_label),
            "s_toxic": round(float(s_toxic), 4),
        },
    )
