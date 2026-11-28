"""Vietnamese / English language detection.

The detector is deliberately lightweight and has two layers:

1. **heuristics** (:mod:`src.langid.heuristics`) - Vietnamese diacritics, a
   Vietnamese/English word list and explicit code-mix signals.  Always runs and
   has no third-party dependency.
2. **statistical backend** - ``langid`` when it is installed (``pip install
   langid``).  Its votes are blended with the heuristics, which fixes cases the
   word list cannot know (short English sentences, unusual Vietnamese slang).

The output is one of ``vi``, ``en``, ``mix``, ``other``.  ``mix`` is not a
failure: it is the signal the router uses to keep both language pipelines
active, so an English insult inside a Vietnamese comment is still scanned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..common.logging_utils import get_logger
from .heuristics import HeuristicSignal, heuristic_signal

LOGGER = get_logger("sensitiveai.langid")

LANGUAGES = ("vi", "en", "mix", "other")


@dataclass
class DetectionConfig:
    """Thresholds for :class:`LanguageDetector`.

    ``min_confidence`` applies to the *winning* language; below it the result is
    ``other`` and the router falls back to the multilingual pipeline.
    """

    backend: str = "auto"  # auto | langid | heuristic
    min_confidence: float = 0.50
    #: both languages must carry at least this evidence for a ``mix`` verdict
    mix_floor: float = 0.12
    #: minimum word count before a ``mix`` verdict is allowed
    mix_min_tokens: int = 3
    #: weight of the heuristic signal when a statistical backend is available
    heuristic_weight: float = 0.60
    #: how many words a token must have before per-token tagging is trusted
    token_min_len: int = 2


@dataclass
class LanguageResult:
    """Document-level language verdict plus the evidence behind it."""

    lang: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    method: str = "heuristic"
    evidence: Dict[str, object] = field(default_factory=dict)

    @property
    def is_mixed(self) -> bool:
        return self.lang == "mix"

    def to_dict(self) -> Dict[str, object]:
        return {
            "lang": self.lang,
            "confidence": round(float(self.confidence), 4),
            "scores": {key: round(float(value), 4) for key, value in self.scores.items()},
            "method": self.method,
            "evidence": self.evidence,
        }


def blend_scores(heuristic: HeuristicSignal, votes: Dict[str, float], heuristic_weight: float) -> Dict[str, float]:
    """Blend heuristic scores and backend votes into ``{vi, en, other}``."""
    weight = min(1.0, max(0.0, float(heuristic_weight)))
    vi = weight * heuristic.vi_score + (1.0 - weight) * votes.get("vi", 0.0)
    en = weight * heuristic.en_score + (1.0 - weight) * votes.get("en", 0.0)
    other = (1.0 - weight) * votes.get("other", 0.0)
    total = vi + en + other
    if total > 0:
        vi, en, other = vi / total, en / total, other / total
    return {"vi": vi, "en": en, "other": other}


class LanguageDetector:
    """Hybrid Vietnamese/English detector."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()
        self._backend = self._resolve_backend(self.config.backend)

    # ------------------------------------------------------------------ #
    # backend
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_backend(name: str) -> Optional[object]:
        name = (name or "auto").lower()
        if name == "heuristic":
            return None
        try:
            import langid  # type: ignore
        except ImportError:
            if name == "langid":
                raise ImportError(
                    "DetectionConfig.backend='langid' but the 'langid' package is not "
                    "installed. Run 'pip install langid' or use backend='heuristic'."
                )
            LOGGER.info("[langid] 'langid' not installed; using heuristics only")
            return None
        return langid

    def _backend_votes(self, text: str) -> Dict[str, float]:
        if self._backend is None or not text.strip():
            return {}
        try:
            # ``rank`` returns [(lang, log_prob), ...] best first.
            ranked: Sequence[tuple[str, float]] = self._backend.rank(text)[:3]  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("[langid] backend failed: %s", exc)
            return {}
        if not ranked:
            return {}
        best = max(logprob for _, logprob in ranked)
        weights: List[tuple[str, float]] = []
        total = 0.0
        for label, logprob in ranked:
            value = math.exp(logprob - best)
            weights.append((label, value))
            total += value
        votes = {"vi": 0.0, "en": 0.0, "other": 0.0}
        for label, value in weights:
            share = value / total if total else 0.0
            if label in ("vi", "en"):
                votes[label] += share
            else:
                votes["other"] += share
        return votes

    # ------------------------------------------------------------------ #
    # detection
    # ------------------------------------------------------------------ #
    def detect(self, text: object) -> LanguageResult:
        """Detect the language of one raw comment."""
        raw = "" if text is None else str(text)
        signal = heuristic_signal(raw)
        votes = self._backend_votes(raw)
        method = "heuristic" if not votes else f"heuristic+{self.config.backend}"

        if not votes:
            scores = {"vi": signal.vi_score, "en": signal.en_score, "other": 0.0}
            total = scores["vi"] + scores["en"]
            if total > 0:
                scores["vi"] /= total
                scores["en"] /= total
                scores["other"] = 0.0
            else:
                scores["other"] = 1.0
        else:
            weight = self.config.heuristic_weight
            # ``langid`` is unreliable on 1-4 words (it often picks Spanish or
            # another Latin language).  Short comments are exactly where the
            # word lists are strongest, so trust the heuristics more.
            if 0 < signal.n_words <= 4:
                weight = max(weight, 0.80)
            scores = blend_scores(signal, votes, weight)

        lang, confidence = self._decide(signal, scores, raw)
        evidence = signal.to_dict()
        evidence["backend_votes"] = {key: round(value, 4) for key, value in votes.items()}
        return LanguageResult(
            lang=lang,
            confidence=confidence,
            scores=scores,
            method=method,
            evidence=evidence,
        )

    def _decide(
        self,
        signal: HeuristicSignal,
        scores: Dict[str, float],
        text: str,
    ) -> tuple[str, float]:
        cfg = self.config
        vi = scores.get("vi", 0.0)
        en = scores.get("en", 0.0)

        # A code-mixed comment needs (a) enough words to be meaningful, (b)
        # clear Vietnamese, and (c) a real English content word - not just an
        # English function word that happens to sit inside Vietnamese text.
        vi_strong = (
            signal.vi_word_ratio >= 0.35
            or signal.specific_ratio >= 0.05
            or vi >= 0.60
        )
        en_strong = signal.en_content_ratio > 0.0 or signal.en_word_ratio >= 0.22

        # Strong English vocabulary with no Vietnamese diacritics and almost no
        # Vietnamese words: accept English even if the backend guessed another
        # Latin language ("do you like it" -> langid says Spanish).
        if (
            signal.specific_ratio == 0.0
            and signal.toned_ratio == 0.0
            and signal.en_word_ratio >= 0.60
            and signal.vi_word_ratio <= 0.30
        ):
            return "en", max(en, min(1.0, signal.en_word_ratio))

        # Unaccented Vietnamese ("may noi the la sai roi"): the statistical
        # backend sees ASCII and often votes English.  Distinctive Vietnamese
        # vocabulary with no English content word is stronger evidence than
        # that vote, so decide before the generic comparison below.
        unaccented_vi = (
            signal.specific_ratio == 0.0
            and signal.toned_ratio == 0.0
            and signal.vi_word_ratio >= 0.40
            and signal.en_content_ratio == 0.0
            and signal.en_word_ratio < 0.35
        )
        if unaccented_vi:
            return "vi", max(vi, min(1.0, signal.vi_word_ratio + 0.10))

        if (
            signal.n_words >= cfg.mix_min_tokens
            and vi_strong
            and en_strong
            and min(vi, en) >= cfg.mix_floor
        ):
            return "mix", max(0.0, min(1.0, min(vi, en) + 0.25))

        if vi >= en and vi >= cfg.min_confidence:
            return "vi", max(0.0, min(1.0, vi))
        if en > vi and en >= cfg.min_confidence:
            return "en", max(0.0, min(1.0, en))

        # Very short text (e.g. "ok", a single emoji) carries no signal.
        if max(vi, en) > 0.0 and signal.n_words == 0:
            return "other", 0.0
        return "other", max(0.0, min(1.0, max(vi, en)))

    def detect_batch(self, texts: Sequence[object]) -> List[LanguageResult]:
        return [self.detect(text) for text in texts]


__all__ = [
    "LANGUAGES",
    "DetectionConfig",
    "LanguageDetector",
    "LanguageResult",
    "blend_scores",
]
