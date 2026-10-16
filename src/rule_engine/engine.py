"""The rule engine: normalization -> views -> matching -> lexical scoring.

Public entry point (section 35/39): :class:`RuleEngine` returns, for one raw
Vietnamese comment, the heuristic scores ``rule_hate_score``,
``rule_offensive_score`` and ``rule_score`` in ``[0, 1]`` plus the explainable
evidence behind them (matched span, dictionary form, label, rule, offsets).

Section 41 keeps the rule engine independent from the AI model: it can run on
its own, and the later fusion step only consumes its scores.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..common.logging_utils import get_logger
from .config import RuleEngineConfig, default_config, load_rule_engine_config
from .lexical_classifier import Lexicon
from .matcher import Matcher
from .normalize import normalize
from .obfuscation import build_views
from .scoring import RuleResult, score_matches

LOGGER = get_logger("sensitiveai.rule_engine")


class RuleEngine:
    """Standalone lexical rule engine."""

    def __init__(self, lexicon: Lexicon, config: Optional[RuleEngineConfig] = None) -> None:
        self.lexicon = lexicon
        self.config = config or default_config()
        self.matcher = Matcher(
            lexicon,
            matcher_cfg=self.config.matcher,
            obfuscation_cfg=self.config.obfuscation,
            match_factors=self.config.scoring.match_factors,
            severity_by_label=self.config.scoring.severity,
        )
        LOGGER.info(
            "[rule_engine] ready: %d entries, dict=%s, obfuscation=%s",
            len(lexicon),
            lexicon.dictionary_version,
            self.config.obfuscation.enabled,
        )

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config_path: str | Path) -> "RuleEngine":
        config = load_rule_engine_config(config_path)
        lexicon = Lexicon.load(config.lexicon_path, config.scoring.min_review_statuses)
        return cls(lexicon, config)

    @classmethod
    def from_lexicon(
        cls, lexicon_path: str | Path, config: Optional[RuleEngineConfig] = None
    ) -> "RuleEngine":
        config = config or default_config()
        config.lexicon_path = str(lexicon_path)
        lexicon = Lexicon.load(lexicon_path, config.scoring.min_review_statuses)
        return cls(lexicon, config)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _analyze_result(self, text: object) -> RuleResult:
        started = time.perf_counter()
        normalized = normalize(text, self.config.normalize)
        if not normalized.normalized_text:
            return RuleResult(
                rule_score=0.0,
                rule_hate_score=0.0,
                rule_offensive_score=0.0,
                matches=[],
                dictionary_version=self.lexicon.dictionary_version,
                meta={"n_matches": 0, "n_spans": 0, "labels": [], "empty": True},
            )

        views = build_views(normalized, self.config.obfuscation)
        matches = self.matcher.match(normalized, views)
        result = score_matches(
            normalized, matches, self.config.scoring, self.lexicon.dictionary_version
        )
        result.meta.update(
            {
                "text_length": len(normalized.normalized_text),
                "raw_length": len(normalized.raw_text),
                "truncated": normalized.truncated,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        return result

    def analyze(self, text: object) -> Dict[str, object]:
        """Analyze one comment and return the plan section 39 evidence dict."""
        return self._analyze_result(text).to_dict()

    def analyze_batch(self, texts: Iterable[object]) -> List[Dict[str, object]]:
        return [self.analyze(text) for text in texts]

    def analyze_scores(self, text: object) -> RuleResult:
        """Like :meth:`analyze` but returns the typed result object."""
        return self._analyze_result(text)
