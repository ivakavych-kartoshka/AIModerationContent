"""Rule engine for Vietnamese sensitive-content detection (plan sections 35-39).

Standalone lexical detector: normalizes text, builds obfuscation views, matches
the labelled lexicon built by ``scripts/16_label_lexicon.py`` and produces
explainable heuristic scores used later by the AI/rule fusion step.

    from src.rule_engine import RuleEngine, load_rule_engine_config

    engine = RuleEngine.from_config("configs/rule_engine.yaml")
    engine.analyze("thằng ngu này, c0n ch0!")
"""

from .config import (
    ConfigError,
    MatcherConfig,
    NormalizeConfig,
    ObfuscationConfig,
    RuleEngineConfig,
    ScoringConfig,
    default_config,
    load_rule_engine_config,
)
from .engine import RuleEngine
from .lexical_classifier import Lexicon, LexiconEntry
from .matcher import Match, Matcher
from .normalize import NormalizedText, normalize
from .obfuscation import View, build_views
from .scoring import RuleResult, ScoredMatch, score_matches

__all__ = [
    "ConfigError",
    "Lexicon",
    "LexiconEntry",
    "Match",
    "Matcher",
    "MatcherConfig",
    "NormalizeConfig",
    "NormalizedText",
    "ObfuscationConfig",
    "RuleEngine",
    "RuleEngineConfig",
    "RuleResult",
    "ScoredMatch",
    "ScoringConfig",
    "View",
    "build_views",
    "default_config",
    "load_rule_engine_config",
    "normalize",
    "score_matches",
]
