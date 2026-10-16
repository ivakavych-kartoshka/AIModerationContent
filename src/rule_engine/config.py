"""Configuration objects for the rule engine (plan sections 37-39, 42).

Every threshold is an *initial hypothesis* that must be tuned on the validation
split (section 39). The YAML file ``configs/rule_engine.yaml`` only has to
declare the keys it changes; missing ones fall back to the defaults here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, get_type_hints

from ..common.config import load_yaml
from ..common.constants import LEXICON_PROCESSED_DIR

DEFAULT_LEXICON_FILE = "verified_lexicon.csv"


@dataclass
class NormalizeConfig:
    """Text normalisation pipeline (section 37)."""

    unicode_form: str = "NFC"
    lowercase: bool = True
    remove_invisible: bool = True
    collapse_whitespace: bool = True
    #: runs longer than this are shortened (0 disables the cap)
    max_repeat_char: int = 3
    #: hard limit on the characters fed to the matcher
    max_text_chars: int = 20000


@dataclass
class ObfuscationConfig:
    """Anti-obfuscation matching views (section 38)."""

    enabled: bool = True
    accentless: bool = True
    leet: bool = True
    collapse_repeats: bool = True
    #: remove separators (spaces/punctuation) between characters
    separator_stripping: bool = True
    #: minimum length of a term before the compact view is used
    min_compact_term_len: int = 3
    max_variant_chars: int = 40000


@dataclass
class MatcherConfig:
    """Dictionary matching (sections 38-39)."""

    min_term_len: int = 2
    #: bounded fuzzy matching (edit distance <= max_distance) for long unigrams
    fuzzy_enabled: bool = True
    fuzzy_max_distance: int = 1
    fuzzy_min_term_len: int = 6
    #: stop collecting matches after this many (largest scores kept)
    max_matches: int = 200


@dataclass
class ScoringConfig:
    """Rule scoring (section 39)."""

    severity: Dict[str, float] = field(default_factory=lambda: {"HATE": 0.90, "OFFENSIVE": 0.60})
    match_factors: Dict[str, float] = field(
        default_factory=lambda: {
            "exact": 1.00,
            "accentless_alias": 0.90,
            "leet": 0.80,
            "repeat_collapsed": 0.75,
            "separator_inserted": 0.80,
            "obfuscated_combined": 0.60,
            "fuzzy": 0.60,
        }
    )
    #: context factors for negation / quotation (section 39)
    negation_factor: float = 0.50
    quote_factor: float = 0.60
    context_required_factor: float = 0.85
    #: how many matches per label enter the aggregation
    aggregation_cap: int = 10
    #: only entries with these review statuses take part in matching
    min_review_statuses: Tuple[str, ...] = ("verified", "auto")


@dataclass
class RuleEngineConfig:
    """Top-level rule-engine configuration."""

    lexicon_path: str = str(LEXICON_PROCESSED_DIR / DEFAULT_LEXICON_FILE)
    dictionary_version: str = "v1"
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    obfuscation: ObfuscationConfig = field(default_factory=ObfuscationConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


class ConfigError(ValueError):
    """Raised when a rule-engine config is invalid."""


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _build_dataclass(cls: type, data: Optional[Mapping[str, Any]]) -> Any:
    """Instantiate a dataclass from a (possibly partial) mapping."""
    data = dict(data or {})
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        field_type = hints.get(f.name, f.type)
        if is_dataclass(field_type):
            kwargs[f.name] = _build_dataclass(field_type, data.get(f.name))
        elif f.name in data:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def load_rule_engine_config(path: str | Path) -> RuleEngineConfig:
    """Load a YAML file on top of :class:`RuleEngineConfig` defaults."""
    data = load_yaml(path)
    cfg = _build_dataclass(RuleEngineConfig, data)
    validate(cfg)
    return cfg


def default_config() -> RuleEngineConfig:
    return RuleEngineConfig()


def validate(cfg: RuleEngineConfig) -> None:
    problems: List[str] = []
    if cfg.normalize.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        problems.append(f"normalize.unicode_form unknown: {cfg.normalize.unicode_form!r}")
    if cfg.normalize.max_text_chars < 1:
        problems.append("normalize.max_text_chars must be >= 1")
    if cfg.matcher.fuzzy_max_distance < 0:
        problems.append("matcher.fuzzy_max_distance must be >= 0")
    for name, value in cfg.scoring.match_factors.items():
        if not 0.0 <= value <= 1.0:
            problems.append(f"scoring.match_factors.{name}={value} outside [0,1]")
    for name, value in cfg.scoring.severity.items():
        if not 0.0 < value <= 1.0:
            problems.append(f"scoring.severity.{name}={value} outside (0,1]")
    if cfg.scoring.aggregation_cap < 1:
        problems.append("scoring.aggregation_cap must be >= 1")
    if not set(cfg.scoring.min_review_statuses) <= {"verified", "auto", "needs_review"}:
        problems.append(f"scoring.min_review_statuses invalid: {cfg.scoring.min_review_statuses}")
    if not Path(cfg.lexicon_path).is_file():
        problems.append(f"lexicon not found: {cfg.lexicon_path}")
    if problems:
        raise ConfigError("Invalid rule-engine configuration:\n  - " + "\n  - ".join(problems))
