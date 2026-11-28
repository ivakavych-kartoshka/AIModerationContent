"""Language-based routing for rule engines and models.

The router decides *which pipeline to use* based on :class:`LanguageResult`,
and :class:`LanguageAwareRouter` goes one step further: for a
``"mix"`` document it can run **both** rule engines (Vietnamese and English)
and return the union of matches/evidence.

The design matches the codebase's principle: keep the rule engine
independent, and let fusion consume the produced scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.logging_utils import get_logger
from ..langid.detector import DetectionConfig, LanguageDetector, LanguageResult
from ..langid.tokenizer import LanguageAwareTokenizer
from ..rule_engine import RuleEngine, load_rule_engine_config

LOGGER = get_logger("sensitiveai.routing")


@dataclass
class RouteConfig:
    """Routing configuration (can be loaded from ``configs/lang_router.yaml``)."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    #: lexicon paths per language; leave empty to use rule-engine defaults
    lexicon_paths: Dict[str, str] = field(default_factory=dict)
    #: rule-engine config paths per language
    rule_config_paths: Dict[str, str] = field(default_factory=dict)
    #: fallback language when detection is ``other`` or confidence is too low
    fallback_lang: str = "multilingual"
    #: run both rule engines for mixed comments
    use_both_for_mix: bool = True
    #: if True, segment the text and scan each segment with its language's lexicon
    segment_by_language: bool = True
    #: AI weight per language, consumed by :class:`src.fusion.lang_aware.LangAwareFusion`
    alphas: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoutedRuleResult:
    """Aggregate of rule-engine results from one or more language pipelines."""

    lang: str
    confidence: float
    segments: List[Dict[str, object]] = field(default_factory=list)
    results: Dict[str, Dict[str, object]] = field(default_factory=dict)
    rule_hate_score: float = 0.0
    rule_offensive_score: float = 0.0
    rule_clean_score: float = 1.0
    rule_score: float = 0.0
    matches: List[Dict[str, object]] = field(default_factory=list)
    dictionary_versions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "lang": self.lang,
            "confidence": round(float(self.confidence), 4),
            "segments": self.segments,
            "results": self.results,
            "rule_hate_score": round(float(self.rule_hate_score), 4),
            "rule_offensive_score": round(float(self.rule_offensive_score), 4),
            "rule_clean_score": round(float(self.rule_clean_score), 4),
            "rule_score": round(float(self.rule_score), 4),
            "matches": self.matches,
            "dictionary_versions": self.dictionary_versions,
        }


class LanguageRouter:
    """Route raw text to the appropriate :class:`RuleEngine`(s)."""

    def __init__(
        self,
        config: Optional[RouteConfig] = None,
        engines: Optional[Dict[str, RuleEngine]] = None,
    ) -> None:
        self.config = config or RouteConfig()
        self.detector = LanguageDetector(self.config.detection)
        self.tokenizer = LanguageAwareTokenizer(self.detector)
        self.engines = engines or {}
        self._default_engine: Optional[RuleEngine] = None
        if "vi" not in self.engines and "default" in self.engines:
            self.engines["vi"] = self.engines["default"]
        if self._default_engine is None:
            self._default_engine = self.engines.get("vi") or self.engines.get("en")

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config_files(
        cls,
        router_config_path: str | Path | None = "configs/language_router.yaml",
        vi_rule_config_path: str | Path = "configs/rule_engine.yaml",
        en_rule_config_path: str | Path | None = None,
    ) -> "LanguageRouter":
        config = RouteConfig()
        if router_config_path and Path(router_config_path).is_file():
            from ..common.config import load_yaml

            data = load_yaml(router_config_path) or {}
            config = _route_config_from_dict(data)
        engines: Dict[str, RuleEngine] = {}
        vi_config = config.rule_config_paths.get("vi") or vi_rule_config_path
        engines["vi"] = RuleEngine.from_config(vi_config)
        if en_rule_config_path and Path(en_rule_config_path).is_file():
            engines["en"] = RuleEngine.from_config(en_rule_config_path)
        elif config.rule_config_paths.get("en"):
            engines["en"] = RuleEngine.from_config(config.rule_config_paths["en"])
        elif config.lexicon_paths.get("en"):
            # Build an English rule engine on top of the same matching/scoring
            # defaults - only the lexicon changes.
            rule_cfg = load_rule_engine_config(vi_rule_config_path)
            rule_cfg.lexicon_path = str(config.lexicon_paths["en"])
            from ..rule_engine.lexical_classifier import Lexicon

            engines["en"] = RuleEngine(
                Lexicon.load(rule_cfg.lexicon_path, rule_cfg.scoring.min_review_statuses),
                rule_cfg,
            )
        else:
            # Best-effort: if no English lexicon/rule config is provided, fall back
            # to the Vietnamese engine.  ``use_both_for_mix`` will still work if an
            # English engine is added later.
            engines["en"] = engines["vi"]
        return cls(config, engines)

    # ------------------------------------------------------------------ #
    # routing
    # ------------------------------------------------------------------ #
    @property
    def default_engine(self) -> RuleEngine:
        if self._default_engine is None:
            self._default_engine = self.engines.get("vi") or self.engines.get("en")
        if self._default_engine is None:
            raise RuntimeError("No rule engine available for routing")
        return self._default_engine

    def _engine_for(self, lang: str) -> Optional[RuleEngine]:
        if lang == "vi":
            return self.engines.get("vi") or self._default_engine
        if lang == "en":
            return self.engines.get("en") or self.engines.get("vi") or self._default_engine
        return self._default_engine
    def route(self, text: object) -> RoutedRuleResult:
        """Route ``text`` and return aggregated rule-engine scores."""
        raw = "" if text is None else str(text)
        segments, doc = self.tokenizer.segment(raw)
        lang = doc.lang
        confidence = doc.confidence

        if lang == "other":
            lang = self.config.fallback_lang

        aggregate = RoutedRuleResult(
            lang=lang,
            confidence=confidence,
            segments=[seg.to_dict() for seg in segments],
        )

        engines_to_run: List[Tuple[str, RuleEngine]] = []
        if lang == "mix" and self.config.use_both_for_mix:
            for name in ("vi", "en"):
                engine = self._engine_for(name)
                if engine is not None and engine not in (e for _, e in engines_to_run):
                    engines_to_run.append((name, engine))
        elif lang == "multilingual" or lang == self.config.fallback_lang:
            for name in ("vi", "en"):
                engine = self._engine_for(name)
                if engine is not None and engine not in (e for _, e in engines_to_run):
                    engines_to_run.append((name, engine))
        else:
            engine = self._engine_for(lang)
            if engine is not None:
                engines_to_run.append((lang, engine))

        if not engines_to_run and self._default_engine is not None:
            engines_to_run.append((lang, self._default_engine))

        # Scan each segment with its language's engine when possible.
        if self.config.segment_by_language and segments:
            for seg in segments:
                seg_lang = seg.lang if seg.lang in ("vi", "en") else lang
                engine = self._engine_for(seg_lang)
                if engine is None:
                    continue
                result = engine.analyze(seg.text)
                _merge_routed_result(aggregate, seg_lang, result)
        else:
            # Whole-text scan with every selected engine.
            for name, engine in engines_to_run:
                result = engine.analyze(raw)
                _merge_routed_result(aggregate, name, result)

        # Derive the clean score like fusion does: rule_clean = 1 - max(hate, offensive)
        aggregate.rule_clean_score = max(0.0, min(1.0, 1.0 - max(aggregate.rule_hate_score, aggregate.rule_offensive_score)))
        aggregate.rule_score = max(aggregate.rule_hate_score, aggregate.rule_offensive_score)
        return aggregate

    def analyze(self, text: object) -> RoutedRuleResult:
        return self.route(text)

    def detect(self, text: object) -> LanguageResult:
        return self.detector.detect(text)


def _merge_routed_result(aggregate: RoutedRuleResult, name: str, result: Dict[str, Any]) -> None:
    aggregate.results[name] = result
    if result.get("dictionary_version"):
        version = str(result["dictionary_version"])
        if version and version not in aggregate.dictionary_versions:
            aggregate.dictionary_versions.append(version)
    aggregate.rule_hate_score = max(aggregate.rule_hate_score, float(result.get("rule_hate_score", 0.0)))
    aggregate.rule_offensive_score = max(aggregate.rule_offensive_score, float(result.get("rule_offensive_score", 0.0)))
    matches = result.get("matches")
    if isinstance(matches, list):
        aggregate.matches.extend(matches)


def _route_config_from_dict(data: Dict[str, Any]) -> RouteConfig:
    detection = DetectionConfig()
    det_data = data.get("detection") or {}
    for field_name in DetectionConfig.__dataclass_fields__:
        if field_name in det_data:
            setattr(detection, field_name, det_data[field_name])
    return RouteConfig(
        detection=detection,
        lexicon_paths=data.get("lexicon_paths") or {},
        rule_config_paths=data.get("rule_config_paths") or {},
        fallback_lang=data.get("fallback_lang", "multilingual"),
        use_both_for_mix=bool(data.get("use_both_for_mix", True)),
        segment_by_language=bool(data.get("segment_by_language", True)),
        alphas={str(k): float(v) for k, v in (data.get("alphas") or {}).items()},
    )


__all__ = [
    "LanguageRouter",
    "RouteConfig",
    "RoutedRuleResult",
]
