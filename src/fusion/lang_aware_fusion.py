"""Language-aware fusion.

Chain::

    raw comment
        |> LanguageDetector (vi / en / mix / other)
        |> LanguageAwareTokenizer (language-tagged segments)
        |> LanguageRouter (vi/en/multilingual rule engines)
        |> AI model selected by language
        |> weighted fusion (alpha per language)

The existing :mod:`src.fusion.weighted_fusion` contract is reused unchanged:
``[HATE, OFFENSIVE, CLEAN]`` AI probabilities are blended channel by channel
with the rule pseudo-probabilities.  The only new thing is that ``alpha``, the
rule engine and the AI model can differ per language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..common.constants import ID2LABEL, LABELS, NUM_LABELS
from ..common.logging_utils import get_logger
from ..fusion.weighted_fusion import fuse_scores, rule_pseudo_proba
from ..routing.language_router import LanguageRouter, RoutedRuleResult

LOGGER = get_logger("sensitiveai.fusion.lang_aware")

Predictor = Callable[[Sequence[str]], Any]


def _as_proba(value: Any, n: int) -> Optional[np.ndarray]:
    """Normalise a predictor's output to an ``(n, 3)`` float matrix."""
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] == 1 and n > 1:
        array = np.repeat(array, n, axis=0)
    if array.ndim != 2 or array.shape[1] != NUM_LABELS:
        raise ValueError(f"predictor must return (N, {NUM_LABELS}) probabilities, got {array.shape}")
    # Make row-wise softmax-safe even if the predictor returns logits.
    if np.any(array < 0) or not np.allclose(array.sum(axis=1), 1.0, atol=1e-3):
        shifted = array - array.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        array = exp / exp.sum(axis=1, keepdims=True)
    return array


class _PredictorAdapter:
    """Accept a function, ``predict_proba`` or ``predict`` object."""

    def __init__(self, target: Any) -> None:
        self.target = target

    def __call__(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if self.target is None:
            return None
        n = len(texts)
        if callable(self.target) and not hasattr(self.target, "predict_proba"):
            return _as_proba(self.target(texts), n)
        if hasattr(self.target, "predict_proba"):
            return _as_proba(self.target.predict_proba(texts), n)
        if hasattr(self.target, "predict"):
            return _as_proba(self.target.predict(texts), n)
        raise TypeError(f"Unsupported predictor: {type(self.target)!r}")


@dataclass
class LangFusionResult:
    """Output of :meth:`LangAwareFusion.analyze`."""

    text: str
    lang: str
    confidence: float
    model_language: str
    alpha: float
    fused_scores: Dict[str, float] = field(default_factory=dict)
    label: str = "CLEAN"
    label_id: int = 2
    rule: Optional[Dict[str, object]] = None
    segments: List[Dict[str, object]] = field(default_factory=list)
    matches: List[Dict[str, object]] = field(default_factory=list)
    ai_used: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "lang": self.lang,
            "confidence": round(float(self.confidence), 4),
            "model_language": self.model_language,
            "alpha": round(float(self.alpha), 4),
            "fused_scores": {key: round(float(v), 4) for key, v in self.fused_scores.items()},
            "label": self.label,
            "label_id": int(self.label_id),
            "ai_used": self.ai_used,
            "segments": self.segments,
            "matches": self.matches,
            "rule": self.rule,
        }


class LangAwareFusion:
    """Route by language, scan with the matching lexicon, then fuse with AI."""

    def __init__(
        self,
        router: Optional[LanguageRouter] = None,
        models: Optional[Mapping[str, Any]] = None,
        default_model: Any = None,
        alphas: Optional[Mapping[str, float]] = None,
        default_alpha: float = 0.70,
        decision_thresholds: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.router = router or LanguageRouter.from_config_files()
        self.detector = self.router.detector
        self.models: Dict[str, Any] = dict(models or {})
        self.default_model = default_model
        self.alphas: Dict[str, float] = dict(alphas or {})
        self.default_alpha = float(default_alpha)
        self.decision_thresholds = dict(decision_thresholds or {})
        self._adapters: Dict[str, _PredictorAdapter] = {}

    @classmethod
    def from_config(
        cls,
        router_config_path: str | Path = "configs/language_router.yaml",
        **kwargs: Any,
    ) -> "LangAwareFusion":
        """Build the router from ``configs/language_router.yaml``.

        Models are *not* loaded from the YAML (checkpoints need the full eval
        stack); pass them with ``models=`` / :meth:`set_model`.
        """
        router = LanguageRouter.from_config_files(router_config_path=router_config_path)
        kwargs.setdefault("alphas", router.config.alphas)
        return cls(router=router, **kwargs)

    # ------------------------------------------------------------------ #
    # model registry
    # ------------------------------------------------------------------ #
    def set_model(self, lang: str, model: Any) -> None:
        self.models[str(lang)] = model
        # Fallback resolution can change for other languages too (e.g. a new
        # ``en`` model is now the fallback for ``multilingual``).
        self._adapters.clear()

    def _model_for(self, lang: str) -> Any:
        if lang in self.models:
            return self.models[lang]
        if "multilingual" in self.models:
            return self.models["multilingual"]
        if "vi" in self.models:
            return self.models["vi"]
        if "en" in self.models:
            return self.models["en"]
        return self.default_model

    def _adapter_for(self, lang: str) -> _PredictorAdapter:
        if lang not in self._adapters:
            self._adapters[lang] = _PredictorAdapter(self._model_for(lang))
        return self._adapters[lang]

    def _alpha_for(self, lang: str) -> float:
        return float(self.alphas.get(lang, self.alphas.get("default", self.default_alpha)))

    # ------------------------------------------------------------------ #
    # analysis
    # ------------------------------------------------------------------ #
    def _decide_label(self, scores: np.ndarray) -> int:
        if not self.decision_thresholds:
            return int(np.argmax(scores))
        thresholded = np.zeros_like(scores)
        for index, label in enumerate(LABELS):
            thresholded[index] = scores[index] >= float(self.decision_thresholds.get(label, 0.0))
        if not thresholded.any():
            return int(np.argmax(scores))
        # Among labels above threshold, pick the highest score (ties -> leftmost).
        return int(np.argmax(np.where(thresholded, scores, -np.inf)))

    def analyze(self, text: object) -> LangFusionResult:
        raw = "" if text is None else str(text)
        routed: RoutedRuleResult = self.router.route(raw)
        lang = routed.lang
        model_lang = lang if lang in ("vi", "en") else "multilingual"
        if model_lang not in self.models and model_lang != "multilingual":
            model_lang = "multilingual"
        alpha = self._alpha_for(lang if lang in self.alphas else model_lang)

        adapter = self._adapter_for(model_lang)
        proba = adapter([raw])
        ai_used = proba is not None
        rule_hate = np.array([routed.rule_hate_score], dtype=float)
        rule_offensive = np.array([routed.rule_offensive_score], dtype=float)
        if ai_used:
            fused = fuse_scores(proba, rule_hate, rule_offensive, alpha)
        else:
            fused = rule_pseudo_proba(rule_hate, rule_offensive)

        label_id = self._decide_label(fused[0])
        return LangFusionResult(
            text=raw,
            lang=lang,
            confidence=routed.confidence,
            model_language=model_lang,
            alpha=alpha if ai_used else 0.0,
            fused_scores={label: float(fused[0][index]) for index, label in enumerate(LABELS)},
            label=ID2LABEL[label_id],
            label_id=label_id,
            rule=routed.to_dict(),
            segments=routed.segments,
            matches=routed.matches,
            ai_used=ai_used,
        )

    def analyze_batch(self, texts: Sequence[object]) -> List[LangFusionResult]:
        return [self.analyze(text) for text in texts]

    def scan(self, text: object) -> Dict[str, object]:
        """Convenience wrapper returning a JSON-serialisable dict."""
        return self.analyze(text).to_dict()


__all__ = [
    "LangAwareFusion",
    "LangFusionResult",
    "Predictor",
]
