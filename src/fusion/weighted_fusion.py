"""Score-level fusion of the lexical rule engine and the AI classifier.

Implements the baseline fusion of the research plan (``Rule_Engine_Plan.md``
section 40)::

    hybrid_hate      = alpha * P(HATE)      + (1 - alpha) * rule_hate_score
    hybrid_offensive = alpha * P(OFFENSIVE) + (1 - alpha) * rule_offensive_score
    hybrid_clean     = alpha * P(CLEAN)     + (1 - alpha) * rule_clean_score

The rule engine exposes two scores (``rule_hate_score``, ``rule_offensive_score``)
but no CLEAN score, so - exactly like ``scripts/17_evaluate_rule_engine.py`` - the
CLEAN channel is derived as ``rule_clean_score = 1 - max(hate, offensive)``.  This
keeps the rule side on the same ``[HATE, OFFENSIVE, CLEAN]`` axis as the neural
model, so the two can be blended channel by channel.

Two fusion families are provided:

* **weighted fusion** - a convex combination with a single mixing weight
  ``alpha``.  ``alpha`` (and, optionally, label-specific decision thresholds) are
  chosen on the **validation** split and frozen before the single test pass.
* **meta-classifier** - a logistic regression over the six scores
  ``[P(HATE), P(OFF), P(CLEAN), rule_hate, rule_off, rule_clean]``, also fitted on
  validation (plan section 40: "logistic regression / meta-classifier").

Nothing in this module ever looks at the test set to choose a parameter; callers
pass the already-computed scores and decide which split is which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common.constants import ID2LABEL, LABELS, NUM_LABELS
from ..evaluation.metrics import apply_decision_rule, compute_metrics
from ..evaluation.thresholds import optimize_thresholds

HATE_ID = LABELS.index("HATE")
OFFENSIVE_ID = LABELS.index("OFFENSIVE")
CLEAN_ID = LABELS.index("CLEAN")

#: Names of the six raw features a meta-classifier sees.
COMPONENTS = (
    "p_hate",
    "p_offensive",
    "p_clean",
    "rule_hate",
    "rule_offensive",
    "rule_clean",
)


def rule_pseudo_proba(hate: np.ndarray, offensive: np.ndarray) -> np.ndarray:
    """``[HATE, OFFENSIVE, CLEAN]`` pseudo-probability from the two rule scores."""
    hate = np.asarray(hate, dtype=float)
    offensive = np.asarray(offensive, dtype=float)
    clean = np.clip(1.0 - np.maximum(hate, offensive), 0.0, 1.0)
    return np.stack([hate, offensive, clean], axis=1)


def fuse_scores(
    p_ai: np.ndarray,
    rule_hate: np.ndarray,
    rule_offensive: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Convex combination of the AI probabilities and the rule pseudo-probabilities.

    ``alpha`` is the AI weight (``1 - alpha`` is the rule weight).  Returns an
    ``(N, 3)`` matrix in canonical ``[HATE, OFFENSIVE, CLEAN]`` order.
    """
    p_ai = np.asarray(p_ai, dtype=float)
    rule = rule_pseudo_proba(rule_hate, rule_offensive)
    if p_ai.shape != rule.shape:
        raise ValueError(f"AI scores {p_ai.shape} and rule scores {rule.shape} do not match")
    a = float(alpha)
    return a * p_ai + (1.0 - a) * rule


def meta_features(
    p_ai: np.ndarray,
    rule_hate: np.ndarray,
    rule_offensive: np.ndarray,
) -> np.ndarray:
    """``(N, 6)`` feature matrix = AI probabilities + rule pseudo-probabilities."""
    p_ai = np.asarray(p_ai, dtype=float)
    rule = rule_pseudo_proba(rule_hate, rule_offensive)
    return np.hstack([p_ai, rule])


def fuse_scores_per_channel(
    p_ai: np.ndarray,
    rule_hate: np.ndarray,
    rule_offensive: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    """Blend each ``[HATE, OFFENSIVE, CLEAN]`` channel with its own weight.

    ``alphas`` is a length-3 vector of AI weights.  A single global ``alpha`` is
    the special case ``alphas = (alpha, alpha, alpha)``; freeing the three
    channels lets the rule engine contribute strongly on HATE (where it is
    precise) while being down-weighted on OFFENSIVE (where it is weak).
    """
    p_ai = np.asarray(p_ai, dtype=float)
    rule = rule_pseudo_proba(rule_hate, rule_offensive)
    if p_ai.shape != rule.shape:
        raise ValueError(f"AI scores {p_ai.shape} and rule scores {rule.shape} do not match")
    a = np.asarray(alphas, dtype=float).reshape(1, -1)
    if a.shape[1] != NUM_LABELS:
        raise ValueError(f"alphas must have {NUM_LABELS} entries, got {a.shape[1]}")
    return a * p_ai + (1.0 - a) * rule


def score_alpha_grid_per_channel(
    y_true: Sequence[int],
    p_ai: np.ndarray,
    rule_hate: np.ndarray,
    rule_offensive: np.ndarray,
    alphas: Sequence[float],
    rounds: int = 3,
    init: float = 1.0,
) -> Tuple[List[float], List[Dict[str, object]]]:
    """Coordinate-ascent search for the per-channel AI weights on a labelled split.

    Starts from the AI-only corner (``alpha=1`` everywhere) and, for each channel
    in turn, moves it to the grid value that maximises macro-F1 of the plain
    ``argmax`` decision.  Returns the weights and the optimisation trace.
    """
    y_true = np.asarray(y_true, dtype=int)
    grid = [float(a) for a in alphas]
    current = np.full(NUM_LABELS, float(init), dtype=float)

    def macro_of(vec: np.ndarray) -> float:
        scores = fuse_scores_per_channel(p_ai, rule_hate, rule_offensive, vec)
        return float(compute_metrics(y_true, scores.argmax(axis=1), scores)["macro_f1"])

    best = macro_of(current)
    trace: List[Dict[str, object]] = [
        {"round": 0, "channel": "init", "alpha": current.round(4).tolist(), "macro_f1": round(best, 6)}
    ]
    for round_index in range(1, max(1, int(rounds)) + 1):
        improved = False
        for channel in range(NUM_LABELS):
            local_alpha = float(current[channel])
            local_value = best
            for candidate in grid:
                trial = current.copy()
                trial[channel] = float(candidate)
                value = macro_of(trial)
                if value > local_value + 1e-12:
                    local_value = value
                    local_alpha = float(candidate)
                    improved = True
            current[channel] = local_alpha
            best = local_value
        trace.append(
            {
                "round": round_index,
                "channel": "all",
                "alpha": current.round(4).tolist(),
                "macro_f1": round(best, 6),
            }
        )
        if not improved:
            break
    return current.round(4).tolist(), trace


def score_alpha_grid(
    y_true: Sequence[int],
    p_ai: np.ndarray,
    rule_hate: np.ndarray,
    rule_offensive: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[float, List[Dict[str, float]]]:
    """Sweep ``alpha`` on a labelled split; return the best value and the grid.

    The selection criterion is macro-F1 of the plain ``argmax`` decision, which
    is cheap and monotone enough to locate the useful ``alpha``; the caller then
    refines the decision with thresholds at that ``alpha`` (if desired).
    """
    y_true = np.asarray(y_true, dtype=int)
    rows: List[Dict[str, float]] = []
    best_alpha = float(alphas[0])
    best_f1 = -1.0
    for alpha in alphas:
        scores = fuse_scores(p_ai, rule_hate, rule_offensive, alpha)
        pred = scores.argmax(axis=1)
        metrics = compute_metrics(y_true, pred, scores)
        rows.append(
            {
                "alpha": round(float(alpha), 4),
                "macro_f1": round(float(metrics["macro_f1"]), 6),
                "accuracy": round(float(metrics["accuracy"]), 6),
                "HATE_f1": round(float(metrics["HATE_f1"]), 6),
                "OFFENSIVE_f1": round(float(metrics["OFFENSIVE_f1"]), 6),
                "CLEAN_f1": round(float(metrics["CLEAN_f1"]), 6),
            }
        )
        if metrics["macro_f1"] > best_f1 + 1e-12:
            best_f1 = float(metrics["macro_f1"])
            best_alpha = float(alpha)
    return best_alpha, rows


def tune_thresholds_at_alpha(
    y_true: Sequence[int],
    p_ai: np.ndarray,
    rule_hate: np.ndarray,
    rule_offensive: np.ndarray,
    alpha: float,
    objective: str = "macro_f1",
    grid_size: int = 101,
    coordinate_rounds: int = 3,
) -> Dict[str, object]:
    """Label-specific thresholds for the fused scores, tuned on the given split.

    Reuses :func:`src.evaluation.thresholds.optimize_thresholds`, so the fused
    decision rule is scored exactly like the AI-only and rule-only systems.
    """
    scores = fuse_scores(p_ai, rule_hate, rule_offensive, alpha)
    payload = optimize_thresholds(
        np.asarray(y_true, dtype=int),
        scores,
        objective=objective,
        grid_size=grid_size,
        coordinate_rounds=coordinate_rounds,
    )
    payload["alpha"] = round(float(alpha), 4)
    return payload


def tune_thresholds_at_alphas(
    y_true: Sequence[int],
    p_ai: np.ndarray,
    rule_hate: np.ndarray,
    rule_offensive: np.ndarray,
    alphas: Sequence[float],
    objective: str = "macro_f1",
    grid_size: int = 101,
    coordinate_rounds: int = 3,
) -> Dict[str, object]:
    """Label-specific thresholds for the **per-channel** fused scores."""
    scores = fuse_scores_per_channel(p_ai, rule_hate, rule_offensive, alphas)
    payload = optimize_thresholds(
        np.asarray(y_true, dtype=int),
        scores,
        objective=objective,
        grid_size=grid_size,
        coordinate_rounds=coordinate_rounds,
    )
    payload["alphas"] = [round(float(a), 4) for a in alphas]
    return payload


@dataclass
class MetaFusion:
    """Logistic-regression stacker over the six rule/AI scores (plan section 40).

    Fitted on validation only; ``predict`` returns ``(N, 3)`` probabilities in
    canonical order.
    """

    C: float = 1.0
    max_iter: int = 1000
    class_weight: str = "balanced"
    seed: int = 42
    _model: object = field(default=None, init=False, repr=False)

    def fit(
        self,
        p_ai: np.ndarray,
        rule_hate: np.ndarray,
        rule_offensive: np.ndarray,
        y_true: Sequence[int],
    ) -> "MetaFusion":
        from sklearn.linear_model import LogisticRegression

        X = meta_features(p_ai, rule_hate, rule_offensive)
        self._model = LogisticRegression(
            C=float(self.C),
            max_iter=int(self.max_iter),
            class_weight=self.class_weight,
            random_state=int(self.seed),
        )
        self._model.fit(X, np.asarray(y_true, dtype=int))
        return self

    def predict_proba(
        self,
        p_ai: np.ndarray,
        rule_hate: np.ndarray,
        rule_offensive: np.ndarray,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MetaFusion.fit must be called before predict_proba")
        X = meta_features(p_ai, rule_hate, rule_offensive)
        proba = self._model.predict_proba(X)
        # LogisticRegression may omit a class that never appears in the fit set;
        # expand back to the fixed 3-column canonical order.
        classes = list(getattr(self._model, "classes_", range(proba.shape[1])))
        if len(classes) != NUM_LABELS:
            full = np.zeros((proba.shape[0], NUM_LABELS), dtype=float)
            for column, label_id in enumerate(classes):
                full[:, int(label_id)] = proba[:, column]
            proba = full
        return proba

    def coefficients(self) -> Optional[Dict[str, object]]:
        if self._model is None:
            return None
        coef = getattr(self._model, "coef_", None)
        intercept = getattr(self._model, "intercept_", None)
        classes = [ID2LABEL[int(c)] for c in getattr(self._model, "classes_", [])]
        return {
            "features": list(COMPONENTS),
            "classes": classes,
            "coef": None if coef is None else np.asarray(coef).round(4).tolist(),
            "intercept": None if intercept is None else np.asarray(intercept).round(4).tolist(),
        }


__all__ = [
    "CLEAN_ID",
    "COMPONENTS",
    "HATE_ID",
    "OFFENSIVE_ID",
    "MetaFusion",
    "fuse_scores",
    "fuse_scores_per_channel",
    "meta_features",
    "rule_pseudo_proba",
    "score_alpha_grid",
    "score_alpha_grid_per_channel",
    "tune_thresholds_at_alpha",
    "tune_thresholds_at_alphas",
]
