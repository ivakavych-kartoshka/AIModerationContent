"""Label-specific decision thresholds (plan section 12).

Strict protocol, implemented here so it cannot be violated by accident:

    VALIDATION  -> threshold optimisation -> freeze (thresholds.json)
                -> TEST evaluation with the frozen thresholds

The test set is never used to search thresholds.  ``thresholds.json`` records
the search space, the objective, the objective value before and after the search
and the exact thresholds, so the number in the paper can be reproduced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score

from ..common.constants import ID2LABEL, LABELS, NUM_LABELS
from ..common.logging_utils import get_logger
from .metrics import apply_decision_rule, compute_metrics

LOGGER = get_logger(__name__)

OBJECTIVES = ("macro_f1", "f1", "accuracy")


def _objective_value(objective: str, y_true: np.ndarray, y_pred: np.ndarray, focus: Optional[int] = None) -> float:
    if objective == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if objective == "f1":
        labels = list(range(NUM_LABELS)) if focus is None else [focus]
        return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    if objective == "accuracy":
        return float((y_true == y_pred).mean())
    raise ValueError(f"Unknown threshold objective {objective!r}; expected one of {OBJECTIVES}")


def optimize_thresholds(
    y_true: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    objective: str = "macro_f1",
    grid_size: int = 101,
    init: float = 0.5,
    coordinate_rounds: int = 3,
    num_labels: int = NUM_LABELS,
) -> Dict[str, Any]:
    """Coordinate ascent over per-label thresholds on the **validation** set.

    Returns a dict with the optimised thresholds, the objective value before and
    after, and the full search grid.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    grid = np.linspace(0.0, 1.0, int(grid_size))

    thresholds = np.full(num_labels, float(init), dtype=float)
    baseline_pred = apply_decision_rule(y_proba, "thresholds", thresholds)
    baseline_value = _objective_value(objective, y_true, baseline_pred)

    best_value = baseline_value
    trace: List[Dict[str, float]] = [
        {"round": 0, "label": "init", "threshold": float(init), "objective": baseline_value}
    ]

    for round_index in range(1, int(coordinate_rounds) + 1):
        improved_round = False
        for label_id in range(num_labels):
            best_t = float(thresholds[label_id])
            for candidate in grid:
                trial = thresholds.copy()
                trial[label_id] = float(candidate)
                pred = apply_decision_rule(y_proba, "thresholds", trial)
                value = _objective_value(objective, y_true, pred, focus=label_id)
                if value > best_value + 1e-12:
                    best_value = value
                    best_t = float(candidate)
                    improved_round = True
            thresholds[label_id] = best_t
            trace.append(
                {
                    "round": round_index,
                    "label": ID2LABEL[label_id],
                    "threshold": float(best_t),
                    "objective": float(best_value),
                }
            )
        if not improved_round:
            LOGGER.info("[thresholds] coordinate ascent converged after round %d", round_index)
            break

    final_pred = apply_decision_rule(y_proba, "thresholds", thresholds)
    final_metrics = compute_metrics(y_true, final_pred, y_proba)

    return {
        "objective": objective,
        "search_space": {"min": 0.0, "max": 1.0, "grid_size": int(grid_size)},
        "coordinate_rounds": int(coordinate_rounds),
        "initial_threshold": float(init),
        "thresholds": {ID2LABEL[i]: round(float(thresholds[i]), 4) for i in range(num_labels)},
        "threshold_list": [round(float(t), 4) for t in thresholds],
        "objective_before": round(float(baseline_value), 6),
        "objective_after": round(float(best_value), 6),
        "validation_macro_f1_with_thresholds": final_metrics["macro_f1"],
        "validation_accuracy_with_thresholds": final_metrics["accuracy"],
        "validation_macro_f1_with_argmax": compute_metrics(
            y_true, apply_decision_rule(y_proba, "argmax"), y_proba
        )["macro_f1"],
        "trace": trace,
        "note": "Optimised on the validation split only; frozen before test evaluation.",
    }


def load_thresholds(path: str | Path) -> Optional[List[float]]:
    """Read ``thresholds.json`` -> threshold list in canonical id order."""
    p = Path(path)
    if not p.is_file():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload.get("threshold_list"), list):
        return [float(t) for t in payload["threshold_list"]]
    mapping = payload.get("thresholds") or {}
    if mapping:
        return [float(mapping[ID2LABEL[i]]) for i in range(NUM_LABELS)]
    return None


def save_thresholds(payload: Dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[thresholds] saved -> %s", p)
    return p


def describe(payload: Dict[str, Any]) -> str:
    lines = [
        "Label-specific thresholds (optimised on VALIDATION, frozen for TEST)",
        "-" * 66,
        f"objective            : {payload.get('objective')}",
        f"objective before     : {payload.get('objective_before')}",
        f"objective after      : {payload.get('objective_after')}",
    ]
    thresholds = payload.get("thresholds", {})
    for name in LABELS:
        lines.append(f"threshold {name:<10}: {thresholds.get(name)}")
    return "\n".join(lines)


__all__ = ["OBJECTIVES", "describe", "load_thresholds", "optimize_thresholds", "save_thresholds"]
