"""Precision-Recall and ROC curves.

All curves are computed **one-vs-rest** because this is a 3-class problem
(plan sections 21 & 22):

    HATE      vs  NOT-HATE
    OFFENSIVE vs  NOT-OFFENSIVE
    CLEAN     vs  NOT-CLEAN

Each model gets ``pr_curve.png`` / ``roc_curve.png`` showing the three classes
plus the macro average, and the raw curve coordinates are written to
``pr_curve_data.json`` / ``roc_curve_data.json`` so the 4-model comparison can
re-plot them without re-running inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from ..common.constants import ID2LABEL, NUM_LABELS
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _onehot(y_true: np.ndarray, num_labels: int = NUM_LABELS) -> np.ndarray:
    return np.eye(num_labels)[np.asarray(y_true, dtype=int)]


def precision_recall_curve_data(
    y_true: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    num_labels: int = NUM_LABELS,
) -> Dict[str, Any]:
    """Per-class PR curves + macro/micro averages + average precision."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    onehot = _onehot(y_true, num_labels)

    data: Dict[str, Any] = {"classes": [ID2LABEL[i] for i in range(num_labels)], "per_class": {}}
    macro_p: List[np.ndarray] = []
    macro_r: List[np.ndarray] = []
    aps: List[float] = []

    for i in range(num_labels):
        precision, recall, _ = precision_recall_curve(onehot[:, i], y_proba[:, i])
        ap = float(average_precision_score(onehot[:, i], y_proba[:, i])) if onehot[:, i].sum() > 0 else float("nan")
        data["per_class"][ID2LABEL[i]] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "average_precision": ap,
            "positive_support": int(onehot[:, i].sum()),
        }
        if not np.isnan(ap):
            aps.append(ap)
        # interpolate onto a common recall axis for the macro curve
        grid = np.linspace(0.0, 1.0, 200)
        macro_p.append(np.interp(grid, recall[::-1], precision[::-1]))
        macro_r.append(grid)

    data["macro_average_precision"] = float(np.mean(aps)) if aps else float("nan")
    data["macro_precision"] = np.mean(np.stack(macro_p), axis=0).tolist()
    data["macro_recall"] = np.linspace(0.0, 1.0, 200).tolist()
    try:
        data["micro_average_precision"] = float(average_precision_score(onehot, y_proba, average="micro"))
    except ValueError:
        data["micro_average_precision"] = float("nan")
    data["note"] = "One-vs-rest curves; macro curve is the recall-wise mean of the three class curves."
    return data


def roc_curve_data(
    y_true: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    num_labels: int = NUM_LABELS,
) -> Dict[str, Any]:
    """Per-class ROC curves + macro/micro AUC."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    onehot = _onehot(y_true, num_labels)

    data: Dict[str, Any] = {"classes": [ID2LABEL[i] for i in range(num_labels)], "per_class": {}}
    grid = np.linspace(0.0, 1.0, 200)
    tprs: List[np.ndarray] = []
    aucs: List[float] = []

    for i in range(num_labels):
        if onehot[:, i].sum() == 0:
            fpr = np.concatenate([[0.0, 1.0], np.linspace(0.0, 1.0, 200)])
            tpr = np.concatenate([[0.0, 1.0], np.linspace(0.0, 1.0, 200)])
            auc_value = float("nan")
        else:
            fpr, tpr, _ = roc_curve(onehot[:, i], y_proba[:, i])
            auc_value = float(roc_auc_score(onehot[:, i], y_proba[:, i]))
            aucs.append(auc_value)
        data["per_class"][ID2LABEL[i]] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": auc_value,
            "positive_support": int(onehot[:, i].sum()),
        }
        tprs.append(np.interp(grid, fpr, tpr))

    data["macro_roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    data["macro_fpr"] = grid.tolist()
    data["macro_tpr"] = np.mean(np.stack(tprs), axis=0).tolist()
    try:
        data["micro_roc_auc"] = float(roc_auc_score(onehot, y_proba, average="micro", multi_class="ovr"))
    except ValueError:
        data["micro_roc_auc"] = float("nan")
    data["note"] = "One-vs-rest curves; macro curve is the FPR-wise mean of the three class curves."
    return data


def curve_to_frame(data: Dict[str, Any], class_name: str, x_key: str, y_key: str):
    """One class curve as a two-column DataFrame (for the comparison plots)."""
    import pandas as pd

    entry = data["per_class"][class_name]
    return pd.DataFrame({x_key: entry[x_key], y_key: entry[y_key]})


def macro_curve_frame(data: Dict[str, Any], kind: str = "pr"):
    import pandas as pd

    if kind == "pr":
        return pd.DataFrame({"recall": data["macro_recall"], "precision": data["macro_precision"]})
    return pd.DataFrame({"fpr": data["macro_fpr"], "tpr": data["macro_tpr"]})


__all__ = [
    "curve_to_frame",
    "macro_curve_frame",
    "precision_recall_curve_data",
    "roc_curve_data",
]
