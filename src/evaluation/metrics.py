"""Classification metrics, inference helpers and decision rules.

Everything reported in the paper goes through :func:`compute_metrics`, so the
four models are guaranteed to be scored with exactly the same code.

Metric conventions (all class names are printed, never raw ids):

* ``accuracy``, ``macro_precision/recall/f1``, ``weighted_f1``
* per-class precision / recall / f1 / support for HATE, OFFENSIVE, CLEAN
* one-vs-rest ROC-AUC and Average Precision per class + macro / micro averages
* 3x3 confusion matrix (raw and normalised)
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..common.constants import ID2LABEL, LABELS, NUM_LABELS

CLASS_NAMES: List[str] = list(LABELS)


# --------------------------------------------------------------------------- #
# decision rules
# --------------------------------------------------------------------------- #
def predict_argmax(proba: np.ndarray) -> np.ndarray:
    return proba.argmax(axis=1)


def predict_with_thresholds(proba: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Label-specific thresholds, tuned on validation only (plan section 12).

    A label is accepted when its probability reaches its own threshold; the
    highest-probability label that passes wins.  If no label passes, the plain
    argmax is used.  Thresholds are expected in canonical id order
    ``[HATE, OFFENSIVE, CLEAN]``.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape[0] != proba.shape[1]:
        raise ValueError(
            f"thresholds has {thresholds.shape[0]} entries but probabilities have {proba.shape[1]} classes"
        )
    order = np.argsort(-proba, axis=1)
    passes = proba >= thresholds[None, :]
    any_pass = passes.any(axis=1)
    best = order[:, 0].copy()
    for row in range(proba.shape[0]):
        if not any_pass[row]:
            continue
        for col in order[row]:
            if passes[row, col]:
                best[row] = col
                break
    return best


def apply_decision_rule(proba: np.ndarray, decision_rule: str = "argmax", thresholds=None) -> np.ndarray:
    rule = str(decision_rule).lower()
    if rule == "argmax":
        return predict_argmax(proba)
    if rule in {"thresholds", "threshold"}:
        if thresholds is None:
            raise ValueError("decision_rule='thresholds' but no thresholds were provided")
        return predict_with_thresholds(proba, thresholds)
    raise ValueError(f"Unknown decision rule {decision_rule!r} (expected 'argmax' or 'thresholds')")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _safe_roc_auc(y_true_onehot: np.ndarray, proba: np.ndarray) -> float:
    """ROC-AUC that tolerates a class missing from ``y_true`` (returns NaN)."""
    if np.unique(y_true_onehot).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true_onehot, proba))
    except ValueError:  # pragma: no cover
        return float("nan")


def _safe_average_precision(y_true_onehot: np.ndarray, proba: np.ndarray) -> float:
    if y_true_onehot.sum() == 0:
        return float("nan")
    try:
        return float(average_precision_score(y_true_onehot, proba))
    except ValueError:  # pragma: no cover
        return float("nan")


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Sequence[str] = tuple(CLASS_NAMES),
    label_ids: Sequence[int] = tuple(range(NUM_LABELS)),
) -> Dict[str, object]:
    """Full metric bundle. ``y_proba`` is required for ROC-AUC / AP."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ids = list(label_ids)
    names = list(labels)

    out: Dict[str, object] = {}

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    out["accuracy"] = float(accuracy_score(y_true, y_pred))

    for i, name in zip(ids, names):
        out[f"{name}_precision"] = float(precision[i])
        out[f"{name}_recall"] = float(recall[i])
        out[f"{name}_f1"] = float(f1[i])
        out[f"{name}_support"] = int(support[i])

    out["macro_precision"] = float(precision.mean())
    out["macro_recall"] = float(recall.mean())
    out["macro_f1"] = float(f1.mean())
    out["weighted_f1"] = float((f1 * support).sum() / max(support.sum(), 1))

    cm = confusion_matrix(y_true, y_pred, labels=ids)
    out["confusion_matrix"] = cm.astype(int).tolist()
    out["confusion_matrix_normalized"] = _normalize_confusion(cm).tolist()

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        onehot = np.eye(len(ids))[y_true]
        aucs, aps = [], []
        for i, name in zip(ids, names):
            auc = _safe_roc_auc(onehot[:, i], y_proba[:, i])
            ap = _safe_average_precision(onehot[:, i], y_proba[:, i])
            out[f"{name}_roc_auc"] = auc
            out[f"{name}_average_precision"] = ap
            if not np.isnan(auc):
                aucs.append(auc)
            if not np.isnan(ap):
                aps.append(ap)
        out["macro_roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
        out["macro_average_precision"] = float(np.mean(aps)) if aps else float("nan")

        try:
            out["micro_roc_auc"] = float(roc_auc_score(onehot, y_proba, average="micro", multi_class="ovr"))
        except ValueError:
            out["micro_roc_auc"] = float("nan")
        try:
            out["micro_average_precision"] = float(
                average_precision_score(onehot, y_proba, average="micro")
            )
        except ValueError:
            out["micro_average_precision"] = float("nan")

    return out


def _normalize_confusion(cm: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix (rows sum to 1, 0 kept as 0)."""
    cm = cm.astype(float)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)


def classification_report_text(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[str] = tuple(CLASS_NAMES),
    label_ids: Sequence[int] = tuple(range(NUM_LABELS)),
    digits: int = 4,
) -> str:
    """Human-readable report using the *label names* instead of 0/1/2."""
    from sklearn.metrics import classification_report

    target_names = [ID2LABEL[i] for i in label_ids]
    return classification_report(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        labels=list(label_ids),
        target_names=target_names,
        digits=digits,
        zero_division=0,
    )


def confusion_matrix_text(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[str] = tuple(CLASS_NAMES),
    label_ids: Sequence[int] = tuple(range(NUM_LABELS)),
) -> str:
    """Confusion matrix printed with HATE / OFFENSIVE / CLEAN as row/column names."""
    names = [ID2LABEL[i] for i in label_ids]
    cm = confusion_matrix(np.asarray(y_true, int), np.asarray(y_pred, int), labels=list(label_ids))
    width = max(len(n) for n in names) + 2
    header = " " * (width + 8) + "".join(f"{n:>{width}}" for n in names)
    lines = ["Confusion matrix (rows = actual, cols = predicted)", header]
    for name, row in zip(names, cm):
        lines.append(f"{name:<{width}}" + "".join(f"{v:>{width}d}" for v in row))
    lines.append("")
    lines.append("Row-normalised")
    norm = _normalize_confusion(cm)
    header = " " * (width + 8) + "".join(f"{n:>{width}}" for n in names)
    lines.append(header)
    for name, row in zip(names, norm):
        lines.append(f"{name:<{width}}" + "".join(f"{v:>{width}.4f}" for v in row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def move_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    return_logits: bool = False,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, np.ndarray]:
    """Softmax probabilities + labels for a whole dataloader (no shuffling).

    When ``criterion`` is given the loss is accumulated during the *same* pass,
    so the reported test loss costs no extra inference.
    """
    model.eval()
    all_proba: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    loss_sum = 0.0
    n_items = 0

    for batch in dataloader:
        labels = batch["labels"]
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits = model(**inputs).logits.float()
        else:
            logits = model(**inputs).logits.float()
        all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if return_logits:
            all_logits.append(logits.cpu().numpy())
        if criterion is not None:
            target = labels.to(device, non_blocking=True)
            loss_sum += float(criterion(logits, target)) * len(target)
            n_items += len(target)

    out = {
        "y_true": np.concatenate(all_labels) if all_labels else np.empty((0, NUM_LABELS), dtype=int),
        "y_proba": np.concatenate(all_proba) if all_proba else np.empty((0, NUM_LABELS)),
    }
    if return_logits:
        out["y_logits"] = np.concatenate(all_logits) if all_logits else np.empty((0, NUM_LABELS))
    if criterion is not None:
        out["loss"] = loss_sum / n_items if n_items else float("nan")
    return out


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    decision_rule: str = "argmax",
    thresholds: Optional[Sequence[float]] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, object]:
    """Validation-time evaluation used by the training loop and the evaluator."""
    outputs = run_inference(model, dataloader, device, amp_dtype=amp_dtype, return_logits=True)
    y_true, y_proba, y_logits = outputs["y_true"], outputs["y_proba"], outputs["y_logits"]

    if criterion is not None:
        # the criterion may carry device buffers (class weights / alpha) -> match its device
        target_device = next(
            (b.device for b in criterion.buffers() if b is not None),
            torch.device("cpu"),
        )
        loss = float(
            criterion(
                torch.from_numpy(y_logits).to(target_device),
                torch.from_numpy(y_true).to(target_device),
            )
        )
    else:
        loss = float("nan")

    y_pred = apply_decision_rule(y_proba, decision_rule, thresholds)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["loss"] = loss
    metrics["num_samples"] = int(len(y_true))
    return metrics


def predictions_dataframe(
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: np.ndarray,
    indices: Optional[Sequence[int]] = None,
):
    """``test_predictions.csv`` content (one row per test sample)."""
    import pandas as pd

    data = {
        "index": list(indices) if indices is not None else list(range(len(y_true))),
        "text": list(texts),
        "true_label_id": list(map(int, y_true)),
        "true_label": [ID2LABEL[int(i)] for i in y_true],
        "pred_label_id": list(map(int, y_pred)),
        "pred_label": [ID2LABEL[int(i)] for i in y_pred],
        "correct": [int(a == b) for a, b in zip(y_true, y_pred)],
    }
    for i, name in enumerate(CLASS_NAMES):
        data[f"prob_{name}"] = y_proba[:, i]
    return pd.DataFrame(data)


__all__ = [
    "CLASS_NAMES",
    "apply_decision_rule",
    "classification_report_text",
    "compute_metrics",
    "confusion_matrix_text",
    "evaluate_model",
    "predict_argmax",
    "predict_with_thresholds",
    "predictions_dataframe",
    "run_inference",
]
