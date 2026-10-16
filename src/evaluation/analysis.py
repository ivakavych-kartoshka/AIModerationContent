"""Post-hoc statistical analysis of the frozen test predictions.

Everything in this module reads ``test_predictions.csv`` (written once per model
by the evaluation step) or the saved ``benchmark.json`` / ``training_summary.json``
files, so every number can be recomputed without retraining:

* bootstrap 95% confidence intervals for accuracy, macro-F1 and per-class F1;
* paired bootstrap difference and the exact McNemar test for model pairs;
* expected calibration error (ECE) with a reliability summary;
* 3x3 confusion matrices;
* an error analysis stratified by surface phenomena of Vietnamese social media
  text (very short / very long comments, teencode, missing diacritics, emoji).

The heavy loops avoid ``sklearn`` because a percentile bootstrap calls the metric
tens of thousands of times; the confusion matrix is accumulated with
``np.bincount`` and the metrics are derived from it, which is ~50x faster and
numerically identical to ``precision_recall_fscore_support`` for classes that
appear in the data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from ..common.constants import LABELS, NUM_LABELS
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Problem-agnostic short labels shared with the paper's tables and figures.
MODEL_LABELS: Dict[str, str] = {
    "sensitiveai-vi": "M1",
    "sensitiveai-vi-custom": "M2",
    "sensitiveai-vi-customdlr2stage": "M3",
    "sensitiveai-vi-custom-curriculum": "M4",
    "sensitiveai-vi-customdlr2stage-ext": "M5",
    "sensitiveai-vi-stage2": "M6",
}

MODEL_ORDER: Sequence[str] = tuple(MODEL_LABELS)

#: Regex probes for common surface phenomena in Vietnamese user-generated text.
TEENCODE_RE = re.compile(
    r"(?i)(\b[a-z]*[0-9]+[a-z0-9]*\b)"          # m, j, z, 9x, 2k ...
    r"|(\b(?:ak|mik|mk|ck|dc|đc|đk|ntn|ns|cmnr|vl|vcl|dkm|chịu)\b)"
)
REPEAT_RE = re.compile(r"(.)\1{2,}")             # 3+ repeated characters
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F0FF" "]",
    flags=re.UNICODE,
)
DIACRITIC_RE = re.compile(
    "["
    "ăâđêôơư"
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
    "]"
)


# --------------------------------------------------------------------------- #
# fast metrics from a confusion matrix
# --------------------------------------------------------------------------- #
def confusion_from_labels(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int = NUM_LABELS) -> np.ndarray:
    """3x3 confusion matrix ``cm[true, pred]`` accumulated with ``bincount``."""
    idx = y_true.astype(np.int64) * num_labels + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=num_labels * num_labels).reshape(num_labels, num_labels).astype(float)


def _prf_from_confusion(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    prf = _prf_from_confusion(cm)
    total = cm.sum()
    out = {
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "macro_f1": float(prf["f1"].mean()),
        "macro_precision": float(prf["precision"].mean()),
        "macro_recall": float(prf["recall"].mean()),
    }
    for i, name in enumerate(LABELS):
        out[f"{name}_f1"] = float(prf["f1"][i])
        out[f"{name}_precision"] = float(prf["precision"][i])
        out[f"{name}_recall"] = float(prf["recall"][i])
    return out


_METRIC_KEYS = {"accuracy", "macro_f1", "macro_precision", "macro_recall"} | {
    f"{name}_{m}" for name in LABELS for m in ("f1", "precision", "recall")
}


def _metric_from_confusion(cm: np.ndarray, metric: str) -> float:
    return metrics_from_confusion(cm)[metric]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_predictions(
    models: Sequence[str] = MODEL_ORDER,
    per_model_root: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load and align the six ``test_predictions.csv`` files.

    The files are checked to contain the same ``index`` order and the same
    ``true_label_id`` column; a mismatch raises, because every paired test would
    otherwise be meaningless.
    """
    root = Path(per_model_root) if per_model_root else Path("outputs/evaluation/per_model")
    frames: Dict[str, pd.DataFrame] = {}
    reference: Optional[pd.DataFrame] = None
    for name in models:
        path = root / name / "test_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing predictions for {name}: {path}")
        frame = pd.read_csv(path, encoding="utf-8")
        if reference is None:
            reference = frame
        else:
            if not np.array_equal(frame["index"].to_numpy(), reference["index"].to_numpy()):
                raise ValueError(f"{name} predictions are not aligned (index differs)")
            if not np.array_equal(frame["true_label_id"].to_numpy(), reference["true_label_id"].to_numpy()):
                raise ValueError(f"{name} predictions are not aligned (true labels differ)")
        frames[name] = frame
    LOGGER.info("[analysis] loaded %d aligned prediction files (%d rows)", len(frames), len(reference))
    return frames


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI for one metric (resampling the test set)."""
    if metric not in _METRIC_KEYS:
        raise ValueError(f"unknown metric {metric!r}")
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    point = _metric_from_confusion(confusion_from_labels(y_true, y_pred), metric)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        take = rng.integers(0, n, size=n)
        samples[b] = _metric_from_confusion(confusion_from_labels(y_true[take], y_pred[take]), metric)
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(samples.std(ddof=1)),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """CI and two-sided bootstrap p-value for ``metric(A) - metric(B)``.

    The same resampled indices are used for both systems, so the interval is the
    paired (correlated) one; the p-value is the fraction of resamples on the
    opposite side of zero, doubled, which is the standard percentile p-value.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    point = _metric_from_confusion(confusion_from_labels(y_true, pred_a), metric) - _metric_from_confusion(
        confusion_from_labels(y_true, pred_b), metric
    )
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        take = rng.integers(0, n, size=n)
        yt = y_true[take]
        deltas[b] = _metric_from_confusion(confusion_from_labels(yt, pred_a[take]), metric) - _metric_from_confusion(
            confusion_from_labels(yt, pred_b[take]), metric
        )
    lo, hi = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    if np.mean(deltas <= 0) < np.mean(deltas >= 0):
        p = 2.0 * np.mean(deltas <= 0)
    else:
        p = 2.0 * np.mean(deltas >= 0)
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p)),
        "n_boot": int(n_boot),
    }


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Exact two-sided McNemar test on the two systems' correctness vectors."""
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n = b + c
    exact = binomtest(min(b, c), n, 0.5, alternative="two-sided") if n > 0 else None
    # continuity-corrected chi-square as a cross-check (df = 1)
    if n > 0:
        stat = (abs(b - c) - 1) ** 2 / n
        p_chi = float(chi2.sf(stat, 1))
    else:
        stat = 0.0
        p_chi = 1.0
    return {
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "discordant": n,
        "p_value": float(exact.pvalue) if exact is not None else 1.0,
        "p_value_chi2": p_chi,
        "odds_ratio": (b / c) if c else float("inf"),
    }


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Top-label ECE and a reliability table.

    Confidence is the maximum softmax probability and correctness is whether the
    argmax matches the gold label, matching the confidence-based (not class-wise)
    definition commonly reported for classifiers.
    """
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=np.int64)
    confidence = proba.max(axis=1)
    prediction = proba.argmax(axis=1)
    correctness = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: List[Dict[str, float]] = []
    n = proba.shape[0]
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"low": float(lo), "high": float(hi), "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correctness[mask].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)
        bins.append(
            {
                "low": float(lo),
                "high": float(hi),
                "count": count,
                "confidence": avg_conf,
                "accuracy": avg_acc,
            }
        )
    return {"ece": float(ece), "n_bins": int(n_bins), "bins": bins}


# --------------------------------------------------------------------------- #
# error stratification
# --------------------------------------------------------------------------- #
def _strata(text: str) -> List[str]:
    tags = []
    stripped = DIACRITIC_RE.sub("", text)
    if len(stripped) != len(text):
        tags.append("with_diacritics")
    if text.strip():
        if not DIACRITIC_RE.search(text) and len(text) >= 8:
            tags.append("no_diacritics")
    if TEENCODE_RE.search(text):
        tags.append("teencode")
    if REPEAT_RE.search(text):
        tags.append("repeat_chars")
    if EMOJI_RE.search(text):
        tags.append("emoji")
    if len(text) <= 20:
        tags.append("very_short")
    if len(text) >= 200:
        tags.append("very_long")
    return tags


def error_stratification(frame: pd.DataFrame) -> Dict[str, Any]:
    """Recall / support per surface stratum for one system's predictions."""
    y_true = frame["true_label_id"].to_numpy()
    y_pred = frame["pred_label_id"].to_numpy()
    texts = frame["text"].fillna("").astype(str).tolist()
    tag_sets = [_strata(t) for t in texts]
    all_tags = sorted({tag for tags in tag_sets for tag in tags})
    out: Dict[str, Any] = {}
    for tag in all_tags:
        mask = np.array([tag in tags for tags in tag_sets])
        n = int(mask.sum())
        if n == 0:
            continue
        per_class = {}
        for i, name in enumerate(LABELS):
            sel = mask & (y_true == i)
            support = int(sel.sum())
            recall = float(np.mean(y_pred[sel] == i)) if support else float("nan")
            per_class[name] = {"support": support, "recall": recall}
        macro_f1 = _metric_from_confusion(confusion_from_labels(y_true[mask], y_pred[mask]), "macro_f1")
        accuracy = float(np.mean(y_true[mask] == y_pred[mask]))
        out[tag] = {
            "count": n,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class,
        }
    return out


# --------------------------------------------------------------------------- #
# top-level driver
# --------------------------------------------------------------------------- #
def analyse(
    models: Sequence[str] = MODEL_ORDER,
    per_model_root: Optional[Path] = None,
    pairs: Optional[Sequence[tuple]] = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run every analysis and return a JSON-serialisable dictionary."""
    frames = load_predictions(models, per_model_root)
    reference = frames[models[0]]
    y_true = reference["true_label_id"].to_numpy()

    if pairs is None:
        pairs = (
            ("sensitiveai-vi-customdlr2stage", "sensitiveai-vi"),
            ("sensitiveai-vi-customdlr2stage", "sensitiveai-vi-custom"),
            ("sensitiveai-vi-customdlr2stage", "sensitiveai-vi-custom-curriculum"),
            ("sensitiveai-vi-customdlr2stage", "sensitiveai-vi-stage2"),
            ("sensitiveai-vi-customdlr2stage", "sensitiveai-vi-customdlr2stage-ext"),
            ("sensitiveai-vi-customdlr2stage-ext", "sensitiveai-vi"),
            ("sensitiveai-vi-stage2", "sensitiveai-vi"),
        )

    confidence_models = {}
    for name in models:
        frame = frames[name]
        y_pred = frame["pred_label_id"].to_numpy()
        proba = frame[["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]].to_numpy()
        confidence_models[name] = {
            "label": MODEL_LABELS[name],
            "accuracy": bootstrap_ci(y_true, y_pred, "accuracy", n_boot=n_boot, seed=seed),
            "macro_f1": bootstrap_ci(y_true, y_pred, "macro_f1", n_boot=n_boot, seed=seed),
            "HATE_f1": bootstrap_ci(y_true, y_pred, "HATE_f1", n_boot=n_boot, seed=seed),
            "OFFENSIVE_f1": bootstrap_ci(y_true, y_pred, "OFFENSIVE_f1", n_boot=n_boot, seed=seed),
            "CLEAN_f1": bootstrap_ci(y_true, y_pred, "CLEAN_f1", n_boot=n_boot, seed=seed),
            "calibration": expected_calibration_error(proba, y_true, n_bins=10),
            "confusion_matrix": confusion_from_labels(y_true, y_pred).astype(int).tolist(),
        }

    pair_results = []
    for a, b in pairs:
        pred_a = frames[a]["pred_label_id"].to_numpy()
        pred_b = frames[b]["pred_label_id"].to_numpy()
        delta = paired_bootstrap_delta(y_true, pred_a, pred_b, "macro_f1", n_boot=n_boot, seed=seed)
        delta_acc = paired_bootstrap_delta(y_true, pred_a, pred_b, "accuracy", n_boot=n_boot, seed=seed)
        mcn = mcnemar(y_true, pred_a, pred_b)
        pair_results.append(
            {
                "a": a,
                "b": b,
                "label_a": MODEL_LABELS[a],
                "label_b": MODEL_LABELS[b],
                "macro_f1": delta,
                "accuracy": delta_acc,
                "mcnemar": mcn,
            }
        )

    error_profiles = {
        name: error_stratification(frames[name]) for name in models
    }

    return {
        "seed": int(seed),
        "n_boot": int(n_boot),
        "num_test": int(y_true.shape[0]),
        "models": list(models),
        "confidence": confidence_models,
        "pairs": pair_results,
        "error_stratification": error_profiles,
    }


def save_analysis(payload: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[analysis] saved -> %s", out)
    return out


__all__ = [
    "MODEL_LABELS",
    "MODEL_ORDER",
    "analyse",
    "bootstrap_ci",
    "confusion_from_labels",
    "error_stratification",
    "expected_calibration_error",
    "load_predictions",
    "mcnemar",
    "metrics_from_confusion",
    "paired_bootstrap_delta",
    "save_analysis",
]
