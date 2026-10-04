"""All figures of the project.

Every function takes the data it plots from a *file or a logged table* (never
from an in-memory training buffer) so the figures can be regenerated from the
saved artefacts alone:

* ``confusion_matrix*.png``  <- the confusion matrix in ``metrics.json``
* ``val_accuracy_curve.png`` / ``val_loss_curve.png`` <- ``training_log.csv``
* ``pr_curve.png`` / ``roc_curve.png`` <- ``pr_curve_data.json`` / ``roc_curve_data.json``

Class names are always rendered as ``HATE`` / ``OFFENSIVE`` / ``CLEAN``; the raw
ids 0/1/2 are never used as axis labels (plan section 20).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..common.constants import ID2LABEL, LABELS, NUM_LABELS  # noqa: E402
from ..common.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

CLASS_NAMES: Sequence[str] = tuple(LABELS)

#: Fixed colour per class across every figure of the project.
CLASS_COLORS = {"HATE": "#d62728", "OFFENSIVE": "#ff7f0e", "CLEAN": "#2ca02c"}

#: Fixed colour per compared model across every comparison figure.
MODEL_COLORS = {
    "sensitiveai-vi": "#1f77b4",
    "sensitiveai-vi-custom": "#2ca02c",
    "sensitiveai-vi-customdlr2stage": "#d62728",
    "sensitiveai-vi-custom-curriculum": "#9467bd",
}
MODEL_MARKERS = {
    "sensitiveai-vi": "o",
    "sensitiveai-vi-custom": "s",
    "sensitiveai-vi-customdlr2stage": "^",
    "sensitiveai-vi-custom-curriculum": "D",
}
MODEL_ORDER = (
    "sensitiveai-vi",
    "sensitiveai-vi-custom",
    "sensitiveai-vi-customdlr2stage",
    "sensitiveai-vi-custom-curriculum",
)

DPI = 200


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "legend.fontsize": 9,
            "figure.autolayout": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    LOGGER.info("[plot] %s", out)
    return out


def model_color(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, None) or plt.cm.tab10(abs(hash(model_name)) % 10)


def model_label(model_name: str) -> str:
    """Short label used inside comparison legends."""
    return {
        "sensitiveai-vi": "Model 1 (baseline)",
        "sensitiveai-vi-custom": "Model 2 (custom head)",
        "sensitiveai-vi-customdlr2stage": "Model 3 (DLR + 2-stage)",
        "sensitiveai-vi-custom-curriculum": "Model 4 (curriculum)",
    }.get(model_name, model_name)


# --------------------------------------------------------------------------- #
# confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    matrix,
    path: str | Path,
    normalize: bool = False,
    title: Optional[str] = None,
    model_name: str = "",
    label_ids: Sequence[int] = tuple(range(NUM_LABELS)),
) -> Path:
    """3x3 confusion matrix with HATE / OFFENSIVE / CLEAN on both axes."""
    import seaborn as sns

    setup_style()
    cm = np.asarray(matrix, dtype=float)
    names = [ID2LABEL[i] for i in label_ids]

    if normalize:
        totals = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
        fmt, vmax, cbar_label = ".2f", 1.0, "Row-normalised"
    else:
        display = cm
        # counts are stored as float (shared code path with the normalised plot),
        # so annotate with ".0f" instead of "d"
        fmt, vmax, cbar_label = ".0f", max(float(cm.max()), 1.0), "Samples"

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(
        display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xticklabels(names, rotation=0)
    ax.set_yticklabels(names, rotation=0)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title or f"Confusion matrix{' (normalised)' if normalize else ''}\n{model_name}")
    ax.grid(False)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# training curves
# --------------------------------------------------------------------------- #
def _stage_boundaries(log_df) -> list[float]:
    """Epoch positions where a new stage starts (for vertical separators)."""
    if "stage" not in log_df.columns or "epoch" not in log_df.columns:
        return []
    stages = log_df.sort_values("epoch")["stage"].tolist()
    boundaries = []
    previous = None
    for index, (stage, epoch) in enumerate(zip(stages, log_df.sort_values("epoch")["epoch"].tolist())):
        if previous is not None and stage != previous:
            boundaries.append(float(epoch) - 0.5)
        previous = stage
    return boundaries


def _draw_stage_markers(ax: plt.Axes, log_df) -> None:
    boundaries = _stage_boundaries(log_df)
    for boundary in boundaries:
        ax.axvline(boundary, color="grey", linestyle=":", linewidth=1.0, alpha=0.8)
    if len(boundaries):
        ax.plot([], [], color="grey", linestyle=":", linewidth=1.0, label="stage transition")


def plot_validation_curve(
    log_df,
    path: str | Path,
    column: str = "val_accuracy",
    ylabel: str = "Validation Accuracy",
    title: Optional[str] = None,
    model_name: str = "",
    as_percent: bool = True,
) -> Path:
    """Line plot of a validation metric against the (global) epoch number."""
    setup_style()
    if log_df is None or len(log_df) == 0:
        raise ValueError("training_log.csv contains no rows - cannot plot the validation curve")

    log_df = log_df.sort_values("epoch")
    epochs = log_df["epoch"].to_numpy()
    values = log_df[column].to_numpy(dtype=float)
    scale = 100.0 if as_percent else 1.0

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        epochs,
        values * scale,
        marker=MODEL_MARKERS.get(model_name, "o"),
        color=model_color(model_name),
        linewidth=2.0,
        markersize=5,
        label=model_name or column,
    )
    _draw_stage_markers(ax, log_df)

    if "is_best" in log_df.columns:
        best = log_df[log_df["is_best"].astype(str).isin(["1", "1.0", "True", "true"])]
        if len(best):
            ax.scatter(
                best["epoch"], best[column].to_numpy(dtype=float) * scale,
                s=90, facecolors="none", edgecolors="black", linewidths=1.4,
                label="best checkpoint", zorder=5,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel + (" (%)" if as_percent else ""))
    ax.set_title(title or f"{ylabel} per epoch\n{model_name}")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_xticks(epochs)
    fig.tight_layout()
    return _save(fig, path)


def plot_validation_accuracy_curve(log_df, path: str | Path, model_name: str = "", title: Optional[str] = None) -> Path:
    return plot_validation_curve(
        log_df, path, column="val_accuracy", ylabel="Validation Accuracy", model_name=model_name, title=title
    )


def plot_validation_loss_curve(log_df, path: str | Path, model_name: str = "", title: Optional[str] = None) -> Path:
    return plot_validation_curve(
        log_df, path, column="val_loss", ylabel="Validation Loss", model_name=model_name,
        title=title, as_percent=False,
    )


def plot_training_loss_curve(log_df, path: str | Path, model_name: str = "", title: Optional[str] = None) -> Path:
    return plot_validation_curve(
        log_df, path, column="loss", ylabel="Training Loss", model_name=model_name,
        title=title, as_percent=False,
    )


# --------------------------------------------------------------------------- #
# PR / ROC
# --------------------------------------------------------------------------- #
def plot_pr_curve(pr_data: Mapping[str, Any], path: str | Path, model_name: str = "", title: Optional[str] = None) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    for name in pr_data.get("classes", CLASS_NAMES):
        entry = pr_data["per_class"][name]
        ap = entry.get("average_precision", float("nan"))
        ax.plot(
            entry["recall"], entry["precision"],
            color=CLASS_COLORS.get(name), linewidth=1.9,
            label=f"{name} (AP={ap:.4f})",
        )

    ax.plot(
        pr_data["macro_recall"], pr_data["macro_precision"],
        color="black", linestyle="--", linewidth=2.0,
        label=f"Macro average (AP={pr_data.get('macro_average_precision', float('nan')):.4f})",
    )
    micro_ap = pr_data.get("micro_average_precision")
    if micro_ap is not None and not np.isnan(micro_ap):
        ax.plot([], [], " ", label=f"Micro AP={micro_ap:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title or f"One-vs-Rest Precision-Recall curve\n{model_name}")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_curve(roc_data: Mapping[str, Any], path: str | Path, model_name: str = "", title: Optional[str] = None) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    for name in roc_data.get("classes", CLASS_NAMES):
        entry = roc_data["per_class"][name]
        ax.plot(
            entry["fpr"], entry["tpr"],
            color=CLASS_COLORS.get(name), linewidth=1.9,
            label=f"{name} (AUC={entry.get('roc_auc', float('nan')):.4f})",
        )

    ax.plot(
        roc_data["macro_fpr"], roc_data["macro_tpr"],
        color="black", linestyle="--", linewidth=2.0,
        label=f"Macro average (AUC={roc_data.get('macro_roc_auc', float('nan')):.4f})",
    )
    micro = roc_data.get("micro_roc_auc")
    if micro is not None and not np.isnan(micro):
        ax.plot([], [], " ", label=f"Micro AUC={micro:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0, label="chance")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title or f"One-vs-Rest ROC curve\n{model_name}")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# extras
# --------------------------------------------------------------------------- #
def plot_per_class_f1_bars(metrics: Mapping[str, Any], path: str | Path, model_name: str = "") -> Path:
    setup_style()
    names = list(CLASS_NAMES)
    values = [float(metrics.get(f"{n}_f1", float("nan"))) for n in names]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    bars = ax.bar(names, values, color=[CLASS_COLORS[n] for n in names], width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title(f"Per-class F1 on the test set\n{model_name}")
    fig.tight_layout()
    return _save(fig, path)


__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "MODEL_COLORS",
    "MODEL_MARKERS",
    "MODEL_ORDER",
    "model_color",
    "model_label",
    "plot_confusion_matrix",
    "plot_per_class_f1_bars",
    "plot_pr_curve",
    "plot_roc_curve",
    "plot_training_loss_curve",
    "plot_validation_accuracy_curve",
    "plot_validation_curve",
    "plot_validation_loss_curve",
    "setup_style",
]
