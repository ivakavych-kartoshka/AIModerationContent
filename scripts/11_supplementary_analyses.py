"""Step 11 - supplementary analyses for the paper (no retraining).

    python scripts/11_supplementary_analyses.py

The twelve-page main text cannot hold every number, so this script recomputes
additional diagnostics from the *frozen* artefacts already on disk
(``test_predictions.csv``, ``metrics.json``, ``training_summary.json`` and the
``configs/*.yaml``) and writes a self-contained supplementary bundle:

    outputs/evaluation/analysis/supplementary_summary.json
    outputs/evaluation/analysis/supplementary_report.md
    papers/supplementary/tables/*.tex
    papers/supplementary/figures/*.png

Nothing here retrains or re-scores a model; every table is a re-reading of the
once-frozen test pass, so running this script is deterministic.

New analyses (relative to the main text)
----------------------------------------
* per-class precision/recall/F1 for all six systems (one table);
* per-class one-vs-rest ROC-AUC and average precision (one table);
* calibration diagnostics: ECE, multiclass Brier score and negative log-likelihood
  plus reliability diagrams;
* the complete 6x6 pairwise significance matrix (all 15 pairs, Holm-corrected);
* a cross-system error-overlap / complementarity analysis (how many systems miss
  the same comment, which comments are missed by all six, which are unique errors);
* a post-hoc class-prior reweighting diagnostic, to test whether a trivial
  reweighting of the frozen posteriors recovers the tuned recipes' macro-F1;
* error stratification by surface phenomenon for *all* six systems;
* the resolved hyperparameters and the wall-clock cost of every run.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_yaml, resolve_config_path  # noqa: E402
from src.common.constants import LABELS, NUM_LABELS  # noqa: E402
from src.evaluation.analysis import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
    confusion_from_labels,
    load_predictions,
    mcnemar,
    metrics_from_confusion,
    paired_bootstrap_delta,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PER_MODEL_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "per_model"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "analysis"
SUPP_DIR = PROJECT_ROOT / "papers" / "supplementary"
SUPP_TABLES = SUPP_DIR / "tables"
SUPP_FIGURES = SUPP_DIR / "figures"
SUPP_SECTIONS = SUPP_DIR / "sections"

#: Source figure directories (all-model comparison plots + per-model plots).
PAPERS_FIGURES = PROJECT_ROOT / "papers" / "figures"
COMPARISON_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "comparison"

#: Where the copied figures land inside the supplementary bundle.
SUPP_COMPARISON_FIG = SUPP_FIGURES / "comparison"
SUPP_PER_MODEL_FIG = SUPP_FIGURES / "per_model"

#: The all-model comparison plots to copy, in the order they are shown.
COMPARISON_FIGS: Tuple[str, ...] = (
    "accuracy_all_models.png",
    "macro_f1_all_models.png",
    "per_class_f1_all_models.png",
    "f1_summary_all_models.png",
    "validation_loss_all_models.png",
    "validation_accuracy_all_models.png",
    "validation_loss_accuracy_all_models.png",
    "pr_curve_all_models.png",
    "roc_curve_all_models.png",
    "pr_hate_all_models.png",
    "pr_offensive_all_models.png",
    "pr_clean_all_models.png",
    "roc_hate_all_models.png",
    "roc_offensive_all_models.png",
    "roc_clean_all_models.png",
)

#: Per-model plots to copy, in the order they are shown inside each system.
PER_MODEL_FIGS: Tuple[str, ...] = (
    "confusion_matrix_normalized.png",
    "pr_curve.png",
    "roc_curve.png",
    "val_loss_curve.png",
    "val_accuracy_curve.png",
)

#: The main paper's statistics live here; the supplementary re-reads them.
SIG_PATH = ANALYSIS_DIR / "significance_summary.json"

PROBA_COLS = ["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]
N_BOOT = 10000
SEED = 42


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _fmt_p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def _holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment (monotone, clipped to 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def _table(lines: List[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_metrics() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in MODEL_ORDER:
        payload = json.loads((PER_MODEL_DIR / name / "metrics.json").read_text(encoding="utf-8"))
        out[name] = payload
    return out


def _load_training() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in MODEL_ORDER:
        payload = json.loads((PER_MODEL_DIR / name / "training_summary.json").read_text(encoding="utf-8"))
        out[name] = payload
    return out


# --------------------------------------------------------------------------- #
# new metrics
# --------------------------------------------------------------------------- #
def brier_score(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score = mean_k (1/N) sum_i (p_ik - onehot_ik)^2."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def negative_log_likelihood(proba: np.ndarray, y_true: np.ndarray) -> float:
    picked = proba[np.arange(len(y_true)), y_true]
    picked = np.clip(picked, 1e-12, 1.0)
    return float(-np.mean(np.log(picked)))


def reliability_bins(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    confidence = proba.max(axis=1)
    prediction = proba.argmax(axis=1)
    correctness = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        bins.append(
            {
                "low": float(lo),
                "high": float(hi),
                "count": count,
                "confidence": float(confidence[mask].mean()) if count else 0.0,
                "accuracy": float(correctness[mask].mean()) if count else 0.0,
            }
        )
    return bins


def error_overlap(frames: Dict[str, pd.DataFrame], y_true: np.ndarray) -> Dict[str, Any]:
    """Cross-system error complementarity from the boolean 'wrong' masks."""
    names = [n for n in MODEL_ORDER if n in frames]
    wrong = np.stack([frames[n]["pred_label_id"].to_numpy() != y_true for n in names])
    n_models = len(names)
    n_items = int(y_true.shape[0])
    per_item = wrong.sum(axis=0)

    k_distribution = {int(k): int(np.sum(per_item == k)) for k in range(n_models + 1)}
    unique_error = {MODEL_LABELS[n]: int(np.sum(wrong[i] & (per_item == 1))) for i, n in enumerate(names)}
    all_wrong = per_item == n_models
    all_correct = per_item == 0
    missed_by_all = {
        "count": int(all_wrong.sum()),
        "true_label_distribution": {
            LABELS[c]: int(np.sum(all_wrong & (y_true == c))) for c in range(NUM_LABELS)
        },
    }
    solved_by_all = {
        "count": int(all_correct.sum()),
        "true_label_distribution": {
            LABELS[c]: int(np.sum(all_correct & (y_true == c))) for c in range(NUM_LABELS)
        },
    }
    # pairwise Jaccard similarity of the error sets (1.0 = identical errors)
    jaccard: Dict[str, Dict[str, float]] = {MODEL_LABELS[n]: {} for n in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            inter = int(np.sum(wrong[i] & wrong[j]))
            union = int(np.sum(wrong[i] | wrong[j]))
            jaccard[MODEL_LABELS[a]][MODEL_LABELS[b]] = float(inter / union) if union else 1.0

    return {
        "n_models": n_models,
        "n_items": n_items,
        "k_distribution": k_distribution,
        "unique_errors": unique_error,
        "missed_by_all_six": missed_by_all,
        "solved_by_all_six": solved_by_all,
        "error_jaccard": jaccard,
    }


def prior_reweighting_sweep(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    factors: np.ndarray,
) -> Dict[str, Any]:
    """Diagnostic: multiply P(HATE),P(OFF.) by a joint factor u, renormalise, argmax.

    u > 1 pushes the decision boundary towards the minority classes.  The sweep is
    computed on the *test* split purely to show how much a trivial, class-agnostic
    recalibration of the frozen posteriors could move macro-F1; the value of u is
    therefore an oracle upper bound, not a validated hyper-parameter.
    """
    proba = frame[PROBA_COLS].to_numpy(dtype=float)
    macro_f1: List[float] = []
    for u in factors:
        adj = proba.copy()
        adj[:, 0] *= u  # HATE
        adj[:, 1] *= u  # OFFENSIVE
        adj /= adj.sum(axis=1, keepdims=True)
        pred = adj.argmax(axis=1)
        macro_f1.append(metrics_from_confusion(confusion_from_labels(y_true, pred))["macro_f1"])
    macro_f1_arr = np.asarray(macro_f1)
    best_idx = int(np.argmax(macro_f1_arr))
    return {
        "factors": factors.tolist(),
        "macro_f1": macro_f1_arr.tolist(),
        "best_factor": float(factors[best_idx]),
        "best_macro_f1": float(macro_f1_arr[best_idx]),
        "argmax_macro_f1": float(macro_f1_arr[np.argmin(np.abs(factors - 1.0))]),
        "gain": float(macro_f1_arr[best_idx] - macro_f1_arr[np.argmin(np.abs(factors - 1.0))]),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_confusion_grid(metrics: Dict[str, Dict[str, Any]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.plots import setup_style

    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.2), constrained_layout=True)
    short = ["HATE", "OFF.", "CLEAN"]
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        cm = np.asarray(metrics[name]["confusion_matrix_normalized"], dtype=float)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for t in range(3):
            for p in range(3):
                ax.text(p, t, f"{cm[t, p]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if cm[t, p] > 0.55 else "black")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(short, fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(f"{MODEL_LABELS[name]} ({name})", fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="row-normalised")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_reliability(calibration: Dict[str, Dict[str, Any]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.plots import MODEL_COLORS, setup_style

    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.0), constrained_layout=True)
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        bins = calibration[name]["bins"]
        xs = [b["confidence"] for b in bins if b["count"]]
        ys = [b["accuracy"] for b in bins if b["count"]]
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0)
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS.get(name, "#1f77b4"), linewidth=1.6, markersize=4)
        ax.set_title(f"{MODEL_LABELS[name]} (ECE={calibration[name]['ece']:.3f})", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
    fig.suptitle("Top-label reliability (10 bins)", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_error_overlap(overlap: Dict[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.plots import setup_style

    setup_style()
    k = list(range(overlap["n_models"] + 1))
    counts = [overlap["k_distribution"][i] for i in k]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar([str(i) for i in k], counts, color="#4c72b0", width=0.62)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{value:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Number of systems that misclassify the comment")
    ax.set_ylabel("Test comments")
    ax.set_title("Complementarity of the six systems' errors")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_threshold_sweep(sweeps: Dict[str, Dict[str, Any]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.plots import MODEL_COLORS, setup_style

    setup_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name in MODEL_ORDER:
        sweep = sweeps[name]
        ax.plot(sweep["factors"], sweep["macro_f1"], color=MODEL_COLORS.get(name, "#333333"),
                linewidth=1.7, label=MODEL_LABELS[name])
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Joint minority-class posterior factor $u$")
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title("Post-hoc prior-reweighting diagnostic (oracle sweep on test)")
    ax.legend(fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
def table_per_class(metrics: Dict[str, Dict[str, Any]], path: Path) -> Path:
    best = {}
    for key in ("HATE_f1", "OFFENSIVE_f1", "CLEAN_f1", "macro_f1"):
        best[key] = max(metrics[n][key] for n in MODEL_ORDER)

    def cell(name: str, key: str) -> str:
        value = metrics[name][key]
        text = _fmt(value)
        return f"\\textbf{{{text}}}" if abs(value - best[key]) < 1e-12 else text

    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\caption{Per-class precision, recall and F1 on the ViHSD test set for all six "
        "systems, with the macro-F1.  Best per column in bold.}",
        "\\label{tab:supp-per-class}",
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        "System & \\multicolumn{3}{c}{HATE} & \\multicolumn{3}{c}{OFFENSIVE} & "
        "\\multicolumn{3}{c}{CLEAN} & M-F1 \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        " & P & R & F1 & P & R & F1 & P & R & F1 & \\\\",
        "\\midrule",
    ]
    for name in MODEL_ORDER:
        m = metrics[name]
        lines.append(
            f"{MODEL_LABELS[name]} & {_fmt(m['HATE_precision'])} & {_fmt(m['HATE_recall'])} "
            f"& {cell(name, 'HATE_f1')} & {_fmt(m['OFFENSIVE_precision'])} & {_fmt(m['OFFENSIVE_recall'])} "
            f"& {cell(name, 'OFFENSIVE_f1')} & {_fmt(m['CLEAN_precision'])} & {_fmt(m['CLEAN_recall'])} "
            f"& {cell(name, 'CLEAN_f1')} & {cell(name, 'macro_f1')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_ranking(metrics: Dict[str, Dict[str, Any]], path: Path) -> Path:
    best = {
        "macro_roc_auc": max(metrics[n]["macro_roc_auc"] for n in MODEL_ORDER),
        "macro_average_precision": max(metrics[n]["macro_average_precision"] for n in MODEL_ORDER),
    }

    def cell(name: str, key: str) -> str:
        value = metrics[name][key]
        text = _fmt(value)
        return f"\\textbf{{{text}}}" if abs(value - best[key]) < 1e-12 else text

    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{One-vs-rest ranking quality per class (ROC-AUC / average precision) "
        "and the macro averages, for all six systems.  Best macro value in bold.}",
        "\\label{tab:supp-ranking}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{HATE} & \\multicolumn{2}{c}{OFFENSIVE} & "
        "\\multicolumn{2}{c}{CLEAN} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        "System & AUC & AP & AUC & AP & AUC & AP \\\\",
        "\\midrule",
    ]
    for name in MODEL_ORDER:
        m = metrics[name]
        lines.append(
            f"{MODEL_LABELS[name]} & {_fmt(m['HATE_roc_auc'])} & {_fmt(m['HATE_average_precision'])} "
            f"& {_fmt(m['OFFENSIVE_roc_auc'])} & {_fmt(m['OFFENSIVE_average_precision'])} "
            f"& {_fmt(m['CLEAN_roc_auc'])} & {_fmt(m['CLEAN_average_precision'])} \\\\"
        )
    lines += [
        "\\midrule",
        "System & \\multicolumn{2}{c}{macro AUC / AP} & \\multicolumn{2}{c}{micro AUC / AP} "
        "& \\multicolumn{2}{c}{MCC} \\\\",
        "\\midrule",
    ]
    for name in MODEL_ORDER:
        m = metrics[name]
        lines.append(
            f"{MODEL_LABELS[name]} & \\multicolumn{{2}}{{c}}{{{cell(name, 'macro_roc_auc')} / "
            f"{cell(name, 'macro_average_precision')}}} & \\multicolumn{{2}}{{c}}"
            f"{{{_fmt(m['micro_roc_auc'])} / {_fmt(m['micro_average_precision'])}}} "
            f"& \\multicolumn{{2}}{{c}}{{{_fmt(m['mcc'])}}} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_calibration(cal: Dict[str, Dict[str, float]], path: Path) -> Path:
    best_ece = min(cal[n]["ece"] for n in MODEL_ORDER)
    best_brier = min(cal[n]["brier"] for n in MODEL_ORDER)
    best_nll = min(cal[n]["nll"] for n in MODEL_ORDER)

    def cell(name: str, key: str, best: float) -> str:
        text = _fmt(cal[name][key], 3)
        return f"\\textbf{{{text}}}" if abs(cal[name][key] - best) < 1e-12 else text

    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Calibration of the frozen posteriors on the test set: top-label "
        "expected calibration error (ECE, 10 bins), multiclass Brier score and "
        "negative log-likelihood (lower is better).  Best per column in bold.}",
        "\\label{tab:supp-calibration}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "System & ECE $\\downarrow$ & Brier $\\downarrow$ & NLL $\\downarrow$ & mean conf. \\\\",
        "\\midrule",
    ]
    for name in MODEL_ORDER:
        c = cal[name]
        lines.append(
            f"{MODEL_LABELS[name]} & {cell(name, 'ece', best_ece)} "
            f"& {cell(name, 'brier', best_brier)} & {cell(name, 'nll', best_nll)} "
            f"& {_fmt(c['mean_confidence'], 3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_confusion_counts(metrics: Dict[str, Dict[str, Any]], path: Path) -> Path:
    order = [("HATE", 0, "H"), ("OFFENSIVE", 1, "O"), ("CLEAN", 2, "C")]
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Raw test-set confusion counts for all six systems over the 6{,}680 test "
        "comments.  The three column blocks are the \\emph{true} class and the sub-columns "
        "the \\emph{predicted} class (H = HATE, O = OFFENSIVE, C = CLEAN); each row "
        "therefore sums to 6{,}680.}",
        "\\label{tab:supp-confusion}",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{true HATE} & \\multicolumn{3}{c}{true OFF.} "
        "& \\multicolumn{3}{c}{true CLEAN} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "System & H & O & C & H & O & C & H & O & C \\\\",
        "\\midrule",
    ]
    for name in MODEL_ORDER:
        cm = metrics[name]["confusion_matrix"]
        vals = []
        for _, true_idx, _ in order:
            for _, pred_idx, _ in order:
                vals.append(cm[true_idx][pred_idx])
        cells = " & ".join(f"{v:,}" for v in vals)
        lines.append(f"{MODEL_LABELS[name]} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_error_overlap(overlap: Dict[str, Any], path: Path) -> Path:
    def breakdown(dist: Dict[str, int]) -> str:
        return ", ".join(f"{label} {cnt}" for label, cnt in dist.items())

    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Error complementarity across the six systems.  \\emph{Top:} how many "
        "test comments are misclassified by exactly $k$ systems.  \\emph{Bottom:} the "
        "comments no system gets right, broken down by their gold label.}",
        "\\label{tab:supp-error-overlap}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\# systems wrong & 0 & 1 & 2 & 3 & 4 & 5 & 6 \\\\",
        "\\midrule",
        "test comments & "
        + " & ".join(f"{overlap['k_distribution'][k]:,}" for k in range(7))
        + " \\\\",
        "\\midrule",
        "\\multicolumn{8}{l}{\\emph{missed by all six:} "
        + f"{overlap['missed_by_all_six']['count']} comments ("
        + breakdown(overlap["missed_by_all_six"]["true_label_distribution"])
        + ")} \\\\",
        "\\multicolumn{8}{l}{\\emph{solved by all six:} "
        + f"{overlap['solved_by_all_six']['count']} comments ("
        + breakdown(overlap["solved_by_all_six"]["true_label_distribution"])
        + ")} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return _table(lines, path)


def table_unique_errors(overlap: Dict[str, Any], path: Path) -> Path:
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Errors made by exactly one system (out of the six).  A high count means "
        "the system's mistakes are largely its own; a low count means its errors are shared "
        "with the others.}",
        "\\label{tab:supp-unique-errors}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "System & unique errors & total errors \\\\",
        "\\midrule",
    ]
    total_errors = {}
    for name in MODEL_ORDER:
        m = json.loads((PER_MODEL_DIR / name / "metrics.json").read_text(encoding="utf-8"))
        cm = np.asarray(m["confusion_matrix"], dtype=float)
        total_errors[MODEL_LABELS[name]] = int(cm.sum() - np.trace(cm))
    for name in MODEL_ORDER:
        label = MODEL_LABELS[name]
        lines.append(f"{label} & {overlap['unique_errors'][label]:,} & {total_errors[label]:,} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_threshold(sweeps: Dict[str, Dict[str, Any]], path: Path) -> Path:
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Post-hoc prior-reweighting diagnostic.  The frozen posteriors are "
        "reweighted by a joint factor $u$ on the two minority classes and re-argmaxed; "
        "$u^{\\star}$ is the \\emph{oracle} factor that maximises test macro-F1.  "
        "``argmax M-F1'' is the macro-F1 of a plain argmax on the frozen posteriors; it "
        "differs from the system's headline macro-F1 only for M4, whose "
        "validation-searched thresholds ($0.01/0.5/0.64$) \\emph{lowered} test macro-F1 "
        "from the argmax $0.6602$ to the reported $0.6590$.  $u^{\\star}$ is chosen on the "
        "test labels and the gain is therefore an oracle upper bound over this one-parameter "
        "reweighting family, not a deployable decision rule.}",
        "\\label{tab:supp-threshold}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "System & argmax M-F1 & $u^{\\star}$ & best M-F1 & oracle gain \\\\",
        "\\midrule",
    ]
    for name in MODEL_ORDER:
        s = sweeps[name]
        lines.append(
            f"{MODEL_LABELS[name]} & {_fmt(s['argmax_macro_f1'])} & {s['best_factor']:.2f} "
            f"& {_fmt(s['best_macro_f1'])} & {s['gain']:+.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_pairs(pairs: List[Dict[str, Any]], path: Path) -> Path:
    p_f1_holm = _holm([p["macro_f1"]["p_value"] for p in pairs])
    p_acc_holm = _holm([p["mcnemar"]["p_value"] for p in pairs])
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Complete pairwise comparison of all $\\binom{6}{2}=15$ system pairs. "
        "$\\Delta$M-F1 is the paired-bootstrap difference ($A$ minus $B$) with a 95\\% CI, "
        "$p_{\\text{F1}}$ its bootstrap $p$-value; $p_{\\text{acc}}$ is the exact two-sided "
        "McNemar $p$-value on accuracy.  $^{\\dagger}$ marks $p<0.05$ after a "
        "Holm--Bonferroni step-down correction over these 15 tests.}",
        "\\label{tab:supp-pairs}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "$A-B$ & $\\Delta$M-F1 [95\\% CI] & $p_{\\text{F1}}$ & $p_{\\text{acc}}$ (McN.) & $p_{\\text{acc}}$ (Holm) \\\\",
        "\\midrule",
    ]
    for idx, pair in enumerate(pairs):
        d = pair["macro_f1"]
        m = pair["mcnemar"]
        star_acc = "$^{\\dagger}$" if p_acc_holm[idx] < 0.05 else ""
        star_f1 = "$^{\\dagger}$" if p_f1_holm[idx] < 0.05 else ""
        lines.append(
            f"{pair['label_a']} $-$ {pair['label_b']} & {d['delta']:+.4f} "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] & {_fmt_p(d['p_value'])}{star_f1} "
            f"& {_fmt_p(m['p_value'])}{star_acc} & {_fmt_p(p_acc_holm[idx])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_stratification(payload: Dict[str, Any], path: Path) -> Path:
    strata = sorted(payload["error_stratification"][MODEL_ORDER[0]].keys())
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Macro-F1 by surface phenomenon of the test comments, for all six "
        "systems.  Strata overlap (a comment can be both short and teencode); counts are "
        "the number of test comments carrying the tag.}",
        "\\label{tab:supp-stratification}",
        "\\begin{tabular}{lccccccr}",
        "\\toprule",
        "Stratum & M1 & M2 & M3 & M4 & M5 & M6 & n \\\\",
        "\\midrule",
    ]
    for tag in strata:
        n = payload["error_stratification"][MODEL_ORDER[0]][tag]["count"]
        cells = " & ".join(
            _fmt(payload["error_stratification"][name][tag]["macro_f1"]) for name in MODEL_ORDER
        )
        lines.append(f"\\texttt{{{tag.replace('_', chr(92) + '_')}}} & {cells} & {n:,} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_hyperparameters(
    training: Dict[str, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    path: Path,
) -> Path:
    rows = []
    for name in MODEL_ORDER:
        cfg = load_yaml(resolve_config_path(f"configs/{name}.yaml"))
        summary = training[name]
        stages = summary["training"]["stages"]["stages"]
        head = summary.get("head", "-")
        seed = cfg.get("seed", "-")
        losses = []
        for stage in stages:
            tag = stage["loss"]["type"]
            losses.append({"cross_entropy": "CE", "focal": "focal", "concept": "concept"}.get(tag, tag))
        loss_seq = " $\\to$ ".join(dict.fromkeys(losses))
        lr = stages[0]["training"]["learning_rate"]
        dlr = stages[0].get("dlr", {}) or {}
        head_lr = dlr.get("head_lr") if dlr.get("enabled") else lr
        warmup = stages[0]["training"].get("warmup_ratio", "-")
        batch = stages[0]["training"].get("batch_size", "-")
        accum = stages[0]["training"].get("gradient_accumulation", "-")
        patience = (cfg.get("early_stopping") or {}).get("patience", "-")
        # The decision rule that was actually applied on the test pass is
        # recorded in the evaluated run's metrics.json, which is authoritative
        # even if the config on disk was edited after the run.
        decision_rule = str(metrics[name].get("decision_rule", "argmax")).lower()
        thresholds = decision_rule == "thresholds"
        rows.append(
            {
                "label": MODEL_LABELS[name],
                "head": head,
                "stages": len(stages),
                "loss": loss_seq,
                "lr": f"${lr:.1e}$",
                "head_lr": f"${head_lr:.1e}$" if isinstance(head_lr, (int, float)) else "-",
                "warmup": warmup,
                "bs": f"{batch}$\\times${accum}",
                "pat": patience,
                "seed": seed,
                "thr": "yes" if thresholds else "no",
            }
        )

    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\caption{Resolved per-system configuration as recorded in each run's "
        "\\texttt{training\\_summary.json} and \\texttt{configs/*.yaml}.  ``lr'' is the "
        "stage-1 encoder rate; ``head lr'' the discriminative head rate when DLR is on.  "
        "``thr.''\\ marks the systems whose test pass applied per-class thresholds "
        "searched on validation (M3--M6, read from each run's \\texttt{metrics.json}); "
        "the search returned the trivial $0.5$ for all but M4.  "
        "AdamW, weight decay 0.01, gradient clipping 1.0 and max length 256 are shared.}",
        "\\label{tab:supp-hyperparameters}",
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        "System & head & stages & loss & lr & head lr & warmup & bs$\\times$acc & pat. & seed & thr. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['head']} & {row['stages']} & {row['loss']} & {row['lr']} "
            f"& {row['head_lr']} & {row['warmup']} & {row['bs']} & {row['pat']} "
            f"& {row['seed']} & {row['thr']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


def table_timing(training: Dict[str, Dict[str, Any]], path: Path) -> Path:
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Wall-clock cost and checkpoint selection per system.  ``best ep.''"
        " is the epoch whose validation macro-F1 was frozen for the test pass and "
        "``stopped'' whether early stopping fired before the planned epoch budget.}",
        "\\label{tab:supp-timing}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "System & epochs & planned & best ep. & stopped & wall clock \\\\",
        "\\midrule",
    ]
    for name in MODEL_ORDER:
        t = training[name]["training"]
        lines.append(
            f"{MODEL_LABELS[name]} & {t['epochs_run']} & {t['total_epochs_planned']} "
            f"& {t['best_epoch']} & {'yes' if t.get('stopped_early') else 'no'} "
            f"& {t.get('training_time_hms', '-')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return _table(lines, path)


# --------------------------------------------------------------------------- #
# copy every already-produced plot into the supplementary bundle
# --------------------------------------------------------------------------- #
#: Caption for each all-model comparison plot (key = file name).
COMPARISON_CAPTIONS: Dict[str, Tuple[str, str]] = {
    # (label, caption) -- label is used for \label{fig:supp-...}
    "accuracy_all_models.png": (
        "acc",
        "Test accuracy for all six systems with 95\\% bootstrap confidence "
        "intervals.  The highest accuracy belongs to \\model{M5}, which does "
        "not have the best macro-F1.",
    ),
    "macro_f1_all_models.png": (
        "macro-f1",
        "Test macro-F1 (mean of the three class F1 scores) for all six "
        "systems.  The six systems sit in a narrow band, which is why no "
        "pairwise macro-F1 difference is significant.",
    ),
    "per_class_f1_all_models.png": (
        "per-class-f1",
        "Per-class F1 for HATE, OFFENSIVE and CLEAN, grouped by system.  The "
        "minority classes (HATE, OFFENSIVE) show the largest spread across "
        "systems, while CLEAN is saturated.",
    ),
    "f1_summary_all_models.png": (
        "f1-summary",
        "Macro-F1, weighted-F1 and per-class F1 side by side for every system.",
    ),
    "validation_loss_all_models.png": (
        "val-loss",
        "Validation loss over the training epochs for all six systems.  The "
        "checkpoint is \emph{not} selected to minimise this curve: it is the "
        "epoch with the highest validation macro-F1 (Fig.~\ref{fig:supp-val-acc}), "
        "which is why the selected checkpoints can sit on a rising-loss epoch.",
    ),
    "validation_accuracy_all_models.png": (
        "val-acc",
        "Validation accuracy over the training epochs for all six systems.",
    ),
    "validation_loss_accuracy_all_models.png": (
        "val-loss-acc",
        "Validation loss and accuracy on the same axes for all six systems.",
    ),
    "pr_curve_all_models.png": (
        "pr-all",
        "One-vs-rest precision--recall curves (micro and macro averages) for "
        "all six systems.",
    ),
    "roc_curve_all_models.png": (
        "roc-all",
        "One-vs-rest ROC curves (micro and macro averages) for all six "
        "systems.",
    ),
    "pr_hate_all_models.png": (
        "pr-hate",
        "Precision--recall curve for the HATE class, all six systems.",
    ),
    "pr_offensive_all_models.png": (
        "pr-offensive",
        "Precision--recall curve for the OFFENSIVE class, all six systems.",
    ),
    "pr_clean_all_models.png": (
        "pr-clean",
        "Precision--recall curve for the CLEAN class, all six systems.",
    ),
    "roc_hate_all_models.png": (
        "roc-hate",
        "ROC curve for the HATE class, all six systems.",
    ),
    "roc_offensive_all_models.png": (
        "roc-offensive",
        "ROC curve for the OFFENSIVE class, all six systems.",
    ),
    "roc_clean_all_models.png": (
        "roc-clean",
        "ROC curve for the CLEAN class, all six systems.",
    ),
}

#: Caption fragment for each per-model plot (key = file name).
PER_MODEL_CAPTIONS: Dict[str, str] = {
    "confusion_matrix_normalized.png": (
        "Normalised confusion matrix (rows = true, columns = predicted)."
    ),
    "pr_curve.png": "Precision--recall curve.",
    "roc_curve.png": "ROC curve.",
    "val_loss_curve.png": "Validation loss over the training epochs.",
    "val_accuracy_curve.png": "Validation accuracy over the training epochs.",
}

#: A one-sentence description of each system, printed above its figures.
PER_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "sensitiveai-vi": (
        "Baseline: the pre-trained encoder with a standard [CLS] softmax head, "
        "single-stage training and plain cross-entropy.  It has the second-best "
        "macro average precision (0.6599)."
    ),
    "sensitiveai-vi-custom": (
        "Baseline plus a \\emph{multi-view pooling} head: the first token "
        "(\\texttt{[CLS]}), masked-mean and masked-max views are concatenated to "
        "a 2{,}304-d vector and passed through a LayerNorm $\\to$ "
        "Linear($2304\\to512$) $\\to$ GELU $\\to$ dropout bottleneck MLP "
        "($512\\to3$).  There is no convolutional layer and the optional "
        "attention-pooling view is disabled; training is single-stage with plain "
        "cross-entropy."
    ),
    "sensitiveai-vi-customdlr2stage": (
        "Multi-view pooling head with discriminative learning rates and a second "
        "focal-loss stage; the best macro-F1 of the six ($0.6638$) and the highest "
        "MCC.  Its threshold search returned the trivial $0.5$."
    ),
    "sensitiveai-vi-custom-curriculum": (
        "Multi-view head trained as a three-stage curriculum, initialised from "
        "\\model{M3}'s checkpoint.  Stage~1 (concept loss, OFFENSIVE) was allocated "
        "10 epochs but early stopping truncated it after 7; stage~2 (concept loss, "
        "HATE) ran 6\\,(absolute epochs $8$--$13$) and stage~3 (cross-entropy, "
        "class weights $3{:}2{:}1$) ran 6\\,(absolute epochs $14$--$19$), for "
        "$19$ of $30$ planned epochs.  The run's best \\emph{validation} macro-F1 "
        "($0.6942$) occurred at absolute epoch~1 and was never beaten, so the "
        "later stages were not selected; its searched thresholds "
        "($0.01/0.5/0.64$) raised validation macro-F1 but did not transfer to test."
    ),
    "sensitiveai-vi-customdlr2stage-ext": (
        "Continuation of \\model{M3}'s stage-1 recipe (DLR plus class-weighted "
        "cross-entropy) for a further 15 epochs, reaching the full budget because "
        "the early-stopping patience equals the budget and could not trigger "
        "earlier.  It has the \\emph{highest} test accuracy ($0.878$) but the "
        "\\emph{lowest} macro-F1 ($0.654$)---the accuracy/macro-F1 trap in one "
        "system."
    ),
    "sensitiveai-vi-stage2": (
        "A focal-only second stage on the baseline head (class-weighted focal "
        "loss, no multi-view head).  It has the best HATE recall ($0.647$), the "
        "best HATE F1 ($0.609$) and the best macro ROC-AUC ($0.893$), but the "
        "lowest CLEAN recall ($0.922$) and the lowest accuracy ($0.863$)."
    ),
}


def _copy_figures() -> None:
    """Copy the already-rendered comparison and per-model plots into the bundle.

    The plots are read-only sources under ``outputs/evaluation/comparison`` and
    ``outputs/evaluation/per_model``; copying them makes the supplementary
    bundle self-contained so the PDF builds without the full run tree.
    """
    SUPP_COMPARISON_FIG.mkdir(parents=True, exist_ok=True)
    for name in COMPARISON_FIGS:
        src = None
        for candidate in (PAPERS_FIGURES / name, COMPARISON_DIR / name):
            if candidate.is_file():
                src = candidate
                break
        if src is not None:
            (SUPP_COMPARISON_FIG / name).write_bytes(src.read_bytes())

    SUPP_PER_MODEL_FIG.mkdir(parents=True, exist_ok=True)
    for name in MODEL_ORDER:
        label = MODEL_LABELS[name]
        dst_dir = SUPP_PER_MODEL_FIG / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fig in PER_MODEL_FIGS:
            src = PER_MODEL_DIR / name / fig
            if src.is_file():
                (dst_dir / fig).write_bytes(src.read_bytes())


def _emit_comparison_figures_tex() -> Path:
    """Write ``sections/comparison_figures.tex`` (all-model plots, 2-up)."""
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "",
    ]
    names = [n for n in COMPARISON_FIGS if n in COMPARISON_CAPTIONS]
    for name in names:
        key = COMPARISON_CAPTIONS[name][0]
        lines.append("\\begin{figure}[htbp]")
        lines.append("\\centering")
        lines.append(
            f"\\begin{{minipage}}{{0.74\\textwidth}}\\centering"
            f"\\includegraphics[width=\\linewidth]{{figures/comparison/{name}}}"
            f"\\caption{{\\label{{fig:supp-{key}}}"
            f"{COMPARISON_CAPTIONS[name][1]}}}\\end{{minipage}}"
        )
        lines.append("\\end{figure}")
        lines.append("")
    lines.append("\\FloatBarrier")
    lines.append("")
    out = SUPP_SECTIONS / "comparison_figures.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _emit_per_system_figures_tex() -> Path:
    """Write ``sections/per_system_figures.tex`` (per-system plots, 2-up)."""
    lines = [
        "% AUTO-GENERATED by scripts/11_supplementary_analyses.py - do not edit by hand.",
        "",
    ]
    for name in MODEL_ORDER:
        label = MODEL_LABELS[name]
        desc = PER_MODEL_DESCRIPTIONS.get(name, "")
        lines.append(f"\\subsection{{\\model{{{label}}}}}")
        lines.append(f"\\label{{sec:supp-per-{label.lower()}}}")
        if desc:
            lines.append(desc)
        lines.append("")
        for fig in PER_MODEL_FIGS:
            caption = PER_MODEL_CAPTIONS.get(fig, "")
            lines.append("\\begin{figure}[htbp]")
            lines.append("\\centering")
            lines.append(
                f"\\begin{{minipage}}{{0.74\\textwidth}}\\centering"
                f"\\includegraphics[width=\\linewidth]"
                f"{{figures/per_model/{label}/{fig}}}"
                f"\\caption{{{caption}}}\\end{{minipage}}"
            )
            lines.append("\\end{figure}")
            lines.append("")
        lines.append("")
    out = SUPP_SECTIONS / "per_system_figures.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
def main() -> int:
    import matplotlib

    matplotlib.use("Agg")

    frames = load_predictions(MODEL_ORDER, PER_MODEL_DIR)
    reference = frames[MODEL_ORDER[0]]
    y_true = reference["true_label_id"].to_numpy()
    metrics = _load_metrics()
    training = _load_training()
    significance = json.loads(SIG_PATH.read_text(encoding="utf-8"))

    # per model: correctness, confusion, mcc, calibration extras
    calibration: Dict[str, Dict[str, float]] = {}
    for name in MODEL_ORDER:
        proba = frames[name][PROBA_COLS].to_numpy(dtype=float)
        y_pred = frames[name]["pred_label_id"].to_numpy()
        cm = confusion_from_labels(y_true, y_pred)
        metrics[name]["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        metrics[name]["confusion_matrix"] = cm.astype(int).tolist()
        calibration[name] = {
            "ece": float(significance["confidence"][name]["calibration"]["ece"]),
            "bins": significance["confidence"][name]["calibration"]["bins"],
            "brier": brier_score(proba, y_true),
            "nll": negative_log_likelihood(proba, y_true),
            "mean_confidence": float(proba.max(axis=1).mean()),
        }

    # complete pairwise matrix (all 15 pairs)
    pairs: List[Dict[str, Any]] = []
    for a, b in combinations(MODEL_ORDER, 2):
        pred_a = frames[a]["pred_label_id"].to_numpy()
        pred_b = frames[b]["pred_label_id"].to_numpy()
        pairs.append(
            {
                "a": a,
                "b": b,
                "label_a": MODEL_LABELS[a],
                "label_b": MODEL_LABELS[b],
                "macro_f1": paired_bootstrap_delta(y_true, pred_a, pred_b, "macro_f1", n_boot=N_BOOT, seed=SEED),
                "mcnemar": mcnemar(y_true, pred_a, pred_b),
            }
        )

    overlap = error_overlap(frames, y_true)

    factors = np.round(np.arange(0.6, 3.001, 0.1), 3)
    sweeps = {name: prior_reweighting_sweep(frames[name], y_true, factors) for name in MODEL_ORDER}

    # ---- write LaTeX tables -------------------------------------------------- #
    table_per_class(metrics, SUPP_TABLES / "per_class.tex")
    table_ranking(metrics, SUPP_TABLES / "ranking.tex")
    table_calibration(calibration, SUPP_TABLES / "calibration.tex")
    table_confusion_counts(metrics, SUPP_TABLES / "confusion_counts.tex")
    table_error_overlap(overlap, SUPP_TABLES / "error_overlap.tex")
    table_unique_errors(overlap, SUPP_TABLES / "unique_errors.tex")
    table_threshold(sweeps, SUPP_TABLES / "threshold.tex")
    table_pairs(pairs, SUPP_TABLES / "pairs_all.tex")
    table_stratification(significance, SUPP_TABLES / "stratification.tex")
    table_hyperparameters(training, metrics, SUPP_TABLES / "hyperparameters.tex")
    table_timing(training, SUPP_TABLES / "timing.tex")

    # ---- write figures ------------------------------------------------------- #
    plot_confusion_grid(metrics, SUPP_FIGURES / "confusion_grid.png")
    plot_reliability(calibration, SUPP_FIGURES / "reliability.png")
    plot_error_overlap(overlap, SUPP_FIGURES / "error_overlap.png")
    plot_threshold_sweep(sweeps, SUPP_FIGURES / "threshold_sweep.png")

    # ---- copy every already-rendered plot into the bundle --------------------- #
    _copy_figures()
    _emit_comparison_figures_tex()
    _emit_per_system_figures_tex()

    # ---- JSON + markdown summary -------------------------------------------- #
    payload = {
        "models": list(MODEL_ORDER),
        "num_test": int(y_true.shape[0]),
        "seed": SEED,
        "n_boot": N_BOOT,
        "calibration": {name: {k: v for k, v in calibration[name].items() if k != "bins"} for name in calibration},
        "mcc": {MODEL_LABELS[n]: metrics[n]["mcc"] for n in MODEL_ORDER},
        "pairs": pairs,
        "error_overlap": overlap,
        "prior_reweighting": {MODEL_LABELS[n]: sweeps[n] for n in MODEL_ORDER},
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "supplementary_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Supplementary analyses (frozen predictions)",
        "",
        f"- test items: {y_true.shape[0]:,}",
        f"- bootstrap resamples: {N_BOOT:,} (seed {SEED})",
        "",
        "## Calibration",
        "",
        "| System | ECE | Brier | NLL | mean conf. | MCC |",
        "|---|---|---|---|---|---|",
    ]
    for name in MODEL_ORDER:
        c = calibration[name]
        md.append(
            f"| {MODEL_LABELS[name]} | {c['ece']:.3f} | {c['brier']:.4f} | {c['nll']:.4f} "
            f"| {c['mean_confidence']:.3f} | {metrics[name]['mcc']:.4f} |"
        )
    md += [
        "",
        "## Error complementarity",
        "",
        "| # systems wrong | " + " | ".join(str(k) for k in range(7)) + " |",
        "|---|" + "---|" * 7,
        "| test comments | "
        + " | ".join(f"{overlap['k_distribution'][k]:,}" for k in range(7))
        + " |",
        "",
        f"- missed by all six: {overlap['missed_by_all_six']['count']} "
        f"({overlap['missed_by_all_six']['true_label_distribution']})",
        f"- solved by all six: {overlap['solved_by_all_six']['count']} "
        f"({overlap['solved_by_all_six']['true_label_distribution']})",
        "",
        "## Prior-reweighting diagnostic (oracle on test)",
        "",
        "| System | argmax M-F1 | u* | best M-F1 | gain |",
        "|---|---|---|---|---|",
    ]
    for name in MODEL_ORDER:
        s = sweeps[name]
        md.append(
            f"| {MODEL_LABELS[name]} | {s['argmax_macro_f1']:.4f} | {s['best_factor']:.2f} "
            f"| {s['best_macro_f1']:.4f} | {s['gain']:+.4f} |"
        )
    (ANALYSIS_DIR / "supplementary_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- mirror into the generated copy under outputs/paper ------------------ #
    mirror = PROJECT_ROOT / "outputs" / "paper" / "supplementary"
    for sub in ("tables", "figures", "sections"):
        src_dir = SUPP_DIR / sub
        dst_dir = mirror / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in src_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(src_dir)
                target = dst_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())

    print(f"supplementary tables -> {SUPP_TABLES}")
    print(f"supplementary figures -> {SUPP_FIGURES}")
    print(f"supplementary sections -> {SUPP_SECTIONS}")
    print(f"report -> {ANALYSIS_DIR / 'supplementary_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
