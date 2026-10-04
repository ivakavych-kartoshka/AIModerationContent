"""Four-model comparison + paper-ready output.

Run **after** all four models have been trained and evaluated:

    python -m src.comparison.compare_models

Produces (plan sections 23-26 and 29-32)
-----------------------------------------
``evaluation/comparison/``
    ``model_comparison.csv``          - the main results table (§30)
    ``model_comparison.md``           - same content, readable
    ``pr_curve_all_models.png``       - macro PR, one line per model
    ``roc_curve_all_models.png``      - macro ROC, one line per model
    ``pr_{hate,offensive,clean}_all_models.png``
    ``roc_{hate,offensive,clean}_all_models.png``
    ``validation_accuracy_all_models.png``
    ``validation_loss_all_models.png``
    ``macro_f1_all_models.png``, ``per_class_f1_all_models.png`` (extras)
    ``comparison_summary.json``

``paper/figures/``  copies of every figure listed in plan section 32
``paper/tables/``   ``model_comparison.tex``, ``classification_results.tex``,
                    ``ablation_results.tex`` - generated, never hand-written
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..common.config import dump_json
from ..common.constants import MODEL_NAMES, PAPER_DIR
from ..common.logging_utils import get_logger
from ..common.paths import comparison_dir, ensure_dir, find_best_report_dir
from ..evaluation.plots import (
    MODEL_MARKERS,
    MODEL_ORDER,
    model_color,
    model_label,
    setup_style,
    _save,
)

LOGGER = get_logger("sensitiveai.compare")

CLASSES = ("HATE", "OFFENSIVE", "CLEAN")


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #
def collect_runs(model_names: Sequence[str], reports_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load ``metrics.json``, curve data and ``training_log.csv`` of each model."""
    runs: Dict[str, Dict[str, Any]] = {}
    for name in model_names:
        try:
            report_path = find_best_report_dir(name, reports_root)
        except FileNotFoundError as exc:
            LOGGER.warning("[compare] skipping %s: %s", name, exc)
            continue

        metrics_path = report_path / "metrics.json"
        if not metrics_path.is_file():
            LOGGER.warning("[compare] skipping %s: %s is missing (run the evaluation first)", name, metrics_path)
            continue

        run: Dict[str, Any] = {
            "report_dir": report_path,
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        }
        for key, filename in (("pr", "pr_curve_data.json"), ("roc", "roc_curve_data.json")):
            path = report_path / filename
            run[key] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        log_path = report_path / "training_log.csv"
        run["log"] = pd.read_csv(log_path) if log_path.is_file() else pd.DataFrame()

        bench_path = report_path / "benchmark.json"
        run["benchmark"] = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.is_file() else {}
        summary_path = report_path / "training_summary.json"
        run["training_summary"] = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        )
        thresholds_path = report_path / "thresholds.json"
        run["thresholds"] = (
            json.loads(thresholds_path.read_text(encoding="utf-8")) if thresholds_path.is_file() else None
        )
        runs[name] = run
        LOGGER.info("[compare] loaded %s -> %s", name, report_path)
    if not runs:
        raise RuntimeError("No evaluated model found. Train and evaluate at least one model first.")
    return runs


def order_models(runs: Dict[str, Any]) -> List[str]:
    known = [m for m in MODEL_ORDER if m in runs]
    extra = sorted(m for m in runs if m not in MODEL_ORDER)
    return known + extra


# --------------------------------------------------------------------------- #
# comparison table
# --------------------------------------------------------------------------- #
def build_comparison_table(runs: Dict[str, Any]) -> pd.DataFrame:
    """Columns follow plan section 30 exactly (plus a few useful extras)."""
    rows = []
    for name in order_models(runs):
        run = runs[name]
        m = run["metrics"]
        bench = run.get("benchmark") or {}
        params = (run.get("training_summary") or {}).get("parameters") or {}
        training = (run.get("training_summary") or {}).get("training") or {}

        rows.append(
            {
                "Model": name,
                "Accuracy": m.get("accuracy"),
                "Macro Precision": m.get("macro_precision"),
                "Macro Recall": m.get("macro_recall"),
                "Macro F1": m.get("macro_f1"),
                "Weighted F1": m.get("weighted_f1"),
                "HATE Precision": m.get("HATE_precision"),
                "HATE Recall": m.get("HATE_recall"),
                "HATE F1": m.get("HATE_f1"),
                "OFFENSIVE Precision": m.get("OFFENSIVE_precision"),
                "OFFENSIVE Recall": m.get("OFFENSIVE_recall"),
                "OFFENSIVE F1": m.get("OFFENSIVE_f1"),
                "CLEAN Precision": m.get("CLEAN_precision"),
                "CLEAN Recall": m.get("CLEAN_recall"),
                "CLEAN F1": m.get("CLEAN_f1"),
                "HATE ROC-AUC": m.get("HATE_roc_auc"),
                "OFFENSIVE ROC-AUC": m.get("OFFENSIVE_roc_auc"),
                "CLEAN ROC-AUC": m.get("CLEAN_roc_auc"),
                "Macro ROC-AUC": m.get("macro_roc_auc"),
                "HATE AP": m.get("HATE_average_precision"),
                "OFFENSIVE AP": m.get("OFFENSIVE_average_precision"),
                "CLEAN AP": m.get("CLEAN_average_precision"),
                "Macro AP": m.get("macro_average_precision"),
                "Parameters": params.get("total_parameters"),
                "Latency": bench.get("latency_ms_per_sample"),
                "FPS": bench.get("fps"),
                # --- extras (documented as such) ---
                "Decision Rule": m.get("decision_rule"),
                "Best Epoch": training.get("best_epoch"),
                "Training Time (s)": training.get("training_seconds"),
                "Micro ROC-AUC": m.get("micro_roc_auc"),
                "Test Samples": m.get("num_test_samples"),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, path: Path) -> Path:
    """Compact markdown rendering: identity + headline metrics."""
    columns = [
        "Model", "Accuracy", "Macro F1", "Weighted F1",
        "HATE F1", "OFFENSIVE F1", "CLEAN F1",
        "Macro ROC-AUC", "Macro AP", "Parameters", "Latency", "FPS",
    ]
    subset = table[[c for c in columns if c in table.columns]].copy()
    for column in subset.columns:
        if subset[column].dtype.kind == "f":
            subset[column] = subset[column].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
        elif subset[column].dtype.kind == "i":
            subset[column] = subset[column].map(lambda v: f"{v:,}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Model comparison (ViHSD test set)\n\n")
        fh.write("Latency = milliseconds per sample; FPS = samples per second. ")
        fh.write("Both measured on the same GPU with the same tokenizer, max_length and batch size.\n\n")
        fh.write(subset.to_markdown(index=False))
        fh.write("\n\n## Full table\n\n")
        fh.write(table.to_markdown(index=False))
        fh.write("\n")
    LOGGER.info("[compare] %s", path)
    return path


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_macro_pr_comparison(runs: Dict[str, Any], path: Path) -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for name in order_models(runs):
        pr = runs[name].get("pr")
        if not pr:
            continue
        ax.plot(
            pr["macro_recall"], pr["macro_precision"],
            color=model_color(name), linewidth=2.0, linestyle="-",
            label=f"{model_label(name)} (mAP={pr.get('macro_average_precision', float('nan')):.4f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Macro Precision-Recall comparison (one-vs-rest)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


def plot_macro_roc_comparison(runs: Dict[str, Any], path: Path) -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0, label="chance")
    for name in order_models(runs):
        roc = runs[name].get("roc")
        if not roc:
            continue
        ax.plot(
            roc["macro_fpr"], roc["macro_tpr"],
            color=model_color(name), linewidth=2.0,
            label=f"{model_label(name)} (AUC={roc.get('macro_roc_auc', float('nan')):.4f})",
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Macro ROC comparison (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


def plot_per_class_pr_comparison(runs: Dict[str, Any], class_name: str, path: Path) -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for name in order_models(runs):
        pr = runs[name].get("pr")
        if not pr or class_name not in pr.get("per_class", {}):
            continue
        entry = pr["per_class"][class_name]
        ax.plot(
            entry["recall"], entry["precision"],
            color=model_color(name), linewidth=1.9,
            label=f"{model_label(name)} (AP={entry.get('average_precision', float('nan')):.4f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-Recall comparison - {class_name} vs rest")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


def plot_per_class_roc_comparison(runs: Dict[str, Any], class_name: str, path: Path) -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0, label="chance")
    for name in order_models(runs):
        roc = runs[name].get("roc")
        if not roc or class_name not in roc.get("per_class", {}):
            continue
        entry = roc["per_class"][class_name]
        ax.plot(
            entry["fpr"], entry["tpr"],
            color=model_color(name), linewidth=1.9,
            label=f"{model_label(name)} (AUC={entry.get('roc_auc', float('nan')):.4f})",
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"ROC comparison - {class_name} vs rest")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


def plot_validation_metric_comparison(
    runs: Dict[str, Any],
    path: Path,
    column: str = "val_accuracy",
    ylabel: str = "Validation Accuracy",
    as_percent: bool = True,
) -> Path:
    """Val-accuracy / val-loss comparison, read from each ``training_log.csv``."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    scale = 100.0 if as_percent else 1.0
    plotted = 0
    for name in order_models(runs):
        log = runs[name].get("log")
        if log is None or len(log) == 0 or column not in log.columns:
            LOGGER.warning("[compare] no '%s' column for %s", column, name)
            continue
        log = log.sort_values("epoch")
        ax.plot(
            log["epoch"], log[column].astype(float) * scale,
            color=model_color(name), linewidth=2.0,
            marker=MODEL_MARKERS.get(name, "o"), markersize=4,
            label=model_label(name),
        )
        plotted += 1
    if plotted == 0:
        raise RuntimeError(f"No training log contains the column {column!r}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel + (" (%)" if as_percent else ""))
    ax.set_title(f"{ylabel} across the four compared models")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_bars(runs: Dict[str, Any], path: Path, metric: str = "macro_f1", title: Optional[str] = None) -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    names = order_models(runs)
    values = [float(runs[n]["metrics"].get(metric, float("nan"))) for n in names]
    labels = [model_label(n).replace(" (", "\n(") for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, values, color=[model_color(n) for n in names], width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    lo = min(values) if values else 0.0
    ax.set_ylim(max(0.0, lo - 0.05), min(1.0, max(values) + 0.05) if values else 1.0)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} on the ViHSD test set")
    fig.tight_layout()
    return _save(fig, path)


def plot_per_class_f1_comparison(runs: Dict[str, Any], path: Path) -> Path:
    setup_style()
    import matplotlib.pyplot as plt

    names = order_models(runs)
    width = 0.8 / max(len(names), 1)
    x = np.arange(len(CLASSES))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, name in enumerate(names):
        values = [float(runs[name]["metrics"].get(f"{c}_f1", float("nan"))) for c in CLASSES]
        ax.bar(x + i * width - 0.4 + width / 2, values, width=width,
               color=model_color(name), label=model_label(name))
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1 comparison on the ViHSD test set")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, float) and np.isnan(value):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _bold_best(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Mark the best value of every metric column in bold."""
    out = table.copy()
    for column in columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        # macro-averaged metrics: higher is better; latency: lower is better
        best = values.min() if column == "Latency" else values.max()
        if pd.isna(best):
            continue
        out[column] = [
            f"\\textbf{{{_fmt(v)}}}" if pd.notna(v) and abs(float(v) - float(best)) < 1e-12 else _fmt(v)
            for v in out[column]
        ]
    return out


def _write_tex(path: Path, lines: Sequence[str]) -> Path:
    """Write a generated ``.tex`` file and return its *path* (``write_text`` returns a count)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_latex_tables(table: pd.DataFrame, runs: Dict[str, Any], paper_dir: Path) -> Dict[str, Path]:
    tables_dir = ensure_dir(paper_dir / "tables")
    outputs: Dict[str, Path] = {}

    # ---- model_comparison.tex --------------------------------------- #
    headline = table[[
        "Model", "Accuracy", "Macro F1", "Weighted F1", "HATE F1", "OFFENSIVE F1", "CLEAN F1",
        "Macro ROC-AUC", "Macro AP",
    ]].copy()
    rendered = _bold_best(headline, headline.columns[1:])
    lines = [
        "% AUTO-GENERATED by src/comparison/compare_models.py - do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Comparison of the four SensitiveAI systems on the ViHSD test set. "
        "All systems share the same fixed split, the same label mapping and the same test set.}",
        "\\label{tab:model-comparison}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * (rendered.shape[1] - 1) + "}",
        "\\toprule",
        " & ".join(str(c).replace("_", " ") for c in rendered.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in rendered.iterrows():
        name = str(row["Model"]).replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(str(row[c]) for c in rendered.columns[1:]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    outputs["model_comparison"] = _write_tex(tables_dir / "model_comparison.tex", lines)

    # ---- classification_results.tex ---------------------------------- #
    lines = [
        "% AUTO-GENERATED by src/comparison/compare_models.py - do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-class precision / recall / F1 on the ViHSD test set.}",
        "\\label{tab:classification-results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "ccc" * 3 + "}",
        "\\toprule",
        "& \\multicolumn{3}{c}{HATE} & \\multicolumn{3}{c}{OFFENSIVE} & \\multicolumn{3}{c}{CLEAN} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "Model & P & R & F1 & P & R & F1 & P & R & F1 \\\\",
        "\\midrule",
    ]
    metric_cols = (
        [f"{c} {m.capitalize()}" for c in CLASSES for m in ("precision", "recall", "f1")]
    )
    rendered_cls = _bold_best(table, metric_cols)
    for _, row in rendered_cls.iterrows():
        name = str(row["Model"]).replace("_", "\\_")
        values = " & ".join(str(row[c]) for c in metric_cols)
        lines.append(f"{name} & {values} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    outputs["classification_results"] = _write_tex(tables_dir / "classification_results.tex", lines)

    # ---- ablation_results.tex ---------------------------------------- #
    lines = [
        "% AUTO-GENERATED by src/comparison/compare_models.py - do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation view: the effect of each design decision "
        "(custom head, discriminative learning rates + two-stage training, curriculum training).}",
        "\\label{tab:ablation-results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Variant & Head & DLR & Stages & Loss & Macro F1 \\\\",
        "\\midrule",
    ]
    features = {
        "sensitiveai-vi": ("standard", "no", "1", "cross-entropy"),
        "sensitiveai-vi-custom": ("custom", "no", "1", "cross-entropy"),
        "sensitiveai-vi-customdlr2stage": ("custom", "yes", "2", "CE stage 1 / focal stage 2"),
        "sensitiveai-vi-custom-curriculum": ("custom", "yes", "3", "concept 1 / concept 2 / CE"),
    }
    for name in order_models(runs):
        head, dlr, stages, loss = features.get(name, ("custom", "yes", "?", "?"))
        f1 = _fmt(runs[name]["metrics"].get("macro_f1"))
        lines.append(f"{name.replace('_', chr(92) + '_')} & {head} & {dlr} & {stages} & {loss} & {f1} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    outputs["ablation_results"] = _write_tex(tables_dir / "ablation_results.tex", lines)

    for key, path in outputs.items():
        LOGGER.info("[compare] %s -> %s", key, Path(path).name)
    return {k: Path(v) for k, v in outputs.items()}


# --------------------------------------------------------------------------- #
def copy_paper_figures(comparison_path: Path, paper_dir: Path, filenames: Sequence[str]) -> Dict[str, Path]:
    figures_dir = ensure_dir(paper_dir / "figures")
    copied: Dict[str, Path] = {}
    for name in filenames:
        src = comparison_path / name
        if src.is_file():
            shutil.copyfile(src, figures_dir / name)
            copied[name] = figures_dir / name
            LOGGER.info("[compare] paper figure -> %s", figures_dir / name)
        else:
            LOGGER.warning("[compare] figure not found for the paper: %s", src)
    return copied


# --------------------------------------------------------------------------- #
def run(
    model_names: Sequence[str] = MODEL_NAMES,
    reports_root: Optional[Path] = None,
    comparison_root: Optional[Path] = None,
    paper_root: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir = ensure_dir(comparison_root or comparison_dir())
    paper_dir = Path(paper_root) if paper_root else PAPER_DIR

    runs = collect_runs(model_names, reports_root)
    ordered = order_models(runs)
    LOGGER.info("[compare] comparing %d model(s): %s", len(ordered), ", ".join(ordered))

    table = build_comparison_table(runs)
    table_path = out_dir / "model_comparison.csv"
    table.to_csv(table_path, index=False, encoding="utf-8")
    LOGGER.info("[compare] %s", table_path)
    write_markdown_table(table, out_dir / "model_comparison.md")

    figures: Dict[str, Path] = {}
    figures["pr_curve_all_models.png"] = plot_macro_pr_comparison(runs, out_dir / "pr_curve_all_models.png")
    figures["roc_curve_all_models.png"] = plot_macro_roc_comparison(runs, out_dir / "roc_curve_all_models.png")
    for cls in CLASSES:
        key = cls.lower()
        figures[f"pr_{key}_all_models.png"] = plot_per_class_pr_comparison(
            runs, cls, out_dir / f"pr_{key}_all_models.png"
        )
        figures[f"roc_{key}_all_models.png"] = plot_per_class_roc_comparison(
            runs, cls, out_dir / f"roc_{key}_all_models.png"
        )
    figures["validation_accuracy_all_models.png"] = plot_validation_metric_comparison(
        runs, out_dir / "validation_accuracy_all_models.png", "val_accuracy", "Validation Accuracy"
    )
    figures["validation_loss_all_models.png"] = plot_validation_metric_comparison(
        runs, out_dir / "validation_loss_all_models.png", "val_loss", "Validation Loss", as_percent=False
    )
    figures["macro_f1_all_models.png"] = plot_metric_bars(runs, out_dir / "macro_f1_all_models.png", "macro_f1")
    figures["accuracy_all_models.png"] = plot_metric_bars(runs, out_dir / "accuracy_all_models.png", "accuracy")
    figures["per_class_f1_all_models.png"] = plot_per_class_f1_comparison(
        runs, out_dir / "per_class_f1_all_models.png"
    )

    latex = write_latex_tables(table, runs, paper_dir)
    paper_figures = copy_paper_figures(
        out_dir,
        paper_dir,
        [
            "validation_accuracy_all_models.png",
            "validation_loss_all_models.png",
            "pr_curve_all_models.png",
            "roc_curve_all_models.png",
            "pr_hate_all_models.png",
            "pr_offensive_all_models.png",
            "pr_clean_all_models.png",
            "roc_hate_all_models.png",
            "roc_offensive_all_models.png",
            "roc_clean_all_models.png",
            "macro_f1_all_models.png",
            "per_class_f1_all_models.png",
        ],
    )

    best = table.loc[table["Macro F1"].idxmax()] if len(table) else None
    summary = {
        "models_compared": ordered,
        "num_models": len(ordered),
        "table_file": str(table_path),
        "markdown_file": str(out_dir / "model_comparison.md"),
        "figures": {k: str(v) for k, v in figures.items()},
        "latex_tables": {k: str(v) for k, v in latex.items()},
        "paper_figures": {k: str(v) for k, v in paper_figures.items()},
        "best_macro_f1_model": str(best["Model"]) if best is not None else None,
        "best_macro_f1": float(best["Macro F1"]) if best is not None else None,
        "metrics": {
            name: {
                "accuracy": runs[name]["metrics"].get("accuracy"),
                "macro_f1": runs[name]["metrics"].get("macro_f1"),
                "weighted_f1": runs[name]["metrics"].get("weighted_f1"),
                "macro_roc_auc": runs[name]["metrics"].get("macro_roc_auc"),
                "macro_average_precision": runs[name]["metrics"].get("macro_average_precision"),
                "decision_rule": runs[name]["metrics"].get("decision_rule"),
                "report_dir": str(runs[name]["report_dir"]),
            }
            for name in ordered
        },
    }
    dump_json(summary, out_dir / "comparison_summary.json")

    LOGGER.info("=" * 78)
    LOGGER.info("Comparison written to %s", out_dir)
    LOGGER.info(table[["Model", "Accuracy", "Macro F1", "Weighted F1", "Macro ROC-AUC"]].to_string(index=False))
    LOGGER.info("=" * 78)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.comparison.compare_models",
        description="Compare the four trained SensitiveAI models and generate paper-ready output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES), help="Models to include.")
    parser.add_argument("--reports-dir", default=None, help="Override the reports root.")
    parser.add_argument("--output-dir", default=None, help="Override the comparison output directory.")
    parser.add_argument("--paper-dir", default=None, help="Override the paper directory.")
    args = parser.parse_args(argv)
    run(
        model_names=args.models,
        reports_root=Path(args.reports_dir) if args.reports_dir else None,
        comparison_root=Path(args.output_dir) if args.output_dir else None,
        paper_root=Path(args.paper_dir) if args.paper_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
