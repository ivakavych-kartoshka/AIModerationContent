"""Step 29 - hình so sánh các model trong hybrid fusion (plan 40-44).

    python scripts/29_plot_fusion_comparison.py

Reads the frozen artefacts that ``scripts/23_evaluate_fusion.py`` and
``scripts/25_fusion_significance.py`` already wrote (one ``metrics.json`` per
AI model folder plus ``fusion_significance.json``) and draws four figures
beside them in ``outputs/evaluation/fusion/``:

    fusion_ai_vs_fusion.png     macro-F1: AI-only vs fusion primary + panel Δ
    fusion_delta_ci.png         forest plot: Δ (fusion - AI) with 95% CI
    fusion_variants_delta.png   mọi biến thể fusion so với AI-only
    fusion_per_class_f1.png     F1 từng lớp: AI-only vs fusion primary

Everything is read from stored JSON - no split is re-scored, so the single
TEST pass of plan section 44 stays intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.29_plot_fusion_comparison")

FUSION_DIR = Path("outputs/evaluation/fusion")
COLOR_AI = "#1f77b4"
COLOR_FUSION = "#ff7f0e"
COLOR_RULE = "#9e9e9e"
COLOR_POS = "#2ca02c"
COLOR_NEG = "#d62728"

VARIANTS = (
    ("fusion_argmax", "fusion (argmax)"),
    ("fusion_thresholds", "fusion (thresholds)"),
    ("fusion_per_channel_thresholds", "fusion (per-channel)"),
    ("fusion_meta_classifier", "fusion (meta)"),
)
CLASS_ROWS = ("HATE", "OFFENSIVE", "CLEAN")


def short_name(model: str) -> str:
    if model == "sensitiveai-vi":
        return "vi (base)"
    return model.replace("sensitiveai-vi-", "")


def load_models(root: Path) -> List[Dict[str, Any]]:
    significance = json.loads((root / "fusion_significance.json").read_text(encoding="utf-8"))
    models: List[Dict[str, Any]] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = folder / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        primary = metrics["primary_variant"]
        sig = significance.get("results", {}).get(folder.name, {}).get("paired_bootstrap_macro_f1", {})
        # significance stores delta = A - B with A = AI, B = fusion; flip it so
        # every figure uses the same convention: Δ = fusion - AI.
        delta = -float(sig.get("delta(A_minus_B)", 0.0))
        ci_low = -float(sig.get("delta_ci_high", 0.0))
        ci_high = -float(sig.get("delta_ci_low", 0.0))
        models.append(
            {
                "name": folder.name,
                "short": short_name(folder.name),
                "metrics": metrics,
                "primary": primary,
                "ai": metrics["ai_only_reference"],
                "fusion": metrics["test"][primary],
                "rule_macro_f1": metrics["test"]["rule_only_frozen_thresholds"]["macro_f1"],
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p": float(sig.get("p", float("nan"))),
                "significant": bool(ci_low > 0.0 or ci_high < 0.0),
            }
        )
    if not models:
        raise FileNotFoundError(f"no <model>/metrics.json under {root}")
    return models


def _annotate_bars(ax: Any, bars: Any, fmt: str = "{:.4f}", fontsize: float = 7.5,
                   rotation: float = 0.0) -> None:
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                    fontsize=fontsize, rotation=rotation)


def plot_ai_vs_fusion(models: List[Dict[str, Any]], path: Path) -> None:
    labels = [m["short"] for m in models]
    xs = np.arange(len(models))
    width = 0.38

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    ai_values = [m["ai"]["macro_f1"] for m in models]
    fusion_values = [m["fusion"]["macro_f1"] for m in models]
    bars_ai = ax_left.bar(xs - width / 2, ai_values, width, label="AI-only (frozen thr.)", color=COLOR_AI)
    bars_fu = ax_left.bar(xs + width / 2, fusion_values, width, label="fusion primary (val-chosen)", color=COLOR_FUSION)
    _annotate_bars(ax_left, bars_ai, rotation=90, fontsize=7)
    _annotate_bars(ax_left, bars_fu, rotation=90, fontsize=7)
    ax_left.axhline(models[0]["rule_macro_f1"], color=COLOR_RULE, linestyle="--", linewidth=1.2,
                    label=f"rule engine (tuned) {models[0]['rule_macro_f1']:.4f}")
    ax_left.set_xticks(xs)
    ax_left.set_xticklabels(labels, fontsize=8)
    ax_left.set_ylim(0, 1.08)
    ax_left.set_ylabel("macro-F1 (test)")
    ax_left.set_title("Macro-F1 on test (n = 6,680)")
    ax_left.legend(fontsize=8, loc="lower right")
    ax_left.grid(axis="y", alpha=0.3)

    deltas = [m["delta"] for m in models]
    colors = [COLOR_POS if d >= 0 else COLOR_NEG for d in deltas]
    bars_d = ax_right.bar(xs, deltas, 0.6, color=colors)
    _annotate_bars(ax_right, bars_d, fmt="{:+.4f}", fontsize=8)
    ax_right.axhline(0.0, color="black", linewidth=0.9)
    limit = max(0.006, max(abs(d) for d in deltas) * 1.6)
    ax_right.set_ylim(-limit, limit)
    ax_right.set_xticks(xs)
    ax_right.set_xticklabels(labels, fontsize=8)
    ax_right.set_ylabel("Δ macro-F1 (fusion − AI)")
    ax_right.set_title("Chênh lệch so với AI-only (trục zoom)")
    ax_right.grid(axis="y", alpha=0.3)

    fig.suptitle("Hybrid fusion vs AI-only - 6 model (rule scores từ rule_engine_tuned)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_ci(models: List[Dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ys = np.arange(len(models))[::-1]

    for y, model in zip(ys, models):
        significant = model["significant"]
        color = COLOR_POS if significant and model["delta"] > 0 else (
            COLOR_NEG if significant else "#7f7f7f")
        marker = "*" if significant else "o"
        ax.errorbar(model["delta"], y,
                    xerr=[[model["delta"] - model["ci_low"]], [model["ci_high"] - model["delta"]]],
                    fmt=marker, color=color, ecolor=color, capsize=4, markersize=11 if significant else 7,
                    linewidth=1.6)
        p_text = "< 1e-4" if model["p"] < 1e-4 else f"p={model['p']:.3f}"
        ax.annotate(f"{model['delta']:+.4f}  [{model['ci_low']:+.4f}, {model['ci_high']:+.4f}]  {p_text}",
                    (model["ci_high"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5,
                    color="black" if significant else "#555555")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([m["short"] for m in models], fontsize=9)
    ax.set_xlabel("Δ macro-F1 = fusion − AI (paired bootstrap 95% CI, 10,000 resamples)")
    ax.set_title("Fusion có khác AI-only? - CI chứa 0 = không kết luận được (test n = 6,680)")
    ax.grid(axis="x", alpha=0.3)
    lo = min(m["ci_low"] for m in models)
    hi = max(m["ci_high"] for m in models)
    pad = (hi - lo) * 0.42 + 0.004
    ax.set_xlim(lo - pad, hi + pad)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="*", color=COLOR_POS, linestyle="None", markersize=12,
               label="significant (CI không chứa 0)"),
        Line2D([], [], marker="o", color="#7f7f7f", linestyle="None", markersize=7,
               label="không significant"),
    ], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variants(models: List[Dict[str, Any]], path: Path) -> None:
    labels = [m["short"] for m in models]
    xs = np.arange(len(models))
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    for index, ((key, label), color) in enumerate(zip(VARIANTS, colors)):
        deltas = [m["metrics"]["test"][key]["macro_f1"] - m["ai"]["macro_f1"] for m in models]
        offset = (index - (len(VARIANTS) - 1) / 2) * width
        bars = ax.bar([x + offset for x in xs], deltas, width, label=label, color=color)
        _annotate_bars(ax, bars, fmt="{:+.4f}", fontsize=6.5, rotation=90)

    ax.axhline(0.0, color="black", linewidth=0.9)
    limit = max(abs(m["metrics"]["test"][key]["macro_f1"] - m["ai"]["macro_f1"])
                for m in models for key, _ in VARIANTS) * 1.7 + 0.001
    ax.set_ylim(-limit, limit)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ macro-F1 so với AI-only")
    ax.set_title("Từng biến thể fusion so với AI-only trên test - 4 cách ghép, cả 4 tune trên validation")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(models: List[Dict[str, Any]], path: Path) -> None:
    labels = [m["short"] for m in models]
    xs = np.arange(len(models))
    width = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, label in zip(axes, CLASS_ROWS):
        ai_values = [m["ai"][f"{label}_f1"] for m in models]
        fusion_values = [m["fusion"][f"{label}_f1"] for m in models]
        bars_ai = ax.bar(xs - width / 2, ai_values, width, label="AI-only", color=COLOR_AI)
        bars_fu = ax.bar(xs + width / 2, fusion_values, width, label="fusion primary", color=COLOR_FUSION)
        _annotate_bars(ax, bars_ai, fontsize=6.5, rotation=90)
        _annotate_bars(ax, bars_fu, fontsize=6.5, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(f"F1 {label}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("F1 từng lớp: AI-only vs fusion primary (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw fusion-vs-AI comparison figures.")
    parser.add_argument("--fusion-dir", default=str(FUSION_DIR),
                        help="folder that holds <model>/metrics.json + fusion_significance.json")
    args = parser.parse_args(argv)

    root = Path(args.fusion_dir)
    models = load_models(root)

    outputs = {
        "ai_vs_fusion": root / "fusion_ai_vs_fusion.png",
        "delta_ci": root / "fusion_delta_ci.png",
        "variants": root / "fusion_variants_delta.png",
        "per_class": root / "fusion_per_class_f1.png",
    }
    plot_ai_vs_fusion(models, outputs["ai_vs_fusion"])
    plot_delta_ci(models, outputs["delta_ci"])
    plot_variants(models, outputs["variants"])
    plot_per_class(models, outputs["per_class"])
    for key, path in outputs.items():
        LOGGER.info("[29] wrote %s", path)

    significant = [m["short"] for m in models if m["significant"]]
    LOGGER.info("[29] %d/%d models significant: %s", len(significant), len(models), significant or "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
