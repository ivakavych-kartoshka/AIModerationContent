"""Step 32 - hình so sánh kết quả ensemble (plan 43-44).

    python scripts/32_plot_ensemble_comparison.py

Reads the frozen artefacts that ``scripts/31_ensemble_experiment.py`` already
wrote (``ensemble_metrics.json`` + ``significance.json`` under
``outputs/evaluation/ensemble/``), plus the stored metrics of the two
references (curriculum fusion primary, best single model), and draws four
figures beside them:

    ensemble_test_macro_f1.png    bar test macro-F1: 4 ứng viên + 2 tham chiếu
    ensemble_val_vs_test.png      val vs test từng ứng viên (đảo chiều)
    ensemble_delta_ci.png         forest plot: Δ (ứng viên − SOTA) + CI 95%
    ensemble_per_class_f1.png     F1 từng lớp: 4 ứng viên + 2 tham chiếu

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
from matplotlib.lines import Line2D  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.32_plot_ensemble_comparison")

ENSEMBLE_DIR = Path("outputs/evaluation/ensemble")
FUSION_DIR = Path("outputs/evaluation/fusion")
PER_MODEL_DIR = Path("outputs/evaluation/per_model")
SOTA_MODEL = "sensitiveai-vi-custom-curriculum"
SINGLE_MODEL = "sensitiveai-vi-customdlr2stage"

COLOR_ENS = "#1f77b4"      # ứng viên không có rule
COLOR_ENS_RULE = "#ff7f0e"  # ứng viên có cộng rule
COLOR_SOTA = "#2ca02c"
COLOR_SINGLE = "#9467bd"
COLOR_RULE = "#9e9e9e"
COLOR_POS = "#2ca02c"
COLOR_NEG = "#d62728"
COLOR_NS = "#7f7f7f"

CANDIDATE_ORDER = ("ens3", "ens6", "ens3_rule", "ens6_rule")
CANDIDATE_LABELS = {
    "ens3": "ens3\n(3 model)",
    "ens6": "ens6\n(6 model)",
    "ens3_rule": "ens3 + rule\n(β=0.11)",
    "ens6_rule": "ens6 + rule\n(β=0.25)",
}


def load_everything(root: Path) -> Dict[str, Any]:
    metrics = json.loads((root / "ensemble_metrics.json").read_text(encoding="utf-8"))
    significance = json.loads((root / "significance.json").read_text(encoding="utf-8"))

    fusion = json.loads((FUSION_DIR / SOTA_MODEL / "metrics.json").read_text(encoding="utf-8"))
    sota_test = fusion["test"][fusion["primary_variant"]]
    rule_only = fusion["test"]["rule_only_frozen_thresholds"]

    single = json.loads((PER_MODEL_DIR / SINGLE_MODEL / "metrics.json").read_text(encoding="utf-8"))

    return {
        "metrics": metrics,
        "significance": significance,
        "sota": {
            "macro_f1": sota_test["macro_f1"],
            "accuracy": sota_test["accuracy"],
            "HATE_f1": sota_test["HATE_f1"],
            "OFFENSIVE_f1": sota_test["OFFENSIVE_f1"],
            "CLEAN_f1": sota_test["CLEAN_f1"],
        },
        "single": {
            "macro_f1": single["macro_f1"],
            "accuracy": single["accuracy"],
            "HATE_f1": single["HATE_f1"],
            "OFFENSIVE_f1": single["OFFENSIVE_f1"],
            "CLEAN_f1": single["CLEAN_f1"],
        },
        "rule_only_macro_f1": rule_only["macro_f1"],
    }


def _annotate_bars(ax: Any, bars: Any, fmt: str = "{:.4f}", fontsize: float = 7.5,
                   rotation: float = 0.0) -> None:
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                    fontsize=fontsize, rotation=rotation)


def plot_test_macro_f1(data: Dict[str, Any], path: Path) -> None:
    metrics = data["metrics"]
    primary = metrics["primary"]
    sota_f1 = data["sota"]["macro_f1"]

    names = list(CANDIDATE_ORDER)
    labels = [CANDIDATE_LABELS[n] for n in names] + ["SOTA cũ\n(curriculum\n+ fusion)", "Model đơn\ntốt nhất\n(dlr2stage)"]
    xs = np.arange(len(names) + 2)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})

    values = [metrics["test"][n]["macro_f1"] for n in names] + [sota_f1, data["single"]["macro_f1"]]
    colors = [COLOR_ENS_RULE if n.endswith("_rule") else COLOR_ENS for n in names] + [COLOR_SOTA, COLOR_SINGLE]
    bars = ax_left.bar(xs, values, 0.62, color=colors)
    _annotate_bars(ax_left, bars, fontsize=8)
    ax_left.axhline(data["rule_only_macro_f1"], color=COLOR_RULE, linestyle="--", linewidth=1.2,
                    label=f"rule engine (tuned) {data['rule_only_macro_f1']:.4f}")
    star_x = names.index(primary)
    ax_left.annotate("★ ứng viên chính\n(chọn theo VAL)", (star_x, metrics["test"][primary]["macro_f1"]),
                     xytext=(0, 42), textcoords="offset points", ha="center", fontsize=8, color=COLOR_NEG)
    ax_left.set_xticks(xs)
    ax_left.set_xticklabels(labels, fontsize=8)
    ax_left.set_ylim(0, 1.08)
    ax_left.set_ylabel("macro-F1 (test)")
    ax_left.set_title("Macro-F1 trên TEST (n = 6,680) - chấm một lần")
    ax_left.legend(fontsize=8, loc="lower right")
    ax_left.grid(axis="y", alpha=0.3)

    deltas = [metrics["test"][n]["macro_f1"] - sota_f1 for n in names]
    colors_d = [COLOR_POS if d >= 0 else COLOR_NEG for d in deltas]
    bars_d = ax_right.bar(np.arange(len(names)), deltas, 0.6, color=colors_d)
    _annotate_bars(ax_right, bars_d, fmt="{:+.4f}", fontsize=8.5)
    for i, name in enumerate(names):
        block = data["significance"].get(f"primary_vs_sota" if name == primary else f"secondary_{name}_vs_sota")
        if block is None:
            ax_right.annotate("chưa kiểm định", (i, deltas[i]), xytext=(0, -14 if deltas[i] < 0 else 14),
                              textcoords="offset points", ha="center", fontsize=7, color=COLOR_NS, style="italic")
    ax_right.axhline(0.0, color="black", linewidth=0.9)
    limit = max(0.006, max(abs(d) for d in deltas) * 1.9)
    ax_right.set_ylim(-limit, limit)
    ax_right.set_xticks(np.arange(len(names)))
    ax_right.set_xticklabels([CANDIDATE_LABELS[n].replace("\n", " ") for n in names], fontsize=7.5)
    ax_right.set_ylabel("Δ macro-F1 (ứng viên − SOTA cũ)")
    ax_right.set_title("Chênh lệch so với SOTA cũ (trục zoom)")
    ax_right.grid(axis="y", alpha=0.3)

    fig.suptitle("Ensemble 6 model AI (+ cộng điểm rule) vs hệ thống hiện tại - mọi tham số đóng băng trên validation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_val_vs_test(data: Dict[str, Any], path: Path) -> None:
    metrics = data["metrics"]
    names = list(CANDIDATE_ORDER)
    xs = np.arange(len(names))
    width = 0.36

    val_values = [metrics["validation"][n]["val_macro_f1_thresholded"] for n in names]
    test_values = [metrics["test"][n]["macro_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars_val = ax.bar(xs - width / 2, val_values, width, label="validation (n = 2,672) - nơi chọn tham số",
                      color="#c6dbef", edgecolor=COLOR_ENS)
    bars_test = ax.bar(xs + width / 2, test_values, width, label="test (n = 6,680) - chấm một lần",
                       color=COLOR_ENS, alpha=0.85)
    _annotate_bars(ax, bars_val, fontsize=7.5)
    _annotate_bars(ax, bars_test, fontsize=7.5)

    val_best = max(names, key=lambda n: metrics["validation"][n]["val_macro_f1_thresholded"])
    test_best = max(names, key=lambda n: metrics["test"][n]["macro_f1"])
    ax.annotate(f"thắng VAL: {val_best}", (names.index(val_best), metrics["validation"][val_best]["val_macro_f1_thresholded"]),
                xytext=(-30, 30), textcoords="offset points", fontsize=8.5, color=COLOR_NEG, style="italic")
    ax.annotate(f"thắng TEST: {test_best}", (names.index(test_best) + 0.18, metrics["test"][test_best]["macro_f1"]),
                xytext=(-15, 46), textcoords="offset points", fontsize=8.5, color=COLOR_POS, style="italic")

    ax.axhline(data["sota"]["macro_f1"], color=COLOR_SOTA, linestyle="--", linewidth=1.2,
               label=f"SOTA cũ trên test {data['sota']['macro_f1']:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([CANDIDATE_LABELS[n] for n in names], fontsize=8.5)
    ax.set_ylim(0.60, 0.76)
    ax.set_ylabel("macro-F1")
    ax.set_title("Validation chọn ens3_rule, test lại thích ens6 - thứ hạng ĐẢO CHIỀU (trục zoom)")
    ax.legend(fontsize=8.5, loc="upper right", ncol=1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_ci(data: Dict[str, Any], path: Path) -> None:
    metrics = data["metrics"]
    significance = data["significance"]
    sota_f1 = data["sota"]["macro_f1"]
    primary = metrics["primary"]

    rows: List[Dict[str, Any]] = []
    for name in ("ens3_rule", "ens6", "ens3", "ens6_rule"):
        block = significance.get(f"primary_vs_{name}") if name == primary else significance.get(f"secondary_{name}_vs_sota")
        if name == primary:
            block = significance.get("primary_vs_sota")
        delta = metrics["test"][name]["macro_f1"] - sota_f1
        if block is not None and "paired_bootstrap_macro_f1" in block:
            pb = block["paired_bootstrap_macro_f1"]
            rows.append({"name": name, "delta": pb["delta"], "ci_low": pb["ci_low"],
                         "ci_high": pb["ci_high"], "p": pb["p_value"], "kind": block.get("note", "")})
        else:
            rows.append({"name": name, "delta": delta, "ci_low": None, "ci_high": None,
                         "p": None, "kind": "point-only"})

    fig, ax = plt.subplots(figsize=(10, 5.0))
    ys = np.arange(len(rows))[::-1]

    for y, row in zip(ys, rows):
        label = CANDIDATE_LABELS[row["name"]].replace("\n", " ")
        if row["ci_low"] is None:
            ax.plot(row["delta"], y, marker="D", color=COLOR_NS, fillstyle="none", markersize=7)
            ax.annotate(f"{row['delta']:+.4f}   (chưa kiểm định - CI chưa tính)", (row["delta"], y),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=COLOR_NS)
            continue
        significant = row["ci_low"] > 0.0 or row["ci_high"] < 0.0
        color = COLOR_POS if significant and row["delta"] > 0 else (COLOR_NEG if significant else COLOR_NS)
        marker = "*" if significant else "o"
        ax.errorbar(row["delta"], y,
                    xerr=[[row["delta"] - row["ci_low"]], [row["ci_high"] - row["delta"]]],
                    fmt=marker, color=color, ecolor=color, capsize=4,
                    markersize=12 if significant else 7, linewidth=1.6)
        p_text = "< 1e-4" if row["p"] < 1e-4 else f"p={row['p']:.3f}"
        suffix = "  (ứng viên chính, chọn theo VAL)" if row["name"] == primary else "  (post-hoc)"
        ax.annotate(f"{row['delta']:+.4f}  [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {p_text}{suffix}",
                    (row["ci_high"], y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color="black" if significant else "#555555")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([CANDIDATE_LABELS[r["name"]].replace("\n", " ") for r in rows], fontsize=9)
    ax.set_xlabel("Δ macro-F1 = ứng viên − SOTA cũ (paired bootstrap 95% CI, 10,000 resamples)")
    ax.set_title("Ensemble khác SOTA cũ bao nhiêu? - CI chứa 0 = không kết luận được (test n = 6,680)")
    ax.grid(axis="x", alpha=0.3)
    lows = [r["ci_low"] for r in rows if r["ci_low"] is not None] + [r["delta"] for r in rows]
    highs = [r["ci_high"] for r in rows if r["ci_high"] is not None] + [r["delta"] for r in rows]
    pad = (max(highs) - min(lows)) * 0.30 + 0.004
    ax.set_xlim(min(lows) - pad, max(highs) + pad)
    ax.legend(handles=[
        Line2D([], [], marker="*", color=COLOR_POS, linestyle="None", markersize=12,
               label="significant (CI không chứa 0)"),
        Line2D([], [], marker="o", color=COLOR_NS, linestyle="None", markersize=7,
               label="không significant (CI chứa 0)"),
        Line2D([], [], marker="D", color=COLOR_NS, linestyle="None", markersize=7, fillstyle="none",
               label="chưa kiểm định (chỉ điểm ước lượng)"),
    ], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(data: Dict[str, Any], path: Path) -> None:
    metrics = data["metrics"]
    names = list(CANDIDATE_ORDER)
    classes = ("HATE", "OFFENSIVE", "CLEAN")
    xs = np.arange(len(names) + 2)
    width = 0.13

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, cls in zip(axes, classes):
        values = ([metrics["test"][n][f"{cls}_f1"] for n in names]
                  + [data["sota"][f"{cls}_f1"], data["single"][f"{cls}_f1"]])
        colors = ([COLOR_ENS_RULE if n.endswith("_rule") else COLOR_ENS for n in names]
                  + [COLOR_SOTA, COLOR_SINGLE])
        offsets = (np.arange(len(xs)) - (len(xs) - 1) / 2) * width
        bars = ax.bar(xs + offsets, values, width, color=colors)
        _annotate_bars(ax, bars, fontsize=6, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([CANDIDATE_LABELS[n] for n in names] + ["SOTA cũ", "dlr2stage"],
                           fontsize=6.5, rotation=25, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(f"F1 {cls}")
        ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ENS),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ENS_RULE),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SOTA),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SINGLE),
    ]
    fig.legend(handles, ["ens3 / ens6", "ens3_rule / ens6_rule", "SOTA cũ (curriculum + fusion)",
                         "model đơn tốt nhất (dlr2stage)"],
               fontsize=8.5, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("F1 từng lớp trên TEST - ensemble vẫn yếu ở OFFENSIVE như mọi hệ thống khác", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw ensemble comparison figures from frozen artefacts.")
    parser.add_argument("--ensemble-dir", default=str(ENSEMBLE_DIR))
    args = parser.parse_args(argv)

    root = Path(args.ensemble_dir)
    data = load_everything(root)

    outputs = {
        "test_macro_f1": root / "ensemble_test_macro_f1.png",
        "val_vs_test": root / "ensemble_val_vs_test.png",
        "delta_ci": root / "ensemble_delta_ci.png",
        "per_class": root / "ensemble_per_class_f1.png",
    }
    plot_test_macro_f1(data, outputs["test_macro_f1"])
    plot_val_vs_test(data, outputs["val_vs_test"])
    plot_delta_ci(data, outputs["delta_ci"])
    plot_per_class(data, outputs["per_class"])
    for path in outputs.values():
        LOGGER.info("[32] wrote %s", path)

    sig = data["significance"]
    pb = sig["primary_vs_sota"]["paired_bootstrap_macro_f1"]
    LOGGER.info("[32] primary %s: delta %+.4f CI [%.4f, %.4f] vs SOTA %.4f",
                sig["primary"], pb["delta"], pb["ci_low"], pb["ci_high"], data["sota"]["macro_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
