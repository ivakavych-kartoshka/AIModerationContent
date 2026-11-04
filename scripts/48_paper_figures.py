"""Build the paper's figures from the frozen ICCIES numbers.

Outputs (papers/iccies2027/figures/):
    architecture.pdf   two-branch hybrid diagram, with a worked example
    results.pdf        three-panel quantitative summary

Run from the repository root:
    $env:PYTHONIOENCODING="utf-8"
    python scripts/48_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "outputs/evaluation/fusion/iccies2027/numbers.json"
# The alpha printed inside the fusion box is read from the frozen fusion run, so
# the diagram cannot drift from the numbers the paper reports.
FUSION_METRICS = ROOT / "outputs/evaluation/fusion/sensitiveai-vi-custom/metrics.json"
OUT = ROOT / "papers/iccies2027/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)

BLUE = "#3B6EA5"
ORANGE = "#C77B3C"
GREEN = "#4E8B57"
GRAY = "#5B5B5B"
LIGHT = "#EDEDED"
RED = "#B0453F"


def load() -> dict:
    with NUMBERS.open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, title, body=None, color=GRAY, light="#FFFFFF", lw=1.15,
        title_dy=0.24, body_dy=-0.26, tsize=9.0, bsize=8.2):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            linewidth=lw,
            edgecolor=color,
            facecolor=light,
            zorder=2,
        )
    )
    if body:
        ax.text(x, y + title_dy, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color="#111111", zorder=3,
                linespacing=1.15)
        ax.text(x, y + body_dy, body, ha="center", va="center",
                fontsize=bsize, color="#111111", zorder=3, linespacing=1.4)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=tsize, color="#111111", zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, color=GRAY, style="-|>", lw=1.15, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=10, linewidth=lw,
            color=color, linestyle=ls, shrinkA=1, shrinkB=1, zorder=1,
        )
    )


def draw_architecture() -> None:
    """Two-branch diagram, drawn at the width it is printed at.

    The figure is included at ``\\textwidth`` (122 mm), so the canvas is exactly
    4.81 in wide: nothing is scaled down on the page and every label keeps its
    nominal point size.  The aspect ratio is held at 2.05 so the typeset height
    stays 169 pt and the page budget does not move.
    """
    fig, ax = plt.subplots(figsize=(4.81, 2.352))
    ax.set_xlim(0, 13.9)
    ax.set_ylim(0.10, 6.00)
    ax.axis("off")

    # input
    box(ax, 1.00, 3.00, 1.85, 1.30, "comment $x$", "“lủ ch*”",
        color=GRAY, tsize=8.8)

    # neural branch
    box(ax, 4.19, 4.55, 2.58, 1.22, "mDeBERTa-v3", "multi-pool head",
        color=BLUE, light="#E8F0F9")
    box(ax, 7.05, 4.55, 2.12, 1.22, "softmax", "$p_H,\\ \\ p_O,\\ \\ p_C$",
        color=BLUE, light="#E8F0F9")

    # rule branch
    box(ax, 4.19, 1.50, 2.58, 1.30, "normalise", "4 matching views",
        color=ORANGE, light="#FBF0E5")
    box(ax, 7.05, 1.50, 2.12, 1.30, "matches", "$r_H,\\ r_O$ + spans",
        color=ORANGE, light="#FBF0E5")

    # fusion and decision: the per-channel weights live inside the fusion box
    # so that they never sit on top of the two incoming arrows
    alpha = json.loads(FUSION_METRICS.read_text(encoding="utf-8"))["per_channel_alphas"]
    alpha_txt = (
        "$\\alpha_H{=}%.1f$   $\\alpha_O{=}%.1f$\n$\\alpha_C{=}%.1f$"
        % (alpha["HATE"], alpha["OFFENSIVE"], alpha["CLEAN"])
    )
    box(ax, 9.85, 3.00, 2.85, 1.85, "per-channel\nfusion",
        alpha_txt,
        color=GREEN, light="#EAF2EA", title_dy=0.48, body_dy=-0.33,
        tsize=8.8, bsize=8.0)
    box(ax, 12.86, 3.00, 1.95, 1.85, "decision",
        "$\\tau_c$ frozen\n$\\to\\ \\hat y$",
        color=GRAY, light=LIGHT, title_dy=0.52, body_dy=-0.30,
        tsize=8.8, bsize=8.2)

    # evidence record
    ax.add_patch(
        FancyBboxPatch(
            (8.75, 0.28), 5.00, 1.35,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            linewidth=1.0, edgecolor=GRAY, facecolor="#F7F7F7",
            linestyle=(0, (3.5, 2.2)), zorder=2,
        )
    )
    ax.text(11.25, 1.42, "evidence record", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#111111", zorder=3)
    ax.text(11.25, 0.86,
            "span $\\cdot$ term $\\cdot$ match type $\\cdot$ score\n"
            "$r_H,\\ r_O$ $\\cdot$ $\\alpha_c$ $\\cdot$ $f_H,\\ f_O,\\ f_C$",
            ha="center", va="center", fontsize=7.4, color="#111111",
            linespacing=1.45, zorder=3)

    # arrows: split input to branches, through branches, to fusion
    arrow(ax, (1.98, 3.00), (2.45, 3.00), color=GRAY, style="-", lw=1.3)
    ax.plot([2.45, 2.45], [1.50, 4.55], color=GRAY, lw=1.3, zorder=1,
            solid_capstyle="butt")
    arrow(ax, (2.45, 4.55), (2.90, 4.55), color=GRAY, lw=1.3)
    arrow(ax, (2.45, 1.50), (2.90, 1.50), color=GRAY, lw=1.3)
    arrow(ax, (5.50, 4.55), (5.95, 4.55), color=BLUE)
    arrow(ax, (5.50, 1.50), (5.95, 1.50), color=ORANGE)
    arrow(ax, (8.11, 4.55), (8.40, 3.68), color=BLUE)
    arrow(ax, (8.11, 1.50), (8.40, 2.34), color=ORANGE)
    arrow(ax, (11.28, 3.00), (11.82, 3.00), color=GREEN, lw=1.3)
    arrow(ax, (12.86, 2.05), (12.86, 1.67), color=GRAY, ls=(0, (3.5, 2.2)))

    # branch labels
    ax.text(2.55, 5.38, "neural branch", fontsize=8.4, color=BLUE,
            fontweight="bold", va="center")
    ax.text(2.55, 0.50, "rule branch", fontsize=8.4, color=ORANGE,
            fontweight="bold", va="center")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "architecture.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "architecture.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------
# Quantitative summary
# --------------------------------------------------------------------------
def draw_results(n: dict) -> None:
    """Three panels, placed with explicit inch offsets.

    Default gridspec spacing let the long ``HATE <- CLEAN`` labels of panel (b)
    run into the value labels of panel (a), so the axes are placed by hand here:
    the gaps reserve exactly the room each tick label block needs.
    """
    W, H = 5.35, 2.12
    fig = plt.figure(figsize=(W, H))

    def rect(x0_in: float, w_in: float) -> list[float]:
        return [x0_in / W, 0.42 / H, w_in / W, 1.32 / H]

    ax_a = fig.add_axes(rect(0.88, 1.16))   # 0.88 in of y labels at the left
    ax_b = fig.add_axes(rect(2.62, 0.96))   # clears (a) and (b) label blocks
    ax_c = fig.add_axes(rect(4.20, 0.94))

    # (a) macro-F1 across decision variants ----------------------------------
    ax = ax_a
    v = n["fusion_variants"]
    order = [
        ("rule_only_argmax", "Rule only (argmax)"),
        ("rule_only_frozen_thresholds", "Rule only (thr.)"),
        ("fusion_meta_classifier", "Meta-classifier"),
        ("ai_only_argmax", "AI only"),
        ("fusion_argmax", "Fusion, global $\\alpha$"),
        ("fusion_per_channel_argmax", "Fusion, per-channel"),
        ("fusion_per_channel_thresholds", "Primary (thr.)"),
    ]
    labels = [lab for _, lab in order]
    vals = [v[key]["macro_f1"] for key, _ in order]
    colors = [GRAY if key.startswith("rule") else BLUE for key, _ in order]
    colors[-1] = GREEN
    ypos = list(range(len(vals)))
    bars = ax.barh(ypos, vals, color=colors, height=0.66,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.7)
    ax.set_ylim(len(vals) - 0.35, -0.65)
    ax.set_xlim(0.42, 0.722)
    ax.set_xlabel("Macro-F1", fontsize=7.0, labelpad=1.5)
    ax.set_title("(a) Decision variants", loc="left", fontsize=7.5,
                 fontweight="bold", pad=3)
    for b, val in zip(bars, vals):
        yc = b.get_y() + b.get_height() / 2
        if val > 0.60:      # label fits inside the bar, in white
            ax.text(val - 0.009, yc, f"{val:.4f}", va="center", ha="right",
                    fontsize=6.6, color="white", zorder=4)
        else:
            ax.text(val + 0.004, yc, f"{val:.4f}", va="center", ha="left",
                    fontsize=6.6, color="#1A1A1A", zorder=4)
    ax.axvline(v["ai_only_argmax"]["macro_f1"], color=BLUE, lw=0.7, ls=":",
               alpha=0.6, zorder=2)

    # (b) provenance of the changed decisions --------------------------------
    ax = ax_b
    fixed = n["fixed_transitions"]
    broken = n["broken_transitions"]
    rows = [
        ("HATE$\\leftarrow$OFF", fixed["HATE<-OFFENSIVE"], ORANGE, 0),
        ("HATE$\\leftarrow$CLEAN", fixed["HATE<-CLEAN"], ORANGE, 1),
        ("CLEAN$\\leftarrow$HATE", fixed["CLEAN<-HATE"], BLUE, 2),
        ("CLEAN$\\leftarrow$OFF", fixed["CLEAN<-OFFENSIVE"], BLUE, 3),
        ("OFF$\\to$HATE", broken["OFFENSIVE->HATE"], RED, 5),
        ("CLEAN$\\to$HATE", broken["CLEAN->HATE"], RED, 6),
        ("HATE$\\to$CLEAN", broken["HATE->CLEAN"], GRAY, 7),
    ]
    bars = ax.barh([r[3] for r in rows], [r[1] for r in rows], height=0.62,
                   color=[r[2] for r in rows], edgecolor="white", linewidth=0.5,
                   zorder=3)
    ax.set_yticks([r[3] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    ax.set_ylim(7.75, -0.65)
    ax.set_xlim(0, 18)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Changed decisions", fontsize=7.0, labelpad=1.5)
    ax.set_title("(b) Provenance of changes", loc="left", fontsize=7.5,
                 fontweight="bold", pad=3)
    for b, val in zip(bars, [r[1] for r in rows]):
        ax.text(val + 0.35, b.get_y() + b.get_height() / 2, str(val),
                va="center", ha="left", fontsize=6.7)
    ax.axhline(4.0, color=GRAY, lw=0.6, ls=(0, (1.2, 1.6)), zorder=2)
    ax.text(16.9, 1.5, "corrections", rotation=90, ha="center", va="center",
            fontsize=6.2, color=GREEN, fontweight="bold")
    ax.text(16.9, 6.0, "regressions", rotation=90, ha="center", va="center",
            fontsize=6.2, color=RED, fontweight="bold")

    # (c) rule channel behaviour ---------------------------------------------
    ax = ax_c
    rc = n["rule_channel"]
    metrics = [
        ("Coverage", rc["coverage_pct"], ORANGE),
        ("Precision", rc["precision_on_firing_pct"], GREEN),
        ("Recall", rc["recall_nonclean_pct"], GREEN),
        ("False alarms", rc["false_alarm_clean_pct"], RED),
    ]
    ypos = list(range(len(metrics)))
    bars = ax.barh(ypos, [m[1] for m in metrics], height=0.60,
                   color=[m[2] for m in metrics], edgecolor="white",
                   linewidth=0.6, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=6.4)
    ax.set_ylim(len(metrics) - 0.35, -0.65)
    ax.set_xlim(0, 85)
    ax.set_xticks([0, 40, 80])
    ax.set_xlabel("Percent", fontsize=7.0, labelpad=1.5)
    ax.set_title("(c) Rule channel", loc="left", fontsize=7.5,
                 fontweight="bold", pad=3)
    for b, (_, val, _) in zip(bars, metrics):
        ax.text(val + 1.6, b.get_y() + b.get_height() / 2, f"{val:.1f}",
                va="center", ha="left", fontsize=6.5)

    fig.savefig(OUT / "results.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "results.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    draw_architecture()
    draw_results(data)
    print("wrote", OUT / "architecture.pdf")
    print("wrote", OUT / "results.pdf")
