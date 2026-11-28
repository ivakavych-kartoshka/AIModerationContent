"""Step 30 - vẽ sơ đồ kiến trúc toàn hệ thống (AI + rule engine + fusion) cho paper.

    python scripts/30_plot_system_architecture.py
    python scripts/30_plot_system_architecture.py --out <png>

Paper-grade architecture figure -> ``papers/figures/system_architecture.png``:

    Vietnamese comment
        |-> NEURAL BRANCH   : mDeBERTa tokenizer -> encoder -> multi-view head -> softmax p
        |-> LEXICON RULE    : decode obfuscation -> 668-term match -> context factors -> r
        v
    fusion   p'_c = p_c + beta * r_c    (AI keeps weight 1; beta = (1-alpha)/alpha
                                          is tuned on validation only)
        |
    frozen per-channel thresholds                         (validation grid)
        |
    HATE / OFFENSIVE / CLEAN

Pure drawing script: no experimental artefact is read or re-scored
(plan section 44). Numbers quoted in --headline come from the frozen runs
(0.5335 rule / 0.6590 AI / 0.6647 best fusion, test macro-F1, n = 6,680).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.30_plot_system_architecture")

DEFAULT_OUT = Path("papers/figures/system_architecture.png")

# ---------------------------------------------------------------------------
# print-friendly palette (soft fills + saturated borders)
# ---------------------------------------------------------------------------
AI_EDGE, AI_FILL = "#3B6EA5", "#EAF1F8"
RULE_EDGE, RULE_FILL = "#D9822B", "#FDF2E5"
FUSION_EDGE, FUSION_FILL = "#4F8A4A", "#ECF4EB"
IO_EDGE, IO_FILL = "#8A8A8A", "#F4F4F4"
TEXT = "#1A1A1A"
ARROW = "#4A4A4A"


def rounded(ax, x0: float, y0: float, x1: float, y1: float, edge: str, fill: str,
            lw: float = 1.4, rounding: float = 1.8, zorder: int = 2) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0.0,rounding_size={rounding}",
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=zorder,
        )
    )


def arrow(ax, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=17,
            linewidth=1.8, color=ARROW, shrinkA=2, shrinkB=2, zorder=4,
        )
    )


def badge(ax, x: float, y: float, number: str, color: str) -> None:
    ax.add_patch(Circle((x, y), 1.55, facecolor=color, edgecolor="white",
                        linewidth=1.2, zorder=5))
    ax.text(x, y, number, ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=6)


def step_box(ax, cx: float, y0: float, y1: float, number: str, edge: str, fill: str,
             title: str, detail: str) -> None:
    x0, x1 = cx - 18, cx + 18
    rounded(ax, x0, y0, x1, y1, edge, fill)
    mid = (y0 + y1) / 2
    ax.text(cx, mid + 2.0, title, ha="center", va="center", fontsize=9.3,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(cx, mid - 2.2, detail, ha="center", va="center", fontsize=8.2,
            color="#3C3C3C", linespacing=1.4, zorder=3)
    badge(ax, x0, y1 - 0.5, number, edge)


def draw(out_path: Path, headline: Optional[Sequence[str]]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 8.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- input -----------------------------------------------------------
    rounded(ax, 34, 93, 66, 99, IO_EDGE, IO_FILL, lw=1.5)
    ax.text(50, 96, "Vietnamese social-media comment", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TEXT, zorder=3)

    # ---- branch containers ----------------------------------------------
    for x0, x1, edge, fill, title, subtitle in (
        (2, 46, AI_EDGE, AI_FILL, "NEURAL BRANCH",
         "mDeBERTa-v3-base encoder + multi-view head (6 training recipes)"),
        (54, 98, RULE_EDGE, RULE_FILL, "LEXICON RULE BRANCH",
         "668-term Vietnamese hate-speech lexicon"),
    ):
        rounded(ax, x0, 39, x1, 90, edge, fill, lw=1.6, rounding=2.4, zorder=1)
        cx = (x0 + x1) / 2
        ax.text(cx, 86.3, title, ha="center", va="center", fontsize=10.8,
                fontweight="bold", color=edge, zorder=3)
        ax.text(cx, 82.3, subtitle, ha="center", va="center", fontsize=8.4,
                color="#4A4A4A", zorder=3)

    # ---- neural branch steps --------------------------------------------
    left = 24
    step_box(ax, left, 70, 79, "1", AI_EDGE, AI_FILL,
             "Tokenize",
             "mDeBERTa SentencePiece\nmax length 256")
    step_box(ax, left, 56, 65, "2", AI_EDGE, AI_FILL,
             "Encoder  mDeBERTa-v3-base",
             "12 layers, hidden 768\nDLR $\\cdot$ focal loss $\\cdot$ class weights $\\cdot$ curriculum")
    step_box(ax, left, 41, 50, "3", AI_EDGE, AI_FILL,
             "Multi-view head",
             "CLS + mean + max pooling $\\rightarrow$ MLP(512)\n$p = (p_{HATE},\\ p_{OFF},\\ p_{CLEAN})$")

    # ---- rule branch steps ----------------------------------------------
    right = 76
    step_box(ax, right, 70, 79, "1", RULE_EDGE, RULE_FILL,
             "Normalise & decode",
             "strip accents $\\cdot$ leet $\\cdot$\nspaced $\\cdot$ repeated letters")
    step_box(ax, right, 56, 65, "2", RULE_EDGE, RULE_FILL,
             "Lexicon matching",
             "severity $\\times$ confidence $\\times$ match factor\nnegation window $\\cdot$ quoted spans")
    step_box(ax, right, 41, 50, "3", RULE_EDGE, RULE_FILL,
             "Saturating aggregation",
             "$1-\\prod(1-s_i)$ per label\n$r = (r_{HATE},\\ r_{OFF},\\ r_{CLEAN})$")

    # internal arrows
    for x in (left, right):
        arrow(ax, (x, 70), (x, 65.2))
        arrow(ax, (x, 56), (x, 50.2))

    # ---- fusion ----------------------------------------------------------
    rounded(ax, 24, 22, 76, 34.5, FUSION_EDGE, FUSION_FILL, lw=1.6)
    ax.text(50, 31.8, "HYBRID FUSION  (per class $c$)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=FUSION_EDGE, zorder=3)
    ax.text(50, 28.4, "$p'_c\\ =\\ p_c\\ +\\ \\beta \\cdot r_c$",
            ha="center", va="center", fontsize=12, color=TEXT, zorder=3)
    ax.text(50, 25.4,
            "AI keeps its full weight 1; the rule engine enters as a down-weighted correction",
            ha="center", va="center", fontsize=8.0, color="#3C3C3C", zorder=3)
    ax.text(50, 23.5,
            "$\\beta=(1-\\alpha)/\\alpha$ tuned on validation only (per-channel $\\beta$; meta-classifier variant)",
            ha="center", va="center", fontsize=8.0, color="#3C3C3C", zorder=3)

    # ---- decision layer --------------------------------------------------
    rounded(ax, 28, 12, 72, 19, IO_EDGE, IO_FILL)
    ax.text(50, 15.5, "Frozen per-channel thresholds   (validation, 101-point grid)",
            ha="center", va="center", fontsize=9.2, color=TEXT, zorder=3)

    # ---- output ----------------------------------------------------------
    rounded(ax, 32, 2, 68, 9, IO_EDGE, IO_FILL, lw=1.5)
    ax.text(50, 5.5, "HATE   /   OFFENSIVE   /   CLEAN", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TEXT, zorder=3)

    # ---- main arrows -----------------------------------------------------
    arrow(ax, (43, 93), (27, 90.2))
    arrow(ax, (57, 93), (73, 90.2))
    arrow(ax, (left, 41), (38, 34.9))
    arrow(ax, (right, 41), (62, 34.9))
    arrow(ax, (50, 22), (50, 19.2))
    arrow(ax, (50, 12), (50, 9.2))

    # ---- optional headline strip ----------------------------------------
    if headline:
        ax.text(50, -1.5, "   |   ".join(headline), ha="center", va="top",
                fontsize=8.2, color="#555555",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#FAFAFA",
                          edgecolor="#DDDDDD", linewidth=0.8))

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the whole-system architecture figure.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output PNG path")
    parser.add_argument("--no-headline", action="store_true",
                        help="omit the test macro-F1 strip at the bottom")
    args = parser.parse_args(argv)

    headline = None
    if not args.no_headline:
        headline = (
            "test macro-F1 (n = 6,680):  rule-only 0.5335  |  AI-only 0.6590  |  AI + fusion 0.6647",
            "all tunable parameters are chosen on validation; test is scored once",
        )
    out_path = Path(args.out)
    draw(out_path, headline)
    LOGGER.info("[30] wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
