"""Generate the composite analysis figures used by the paper.

Creates (in ``papers/figures/`` and mirrored under ``outputs/``):

* ``f1_summary_all_models.png`` - one 1x2 panel: (a) macro-F1 per system and
  (b) per-class F1 (HATE / OFFENSIVE / CLEAN) as grouped bars, for M1--M6.

The validation loss/accuracy pair figure already exists
(``validation_loss_accuracy_all_models.png``) and is only copied, not rebuilt.

Run::

    python scripts/07_paper_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.plots import setup_style, model_color, _save  # noqa: E402

CLASSES = ("HATE", "OFFENSIVE", "CLEAN")
SHORT = {
    "sensitiveai-vi": "M1",
    "sensitiveai-vi-custom": "M2",
    "sensitiveai-vi-customdlr2stage": "M3",
    "sensitiveai-vi-custom-curriculum": "M4",
    "sensitiveai-vi-customdlr2stage-ext": "M5",
    "sensitiveai-vi-stage2": "M6",
}
ORDER = list(SHORT)


def load_metrics() -> dict:
    summary = json.loads(
        (ROOT / "outputs/evaluation/comparison/comparison_summary.json").read_text(encoding="utf-8")
    )
    metrics = {}
    for name in ORDER:
        report_dir = Path(summary["metrics"][name]["report_dir"])
        metrics[name] = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    return metrics


def plot_f1_summary(metrics: dict, path: Path) -> Path:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})

    # (a) macro-F1 as horizontal bars ------------------------------------- #
    ax = axes[0]
    values = [float(metrics[n]["macro_f1"]) for n in ORDER]
    ypos = np.arange(len(ORDER))[::-1]
    bars = ax.barh(ypos, values, color=[model_color(n) for n in ORDER], height=0.66)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.0015, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", ha="left", fontsize=7.5)
    lo, hi = min(values), max(values)
    pad = max(hi - lo, 0.01)
    ax.set_xlim(max(0.0, lo - 1.5 * pad), min(1.0, hi + 1.7 * pad))
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT[n] for n in ORDER], fontsize=8)
    ax.set_xlabel("Macro-F1")
    ax.set_title("(a) Macro-F1", fontsize=9)

    # (b) per-class F1 as grouped bars ------------------------------------ #
    ax = axes[1]
    width = 0.8 / len(ORDER)
    x = np.arange(len(CLASSES))
    for i, name in enumerate(ORDER):
        vals = [float(metrics[name].get(f"{c}_f1", np.nan)) for c in CLASSES]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               color=model_color(name), label=SHORT[name])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_ylim(0, 1.34)
    ax.set_ylabel("F1-score")
    ax.set_title("(b) Per-class F1", fontsize=9)
    ax.legend(fontsize=6.5, ncol=6, loc="upper center", framealpha=0.9,
              columnspacing=0.8, handlelength=0.9, handletextpad=0.4)
    fig.tight_layout()
    return _save(fig, path)


def main() -> None:
    metrics = load_metrics()
    targets = [
        ROOT / "papers/figures/f1_summary_all_models.png",
        ROOT / "outputs/evaluation/comparison/f1_summary_all_models.png",
        ROOT / "outputs/paper/figures/f1_summary_all_models.png",
    ]
    plot_f1_summary(metrics, targets[0])
    for extra in targets[1:]:
        extra.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(targets[0], extra)

    # Make sure the two validation panels are present where the paper expects.
    val_src = ROOT / "outputs/evaluation/comparison/validation_loss_accuracy_all_models.png"
    for dst in (ROOT / "papers/figures/validation_loss_accuracy_all_models.png",
                ROOT / "outputs/paper/figures/validation_loss_accuracy_all_models.png"):
        if val_src.is_file() and not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(val_src, dst)

    print("wrote", targets[0])


if __name__ == "__main__":
    main()
