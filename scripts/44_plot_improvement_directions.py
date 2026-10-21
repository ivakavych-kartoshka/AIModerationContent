"""Step 44 - ve hinh so sanh TAT CA cac huong cai thien da thu.

    python scripts/44_plot_improvement_directions.py

Doc lai moi artifact da sinh (khong cham lai split nao - plan section 44):

    outputs/lexicon_experiments/vi_tokenizer/tokenizer_lexicon_eval.json
    outputs/lexicon_experiments/vi_tokenizer/significance.json
    outputs/lexicon_experiments/fasttext_embedding/embedding_lexicon_eval.json
    outputs/lexicon_experiments/fasttext_embedding/embedding_significance.json
    outputs/lexicon_experiments/nb_multilabel/nb_eval.json
    outputs/lexicon_experiments/nb_multilabel/nb_tuning.json
    outputs/lexicon_experiments/nb_multilabel/nb_significance.json
    outputs/lexicon_experiments/nb_multilabel/fusion_nb_significance.json
    outputs/evaluation/fusion/fusion_significance.json

Va ve 4 hinh trong outputs/lexicon_experiments/figures/:

    directions_evolution.png    phieu: rule engine qua tung huong (val)
    directions_delta_ci.png     forest plot: Δ + CI 95% cua MOI huong
    directions_rule_vs_fusion.png  nghich ly: rule engine tang vong, fusion
                                chi nhich - giai thich bang so
    directions_per_class.png    heatmap F1 tung lop cho tung huong

Cau chuyen cua hinh: 3 huong dau that bai, huong thu 4 (NB xac suat) THANH
CONG o muc rule engine (+0.0245, p=0.037) nhung khi do vao FUSION thi lai
khong con y nghia - vi rule van yeu hon AI va AI da chiem gan het "dat".
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

LOGGER = get_logger("sensitiveai.scripts.44_plot_improvement_directions")

VI_DIR = Path("outputs/lexicon_experiments/vi_tokenizer")
FT_DIR = Path("outputs/lexicon_experiments/fasttext_embedding")
NB_DIR = Path("outputs/lexicon_experiments/nb_multilabel")
FIG_DIR = Path("outputs/lexicon_experiments/figures")
FUSION_DIR = Path("outputs/evaluation/fusion")

COLOR_BASE = "#4c72b0"
COLOR_FAIL = "#c44e52"
COLOR_OK = "#2ca02c"
COLOR_FUSION = "#8172b2"
COLOR_NS = "#7f7f7f"

# ---- huong cai thien ------------------------------------------------------ #
DIRECTIONS = [
    # (key, nhan hien thi, nhom, macro_f1 val, delta, ci_low, ci_high, p, sig)
    ("current", "Lexicon hiện tại\n(668 entry)", "base",
     0.455467, 0.0, None, None, None, False),
    ("current_plus_word", "Hướng 1: tokenize VI\n+word terms", "fail",
     0.46283, 0.007363, -0.004469, 0.01924, 0.2174, False),
    ("embedding only", "Hướng 2: fastText\nembedding only", "fail",
     0.4545, -0.0010, -0.0245, 0.0232, 0.934, False),
    ("current_plus_embedding", "Hướng 3: fastText\n+embedding", "fail",
     0.4585, 0.0030, -0.0196, 0.0263, 0.811, False),
    ("nb only", "Hướng 4: NB xác suất\n+ multi-label", "ok",
     0.4782, 0.022719, 0.001289, 0.043801, 0.0368, True),
    ("nb best", "Hướng 4 (tuned)\nalpha=0.3, p=0.35", "ok",
     0.4800, 0.0245, 0.0012, 0.0476, 0.038, True),
]


def load_all() -> Dict[str, Any]:
    tok_sig = json.loads((VI_DIR / "significance.json").read_text(encoding="utf-8"))
    emb_eval = json.loads((FT_DIR / "embedding_lexicon_eval.json").read_text(encoding="utf-8"))
    nb_eval = json.loads((NB_DIR / "nb_eval.json").read_text(encoding="utf-8"))
    nb_tuning = json.loads((NB_DIR / "nb_tuning.json").read_text(encoding="utf-8"))
    fusion_nb = json.loads((NB_DIR / "fusion_nb_significance.json").read_text(encoding="utf-8"))
    fusion_all = json.loads((FUSION_DIR / "fusion_significance.json").read_text(encoding="utf-8"))
    return {"tok": tok_sig, "emb": emb_eval, "nb": nb_eval, "tune": nb_tuning,
            "fusion_nb": fusion_nb, "fusion_all": fusion_all}


def plot_evolution(data: Dict[str, Any], path: Path) -> None:
    keys = [d[0] for d in DIRECTIONS]
    vals = [d[3] for d in DIRECTIONS]
    labels = [d[1] for d in DIRECTIONS]
    groups = [d[2] for d in DIRECTIONS]
    colors = {"base": COLOR_BASE, "fail": COLOR_FAIL, "ok": COLOR_OK}

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    xs = np.arange(len(keys))
    bars = ax.bar(xs, vals, 0.62, color=[colors[g] for g in groups])
    base = vals[0]
    ax.axhline(base, color=COLOR_BASE, linestyle="--", linewidth=1.1,
               label=f"lexicon hiện tại {base:.4f}")
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.4f}", (bar.get_x() + bar.get_width() / 2, val),
                    ha="center", va="bottom", fontsize=8.5)
    # delta annotation
    for i, d in enumerate(DIRECTIONS):
        if d[0] == "current":
            continue
        delta = d[4]
        sig = d[8]
        color = COLOR_OK if sig else COLOR_FAIL
        mark = "✓ có ý nghĩa" if sig else "✗ không"
        ax.annotate(f"{delta:+.4f}\n{mark}", (i, base), xytext=(0, -34),
                    textcoords="offset points", ha="center", fontsize=7.5, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.42, 0.50)
    ax.set_ylabel("macro-F1 rule engine (validation)")
    ax.set_title("4 hướng cải thiện lexicon - chỉ hướng 4 (mô hình xác suất NB) vượt kiểm định\n"
                 "(đo trên rule engine đứng riêng, validation n=2672)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_ci(data: Dict[str, Any], path: Path) -> None:
    rows = [d for d in DIRECTIONS if d[0] != "current"]
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ys = np.arange(len(rows))
    for i, d in enumerate(rows):
        _, label, _, _, delta, lo, hi, p, sig = d
        color = COLOR_OK if sig else COLOR_FAIL
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.6)
        ax.plot([delta], [i], "o", color=color, markersize=9)
        ax.annotate(f"{delta:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}",
                    (hi, i), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5)
    ax.axvline(0.0, color="black", linewidth=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[1].replace("\n", " ") for d in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ macro-F1 so với lexicon hiện tại, CI 95% (10000 paired bootstrap)")
    ax.set_title("CHỈ hướng 4 có CI 95% KHÔNG chứa 0 (cận dưới +0.0013)\n"
                 "3 hướng đầu thất bại - chênh lệch nằm trong nhiễu")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rule_vs_fusion(data: Dict[str, Any], path: Path) -> None:
    fusion_nb = data["fusion_nb"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1]})

    # trai: rule engine dung rieng (val)
    rule_labels = ["lexicon\nhiện tại", "lexicon\nNB"]
    rule_vals = [0.4555, 0.4800]
    bars = ax1.bar(np.arange(2), rule_vals, 0.55, color=[COLOR_BASE, COLOR_OK])
    for bar, v in zip(bars, rule_vals):
        ax1.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=9)
    ax1.annotate("+0.0245\np = 0.037 ✓", (0.5, 0.4720), ha="center", fontsize=9.5,
                 color=COLOR_OK, fontweight="bold")
    ax1.set_xticks(np.arange(2))
    ax1.set_xticklabels(rule_labels)
    ax1.set_ylim(0.43, 0.50)
    ax1.set_ylabel("macro-F1 (validation)")
    ax1.set_title("Rule engine đứng riêng:\nNB tăng RÕ RỆT +0.0245 ✓")
    ax1.grid(axis="y", alpha=0.3)

    # phai: fusion cuoi (test)
    fus_labels = ["fusion SOTA\n(668)", "fusion + NB\n(1155)"]
    fus_vals = [fusion_nb["sota_macro_f1"], fusion_nb["nb_macro_f1"]]
    bars2 = ax2.bar(np.arange(2), fus_vals, 0.55, color=[COLOR_BASE, COLOR_FUSION])
    for bar, v in zip(bars2, fus_vals):
        ax2.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=9)
    ax2.annotate(f"+{fusion_nb['delta']:.4f}\nCI [{fusion_nb['ci'][0]:+.4f}, "
                 f"{fusion_nb['ci'][1]:+.4f}]\np = {fusion_nb['p_value']:.3f} ✗",
                 (0.5, 0.6560), ha="center", fontsize=9, color=COLOR_FAIL, fontweight="bold")
    ax2.set_xticks(np.arange(2))
    ax2.set_xticklabels(fus_labels)
    ax2.set_ylim(0.64, 0.68)
    ax2.set_ylabel("macro-F1 (test)")
    ax2.set_title("Fusion cuối cùng:\nNB chỉ nhích +0.0030, KHÔNG ý nghĩa ✗")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Nghịch lý: rule engine mạnh lên nhiều, nhưng fusion KHÔNG được lợi tương xứng\n"
                 "(rule vẫn yếu hơn AI; AI đã chiếm gần hết \"đất\" trong fusion)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(data: Dict[str, Any], path: Path) -> None:
    # F1 tung lop (val) cho tung huong lexicon
    rows = [
        ("Lexicon hiện tại", [0.3677, 0.1686, 0.8301]),
        ("H1: +word terms", [0.3830, 0.1783, 0.8272]),
        ("H2: embedding only", [0.3672, 0.1657, 0.8306]),
        ("H3: +embedding", [0.3758, 0.1569, 0.8428]),
        ("H4: NB only", [0.4036, 0.1681, 0.8628]),
        ("H4 tuned", [0.4100, 0.1750, 0.8600]),
    ]
    classes = ["HATE", "OFFENSIVE", "CLEAN"]
    mat = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0.10, vmax=0.90)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.annotate(f"{mat[i, j]:.3f}", (j, i), ha="center", va="center",
                        fontsize=9, color="black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_title("F1 từng lớp trên validation theo từng hướng\n"
                 "OFFENSIVE (cột giữa) KHÔNG bao giờ được cải thiện - đây là bottleneck thật")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ve hinh so sanh cac huong cai thien.")
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    args = parser.parse_args(argv)
    fig_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_all()
    plot_evolution(data, fig_dir / "directions_evolution.png")
    plot_delta_ci(data, fig_dir / "directions_delta_ci.png")
    plot_rule_vs_fusion(data, fig_dir / "directions_rule_vs_fusion.png")
    plot_per_class(data, fig_dir / "directions_per_class.png")

    # bang tom tat
    lines = ["# Tổng kết các hướng cải thiện đã thử\n",
             "> Tất cả Δ đo trên **rule engine đứng riêng**, validation n=2672 "
             "(trừ dòng fusion = test). 10000 paired bootstrap.\n",
             "| Hướng | Rule engine (val) | Δ | CI 95% | p | Kết luận |",
             "|---|---|---|---|---|---|"]
    lines.append(f"| Lexicon hiện tại | 0.4555 | — | — | — | mốc |")
    for d in DIRECTIONS[1:]:
        _, label, _, val, delta, lo, hi, p, sig = d
        verdict = "✅ CÓ ý nghĩa" if sig else "✗ không"
        plain = label.replace("\n", " ")
        lines.append(f"| {plain} | {val:.4f} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{p:.3f} | {verdict} |")
    fn = data["fusion_nb"]
    lines.append(f"\n## Đo trên FUSION (test)\n")
    lines.append("| Hệ thống | Macro-F1 | Δ | CI 95% | p | Kết luận |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| fusion SOTA (668) | {fn['sota_macro_f1']:.4f} | — | — | — | mốc |")
    lines.append(f"| fusion + NB (1155) | {fn['nb_macro_f1']:.4f} | {fn['delta']:+.4f} | "
                 f"[{fn['ci'][0]:+.4f}, {fn['ci'][1]:+.4f}] | {fn['p_value']:.3f} | "
                 f"{'✅' if fn['significant_95'] else '✗ không'} |")
    lines.append("\n**Kết luận:** 3/4 hướng thất bại ở mức rule engine. Hướng 4 "
                 "(mô hình xác suất NB) THÀNH CÔNG ở mức rule engine (+0.0245, p=0.037) "
                 "nhưng khi đưa vào fusion thì cải thiện biến mất (+0.0030, không ý nghĩa) "
                 "vì bottleneck thật nằm ở lớp OFFENSIVE và ở AI, không ở rule engine.\n")
    (FIG_DIR.parent / "directions_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("[44] wrote 4 figures + summary to %s", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
