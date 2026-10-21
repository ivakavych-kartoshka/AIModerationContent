"""Step 39 - ve hinh so sanh TAT CA bien the lexicon (rule engine, val).

    python scripts/39_plot_lexicon_variants.py

Doc lai cac artifact da sinh (khong cham lai split nao - plan section 44):
    outputs/lexicon_experiments/vi_tokenizer/tokenizer_lexicon_eval.json
    outputs/lexicon_experiments/vi_tokenizer/significance.json
    outputs/lexicon_experiments/fasttext_embedding/embedding_lexicon_eval.json
    outputs/lexicon_experiments/fasttext_embedding/embedding_significance.json

Va ve 4 hinh trong outputs/lexicon_experiments/figures/:

    lexicon_macro_f1.png        cot: macro-F1 val cua moi bien the lexicon
    lexicon_per_class_f1.png    F1 tung lop: moi bien the
    lexicon_delta_ci.png        forest plot: Δ vs current + CI 95%
    lexicon_mcnemar.png         so mau doi ket qua (dung/sai) - giai thich
                                vi sao macro-F1 khong nhuc nhich du accuracy len

Muc dich: nhin MOT cho de tra loi cau hoi "co cai thien hon khong" - va cau
tra loi la KHONG, voi bang chung CI + McNemar di kem.
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

LOGGER = get_logger("sensitiveai.scripts.39_plot_lexicon_variants")

VI_DIR = Path("outputs/lexicon_experiments/vi_tokenizer")
FT_DIR = Path("outputs/lexicon_experiments/fasttext_embedding")
FIG_DIR = Path("outputs/lexicon_experiments/figures")

COLOR_CURRENT = "#4c72b0"
COLOR_TOKENIZER = "#dd8452"
COLOR_EMBEDDING = "#55a868"
COLOR_POS = "#2ca02c"
COLOR_NEG = "#d62728"

# thu tu hien thi: current truoc, roi cac bien the
LABELS = {
    "current": "current\n(668)",
    "current_plus_word": "+word terms\n(929)",
    "word_only": "word-only\n(261)",
    "embedding only": "embedding only\n(1207)",
    "current + embedding": "+embedding\n(1272)",
}
GROUPS = {
    "current": "current",
    "current_plus_word": "tokenizer",
    "word_only": "tokenizer",
    "embedding only": "embedding",
    "current + embedding": "embedding",
}
COLORS = {"current": COLOR_CURRENT, "tokenizer": COLOR_TOKENIZER, "embedding": COLOR_EMBEDDING}


def load_all() -> Dict[str, Any]:
    tok = json.loads((VI_DIR / "tokenizer_lexicon_eval.json").read_text(encoding="utf-8"))
    tok_sig = json.loads((VI_DIR / "significance.json").read_text(encoding="utf-8"))
    emb = json.loads((FT_DIR / "embedding_lexicon_eval.json").read_text(encoding="utf-8"))
    emb_sig = json.loads((FT_DIR / "embedding_significance.json").read_text(encoding="utf-8"))

    variants: Dict[str, Dict[str, Any]] = {}
    # tokenizer eval: key = friendly name
    variants["current"] = _mk(tok["current (verified_lexicon.csv)"])
    variants["current_plus_word"] = _mk(tok["current + word terms"])
    variants["word_only"] = _mk(tok["word-mode terms only"])

    er = emb["results"]
    variants["embedding only"] = _mk(er["embedding only"])
    variants["current + embedding"] = _mk(er["current + embedding"])

    sig: Dict[str, Dict[str, Any]] = {}
    for name, block in tok_sig["comparisons"].items():
        b = block["paired_bootstrap"]
        m = block["mcnemar"]
        sig[name] = {
            "delta": b["delta"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
            "p_value": b["p_value"],
            "right": m["a_correct_b_wrong"], "wrong": m["a_wrong_b_correct"],
            "sig": block["significant_95"],
        }
    for name, block in emb_sig.items():
        key = "embedding only" if name == "embedding only" else "current_embedding"
        if "embedding only" in name:
            key = "embedding only"
        elif name.strip().startswith("current + embedding"):
            key = "current_plus_embedding"
        else:
            key = name
        sig[key] = {
            "delta": block["delta"], "ci_low": block["ci"][0], "ci_high": block["ci"][1],
            "p_value": block["p_value"],
            "right": block["mcnemar"]["new_right_old_wrong"],
            "wrong": block["mcnemar"]["new_wrong_old_right"],
            "sig": block["significant_95"],
        }
    return {"variants": variants, "sig": sig}


def _mk(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "macro_f1": block["macro_f1"],
        "macro_f1_argmax": block.get("macro_f1_argmax", float("nan")),
        "accuracy": block["accuracy"],
        "HATE_f1": block["HATE_f1"],
        "OFFENSIVE_f1": block["OFFENSIVE_f1"],
        "CLEAN_f1": block["CLEAN_f1"],
    }


ORDER = ["current", "current_plus_word", "word_only", "embedding only", "current + embedding"]


def plot_macro_f1(data: Dict[str, Any], path: Path) -> None:
    v = data["variants"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})

    xs = np.arange(len(ORDER))
    vals = [v[k]["macro_f1"] for k in ORDER]
    colors = [COLORS[GROUPS[k]] for k in ORDER]
    bars = ax1.bar(xs, vals, 0.62, color=colors)
    for bar, val in zip(bars, vals):
        ax1.annotate(f"{val:.4f}", (bar.get_x() + bar.get_width() / 2, val),
                     ha="center", va="bottom", fontsize=8)
    base = v["current"]["macro_f1"]
    ax1.axhline(base, color=COLOR_CURRENT, linestyle="--", linewidth=1.1)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([LABELS[k] for k in ORDER], fontsize=8)
    ax1.set_ylim(0.42, 0.48)
    ax1.set_ylabel("macro-F1 (validation)")
    ax1.set_title("Macro-F1 tren VALIDATION - moi bien the lexicon")
    ax1.grid(axis="y", alpha=0.3)

    argmax = [v[k]["macro_f1_argmax"] for k in ORDER]
    bars2 = ax2.bar(xs, argmax, 0.62, color=colors, alpha=0.85)
    for bar, val in zip(bars2, argmax):
        ax2.annotate(f"{val:.4f}", (bar.get_x() + bar.get_width() / 2, val),
                     ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([LABELS[k] for k in ORDER], fontsize=8)
    ax2.set_ylim(0.0, 0.5)
    ax2.set_ylabel("macro-F1 argmax (validation)")
    ax2.set_title("Khi KHONG tune threshold (argmax) - chi de tham khao")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("So sanh cac cach mo rong lexicon cho rule engine (validation, n=2672) - "
                 "chenh lech deu nho va khong co y nghia")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(data: Dict[str, Any], path: Path) -> None:
    v = data["variants"]
    classes = ["HATE_f1", "OFFENSIVE_f1", "CLEAN_f1"]
    names = ["HATE", "OFFENSIVE", "CLEAN"]
    width = 0.16
    xs = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for i, k in enumerate(ORDER):
        offs = (i - (len(ORDER) - 1) / 2) * width
        vals = [v[k][c] for c in classes]
        bars = ax.bar(xs + offs, vals, width, label=LABELS[k].replace("\n", " "),
                      color=COLORS[GROUPS[k]], alpha=0.6 + 0.4 * (i % 2))
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("F1 tung lop tren validation - moi bien the lexicon "
                 "(chu y: OFFENSIVE khong bao gio duoc cai thien)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_ci(data: Dict[str, Any], path: Path) -> None:
    sig = data["sig"]
    keys = ["current_plus_word", "word_only", "embedding only", "current_plus_embedding"]
    display = {
        "current_plus_word": "+word terms vs current",
        "word_only": "word-only vs current",
        "embedding only": "embedding only vs current",
        "current_plus_embedding": "+embedding vs current",
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ys = np.arange(len(keys))
    for i, k in enumerate(keys):
        block = sig[k]
        d, lo, hi = block["delta"], block["ci_low"], block["ci_high"]
        sig_flag = block["sig"]
        color = COLOR_POS if sig_flag else COLOR_NEG
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.4)
        ax.plot([d], [i], "o", color=color, markersize=8)
        ax.annotate(f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] p={block['p_value']:.3f}",
                    (hi, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([display[k] for k in keys], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ macro-F1 (bien the - current), CI 95% (10000 bootstrap)")
    ax.set_title("CA HAI huong mo rong deu KHONG co y nghia: moi CI deu chua 0")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mcnemar(data: Dict[str, Any], path: Path) -> None:
    sig = data["sig"]
    keys = ["current_plus_word", "word_only", "embedding only", "current_plus_embedding"]
    display = {
        "current_plus_word": "+word terms",
        "word_only": "word-only",
        "embedding only": "embedding only",
        "current_plus_embedding": "+embedding",
    }
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = np.arange(len(keys))
    width = 0.36
    right = [sig[k]["right"] for k in keys]
    wrong = [sig[k]["wrong"] for k in keys]
    b1 = ax.bar(xs - width / 2, right, width, label="bien the dung, current sai", color=COLOR_POS)
    b2 = ax.bar(xs + width / 2, wrong, width, label="bien the sai, current dung", color=COLOR_NEG)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{int(bar.get_height())}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([display[k] for k in keys], fontsize=9)
    ax.set_ylabel("so mau doi ket qua")
    ax.set_title("McNemar: cac bien the doi RAT NHIEU quyet dinh nhung huy nhau\n"
                 "(vi du +embedding: +55 mau rong, phan lon la CLEAN, trong khi OFFENSIVE tut)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ve hinh so sanh cac bien the lexicon.")
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    args = parser.parse_args(argv)
    fig_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_all()
    LOGGER.info("[39] %d bien the, %d kiem dinh", len(data["variants"]), len(data["sig"]))

    plot_macro_f1(data, fig_dir / "lexicon_macro_f1.png")
    plot_per_class(data, fig_dir / "lexicon_per_class_f1.png")
    plot_delta_ci(data, fig_dir / "lexicon_delta_ci.png")
    plot_mcnemar(data, fig_dir / "lexicon_mcnemar.png")

    # bang tom tat dang markdown
    v = data["variants"]
    lines = ["# Tom tat so sanh cac bien the lexicon (validation)\n",
             "> Tat ca so do tren validation, tune threshold rieng tung lexicon (plan §44). "
             "Khong cham test.\n",
             "| Lexicon | Entries | Macro-F1 | Δ vs current | CI 95% | p | Mac-Nemar +/− | Ket luan |",
             "|---|---|---|---|---|---|---|---|"]
    entries = {"current": 668, "current_plus_word": 929, "word_only": 261,
               "embedding only": 1207, "current + embedding": 1272}
    base = v["current"]["macro_f1"]
    sig = data["sig"]
    lines.append(f"| current | {entries['current']} | {base:.4f} | - | - | - | - | moc |")
    pairs = [("current_plus_word", "+word terms"), ("word_only", "word-only"),
             ("embedding only", "embedding only"), ("current + embedding", "+embedding")]
    for key, name in pairs:
        d = v[key]["macro_f1"]
        block = sig.get("embedding only" if key == "embedding only" else
                        "current_plus_embedding" if key == "current + embedding" else key)
        if block:
            verdict = "CO y nghia" if block["sig"] else "khong (CI chua 0)"
            lines.append(f"| {name} | {entries[key]} | {d:.4f} | {d - base:+.4f} | "
                         f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}] | "
                         f"{block['p_value']:.3f} | {block['right']}/{block['wrong']} | {verdict} |")
    lines.append("\n**Ket luan:** khong cach mo rong lexicon nao cai thien macro-F1 mot cach "
                 "co y nghia thong ke. Xem 4 hinh trong `outputs/lexicon_experiments/figures/`.\n")
    (Path(args.out_dir).parent / "lexicon_variants_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("[39] wrote 4 figures + summary to %s", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
