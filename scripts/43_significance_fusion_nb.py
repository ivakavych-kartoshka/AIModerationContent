"""Step 43 - kiem dinh: fusion voi lexicon NB co tot hon SOTA hien tai khong?

    python scripts/43_significance_fusion_nb.py

Script 42 cho thay lexicon Naive Bayes + multi-label lam rule engine manh len
ro ret tren validation (0.4555 -> 0.4800, p=0.037). Script nay do xem no co
day FUSION len khong, va chenh lech co y nghia khong.

So sanh (tat ca tren TEST, chi doc lai - KHONG cham lai model AI):
    A. SOTA cu  : outputs/evaluation/fusion/sensitiveai-vi-custom-curriculum
                  (rule lexicon 668 entry, frozen)
    B. fusion NB: outputs/evaluation/fusion_nb/sensitiveai-vi-custom-curriculum
                  (rule lexicon NB 1155 entry, cung trong so da tune)

Ca hai doc cung AI predictions cache (outputs/evaluation/per_model/...), nen
chenh lech CHI den tu lexicon rule. Kiem dinh paired bootstrap + McNemar tren
macro-F1, dung src/evaluation/analysis.

Outputs (outputs/lexicon_experiments/nb_multilabel/)
    fusion_nb_significance.md/json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.analysis import mcnemar, paired_bootstrap_delta  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.43_significance_fusion_nb")

MODEL = "sensitiveai-vi-custom-curriculum"
SOTA_DIR = Path("outputs/evaluation/fusion") / MODEL
NB_DIR = Path("outputs/evaluation/fusion_nb") / MODEL
OUT_DIR = Path("outputs/lexicon_experiments/nb_multilabel")


def load_preds(path: Path, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "index"})
    return df[["index", col]].copy()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiem dinh fusion NB vs SOTA.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sota_path = SOTA_DIR / "fusion_predictions.csv"
    nb_path = NB_DIR / "fusion_predictions.csv"
    for p in (sota_path, nb_path):
        if not p.is_file():
            LOGGER.error("[43] thieu %s", p)
            return 1

    sota = pd.read_csv(sota_path)
    nb = pd.read_csv(nb_path)
    col = "fused_pred"
    merged = sota[["index", "true_label", col]].merge(
        nb[["index", col]], on="index", suffixes=("_sota", "_nb"))
    LOGGER.info("[43] aligned n=%d", len(merged))

    from src.common.constants import LABEL2ID

    y = merged["true_label"].map(LABEL2ID).to_numpy(int)
    pred_sota = merged[f"{col}_sota"].map(LABEL2ID).to_numpy(int)
    pred_nb = merged[f"{col}_nb"].map(LABEL2ID).to_numpy(int)

    d = paired_bootstrap_delta(y, pred_nb, pred_sota, metric="macro_f1",
                               n_boot=args.n_boot, seed=42)
    mc = mcnemar(y, pred_nb, pred_sota)
    significant = (d["ci_low"] > 0.0) or (d["ci_high"] < 0.0)

    def macro_f1(pred: np.ndarray) -> float:
        return _macro_f1(y, pred)

    sota_f1 = _macro_f1(y, pred_sota)
    nb_f1 = _macro_f1(y, pred_nb)

    LOGGER.info("[43] SOTA=%.4f  fusion_nb=%.4f  delta=%+.4f CI[%+.4f,%+.4f] p=%.3f sig=%s",
                sota_f1, nb_f1, d["delta"], d["ci_low"], d["ci_high"], d["p_value"], significant)
    LOGGER.info("[43] McNemar: nb dung/sota sai=%d, nb sai/sota dung=%d, p=%.4f",
                mc["a_correct_b_wrong"], mc["a_wrong_b_correct"], mc["p_value"])

    # per-class
    per_class = {}
    for name, pred in (("SOTA", pred_sota), ("fusion_nb", pred_nb)):
        per_class[name] = _per_class_f1(y, pred)

    lines: List[str] = []
    add = lines.append
    add("# Fusion voi lexicon NB vs SOTA hien tai (test)\n")
    add(f"> n={len(y)} test, ca hai dung cung AI predictions cache; chenh lech chi den tu lexicon rule. "
        f"{args.n_boot} bootstrap.\n")
    add("## Macro-F1 (test)\n")
    add("| He thong | Macro-F1 | HATE F1 | OFF F1 | CLEAN F1 |")
    add("|---|---|---|---|---|")
    for name in ("SOTA", "fusion_nb"):
        pc = per_class[name]
        f1 = sota_f1 if name == "SOTA" else nb_f1
        add(f"| {name} | **{f1:.4f}** | {pc['HATE']:.4f} | {pc['OFFENSIVE']:.4f} | {pc['CLEAN']:.4f} |")
    add("\n## Kiem dinh (fusion_nb vs SOTA)\n")
    add(f"- Δ macro-F1 = **{d['delta']:+.4f}**")
    add(f"- CI 95% = [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    add(f"- p (bootstrap) = {d['p_value']:.3f}")
    add(f"- McNemar: {mc['a_correct_b_wrong']} dung / {mc['a_wrong_b_correct']} sai "
        f"(p = {mc['p_value']:.4f})")
    add(f"- **Ket luan: {'CO y nghia thong ke' if significant else 'KHONG co y nghia (CI chua 0)'}**")
    add("\n## Cach doc\n")
    add("- Δ duong = lexicon NB day fusion len. Day la so TEST, chi doc lai tu artifact da co.")
    add("- AI predictions khong doi, nen chenh lech hoan toan do lexicon rule.")
    (out_dir / "fusion_nb_significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "fusion_nb_significance.json").write_text(json.dumps({
        "n": int(len(y)), "sota_macro_f1": round(sota_f1, 6), "nb_macro_f1": round(nb_f1, 6),
        "delta": round(float(d["delta"]), 6),
        "ci": [round(float(d["ci_low"]), 6), round(float(d["ci_high"]), 6)],
        "p_value": round(float(d["p_value"]), 6),
        "mcnemar": {"nb_right_sota_wrong": mc["a_correct_b_wrong"],
                    "nb_wrong_sota_right": mc["a_wrong_b_correct"],
                    "p_value": round(float(mc["p_value"]), 6)},
        "significant_95": bool(significant),
        "per_class": per_class,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[43] wrote %s/fusion_nb_significance.md", out_dir)
    return 0


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    from src.evaluation.analysis import confusion_from_labels, _metric_from_confusion
    return float(_metric_from_confusion(confusion_from_labels(y, pred), "macro_f1"))


def _per_class_f1(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    out = {}
    for label_id, name in enumerate(("HATE", "OFFENSIVE", "CLEAN")):
        tp = int(np.sum((pred == label_id) & (y == label_id)))
        fp = int(np.sum((pred == label_id) & (y != label_id)))
        fn = int(np.sum((pred != label_id) & (y == label_id)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[name] = round(2 * prec * rec / (prec + rec), 6) if prec + rec else 0.0
    return out


if __name__ == "__main__":
    raise SystemExit(main())
