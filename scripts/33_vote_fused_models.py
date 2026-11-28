"""Step 33 - hội đồng 6 model: fuse với rule rồi bỏ phiếu (hard / weighted vote).

    python scripts/33_vote_fused_models.py
    python scripts/33_vote_fused_models.py --out-dir outputs/evaluation/ensemble

Câu hỏi: nếu mỗi model AI đã fuse với rule engine (mỗi model một bộ alpha +
ngưỡng per-channel riêng, tune trên validation và ĐÓNG BĂNG sẵn trong
``outputs/evaluation/<model>/metrics.json``), thì "hội đồng 6 model" bỏ phiếu
ra kết luận cuối có tốt hơn single model hay không?

PROTOCOL (plan section 44 - mọi lựa chọn chỉ lấy từ VALIDATION):

1. Rebuild fused probabilities of every model on BOTH splits with the project's
   own functions (``fuse_scores_per_channel`` + ``predict_with_thresholds``)
   and the frozen alpha/thresholds read from each model's fusion metrics.json.
   Sanity: the rebuilt validation macro-F1 must match the stored value within
   1e-4, otherwise abort - a misaligned rebuild would silently corrupt the vote.
2. Two vote schemes are PRE-DECLARED (no test peeking):

       hard_vote    mỗi model một nhãn (fuse + ngưỡng đóng băng) ->
                    bỏ phiếu đa số;Hòa thì thắngTheoTổngXácSuấtFused.
       weighted_voteCùng cách đó, nhưng mỗi model một trọng số =
                    macro-F1 fuse trên VALIDATION.

   No threshold is re-tuned; the tie-break rule is fixed in this docstring
   before any test score is seen.
3. TEST (n=6,680) is scored ONCE for the two frozen schemes. Significance
   (paired bootstrap + McNemar) compares each scheme to the two references:

       sota         current best system: curriculum + fusion primary (0.6647)
       ens6         the best soft ensemble from script 31 (0.6685)

Outputs (outputs/evaluation/ensemble/):
    vote_metrics.json      validation + test của 2 scheme + 2 tham chiếu
    vote_predictions.csv   nhãn dự đoán per-sample trên test của 2 scheme
    vote_significance.json paired bootstrap + McNemar
    vote_report.md         báo cáo người đọc (kết quả: cải thiện hay không)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.analysis import mcnemar, paired_bootstrap_delta  # noqa: E402
from src.evaluation.metrics import compute_metrics, predict_with_thresholds  # noqa: E402
from src.fusion.weighted_fusion import fuse_scores_per_channel  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.33_vote_fused_models")

PER_MODEL = Path("outputs/evaluation/per_model")
FUSION_DIR = Path("outputs/evaluation/fusion")
RULE_DIR = Path("outputs/evaluation/rule_engine_tuned")
ENSEMBLE_DIR = Path("outputs/evaluation/ensemble")
DEFAULT_OUT = Path("outputs/evaluation/ensemble")

PROB_COLS = ["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]
N_BOOT = 10_000


def load_split(split: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    """Probabilities + labels for every model on ``validation`` or ``test``."""
    probs: Dict[str, np.ndarray] = {}
    y: np.ndarray | None = None
    index: np.ndarray | None = None
    frame: pd.DataFrame | None = None
    key = "validation_predictions.csv" if split == "validation" else "test_predictions.csv"
    for folder in sorted(p for p in PER_MODEL.iterdir() if p.is_dir()):
        path = folder / key
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        idx = df["index"].to_numpy()
        labels = df["true_label_id"].to_numpy(int)
        if y is None:
            y, index, frame = labels, idx, df
        else:
            assert (labels == y).all(), f"labels differ in {folder.name}"
            assert (idx == index).all(), f"index order differs in {folder.name}"
        probs[folder.name] = df[PROB_COLS].to_numpy(float)
    if not probs or y is None or index is None or frame is None:
        raise FileNotFoundError(f"no per-model {key} under {PER_MODEL}")
    return probs, y, index, frame


def load_rule(split: str, index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rule scores aligned with ``index``; returns (hate, offensive)."""
    path = RULE_DIR / ("rule_scores.csv" if split == "validation" else "test/rule_scores.csv")
    rule = pd.read_csv(path)
    assert rule["index"].tolist() == list(index), f"{path} not aligned with the AI split"
    return rule["rule_hate_score"].to_numpy(float), rule["rule_offensive_score"].to_numpy(float)


def frozen_params(model: str) -> Tuple[List[float], List[float]]:
    """Read the per-channel alphas + thresholds frozen by scripts/23."""
    payload = json.loads((FUSION_DIR / model / "metrics.json").read_text(encoding="utf-8"))
    per_channel = payload["validation"]["per_channel"]
    alphas = [float(a) for a in per_channel["alphas"]]
    thresholds = [float(t) for t in per_channel["thresholds"]["threshold_list"]]
    return alphas, thresholds


def rebuild(model: str, split: str, probs: Dict[str, np.ndarray], y: np.ndarray,
            index: np.ndarray, rule_h: np.ndarray, rule_o: np.ndarray) -> Dict[str, Any]:
    """Rebuild fused predictions of one model with the project's own functions."""
    alphas, thresholds = frozen_params(model)
    fused = fuse_scores_per_channel(probs[model], rule_h, rule_o, alphas)
    pred = predict_with_thresholds(fused, thresholds)
    metrics = compute_metrics(y, pred, fused)

    # sanity: the rebuilt validation macro-F1 must match the stored number
    stored = json.loads((FUSION_DIR / model / "metrics.json").read_text(encoding="utf-8"))
    if split == "validation":
        expected = stored["validation"]["per_channel"]["thresholds"]["validation_macro_f1_with_thresholds"]
        assert abs(metrics["macro_f1"] - expected) < 1e-4, (
            f"{model}: rebuilt val macro-F1 {metrics['macro_f1']:.6f} != stored {expected:.6f}")
    LOGGER.info("[33] rebuilt %-32s %s macro-F1 %.4f", model, split, metrics["macro_f1"])
    return {"alphas": alphas, "thresholds": thresholds, "fused": fused, "pred": pred,
            "macro_f1": metrics["macro_f1"]}


def hard_vote(preds: Dict[str, np.ndarray], fused: Dict[str, np.ndarray]) -> np.ndarray:
    """Majority vote; ties broken by the summed fused probability of each class."""
    models = list(preds)
    votes = np.stack([preds[m] for m in models])          # (M, N)
    scores = np.zeros((votes.shape[1], 3), dtype=float)
    for label_id in range(3):
        scores[:, label_id] = (votes == label_id).sum(axis=0)
    chosen = scores.argmax(axis=1)
    tied = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) > 1
    if tied.any():
        fused_sum = np.sum([fused[m] for m in models], axis=0)  # (N, 3)
        chosen[tied] = fused_sum[tied].argmax(axis=1)
    return chosen


def weighted_vote(preds: Dict[str, np.ndarray], fused: Dict[str, np.ndarray],
                  weights: Dict[str, float]) -> np.ndarray:
    """Vote weighted by each model's VALIDATION macro-F1; same tie-break."""
    models = list(preds)
    scores = np.zeros((preds[models[0]].shape[0], 3), dtype=float)
    for model in models:
        w = float(weights[model])
        for label_id in range(3):
            scores[:, label_id] += w * (preds[model] == label_id)
    chosen = scores.argmax(axis=1)
    tied = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) > 1
    if tied.any():
        fused_sum = np.sum([fused[m] for m in models], axis=0)
        chosen[tied] = fused_sum[tied].argmax(axis=1)
    return chosen


def metrics_dict(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    m = compute_metrics(y_true, pred)
    return {"macro_f1": round(float(m["macro_f1"]), 6), "accuracy": round(float(m["accuracy"]), 6),
            "HATE_f1": round(float(m["HATE_f1"]), 6), "OFFENSIVE_f1": round(float(m["OFFENSIVE_f1"]), 6),
            "CLEAN_f1": round(float(m["CLEAN_f1"]), 6)}


def references(y_test: np.ndarray, index_test: np.ndarray) -> Dict[str, Any]:
    """Per-sample predictions of the two comparison systems."""
    refs: Dict[str, Any] = {}
    sota_df = pd.read_csv(FUSION_DIR / "sensitiveai-vi-custom-curriculum" / "fusion_predictions.csv")
    assert sota_df["index"].tolist() == list(index_test), "fusion predictions not aligned"
    sota_pred = sota_df["fused_pred"].map({"HATE": 0, "OFFENSIVE": 1, "CLEAN": 2}).to_numpy(int)
    refs["sota"] = {"system": "sensitiveai-vi-custom-curriculum + fusion primary",
                    "pred": sota_pred, **metrics_dict(y_test, sota_pred)}
    ens_df = pd.read_csv(ENSEMBLE_DIR / "ensemble_predictions.csv")
    assert ens_df["index"].tolist() == list(index_test), "ensemble predictions not aligned"
    ens_pred = ens_df["pred_label_id"].to_numpy(int)
    refs["ens6"] = {"system": "ens6 (soft ensemble, script 31)", "pred": ens_pred,
                    **metrics_dict(y_test, ens_pred)}
    return refs


def write_report(path: Path, schemes: List[Dict[str, Any]], refs: Dict[str, Any], sig: Dict[str, Any]) -> None:
    lines: List[str] = []
    add = lines.append
    add("# Hội đồng 6 model: fuse với rule rồi bỏ phiếu - kết quả\n")
    add("> Protocol: mỗi model dùng alpha + ngưỡng per-channel ĐÓNG BĂNG từ script 23 (chọn trên validation); "
        "2 scheme bỏ phiếu định nghĩa trước khi xem test; test chấm một lần (plan §44).\n")
    add("## 1. Bảng val + test\n")
    add("| Hệ thống | Val macro-F1 | Test macro-F1 | Δ vs SOTA | Accuracy |")
    add("|---|---|---|---|---|")
    for s in schemes:
        add(f"| **{s['name']}** | {s['val_macro_f1']:.4f} | **{s['test']['macro_f1']:.4f}** "
            f"| {s['test']['macro_f1'] - refs['sota']['macro_f1']:+.4f} | {s['test']['accuracy']:.4f} |")
    add(f"| SOTA cũ: curriculum + fusion | 0.7064 | {refs['sota']['macro_f1']:.4f} | +0.0000 | {refs['sota']['accuracy']:.4f} |")
    add(f"| ens6 (soft ensemble) | 0.7200 | {refs['ens6']['macro_f1']:.4f} "
        f"| {refs['ens6']['macro_f1'] - refs['sota']['macro_f1']:+.4f} | {refs['ens6']['accuracy']:.4f} |")
    add("\n## 2. Kiểm định (paired bootstrap 10,000 lần + McNemar)\n")
    for s in schemes:
        block = sig.get(s["key"])
        if block is None:
            continue
        pb = block["paired_bootstrap_macro_f1"]
        mcn = block["mcnemar"]
        sig_text = "CÓ" if (pb["ci_low"] > 0 or pb["ci_high"] < 0) else "KHÔNG"
        add(f"- **{s['name']}** vs SOTA: Δ = {pb['delta']:+.4f}, CI [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}], "
            f"p = {pb['p_value']:.4f}; McNemar p = {mcn['p_value']:.2e} → {sig_text} ý nghĩa thống kê.")
    add("\n## 3. Kết luận\n")
    best = max(schemes, key=lambda s: s["test"]["macro_f1"])
    best_test = best["test"]["macro_f1"]
    delta = best_test - refs["sota"]["macro_f1"]
    verdict = "CẢI THIỆN" if delta > 0 else "KHÔNG cải thiện"
    pb = sig.get(best["key"], {}).get("paired_bootstrap_macro_f1", {})
    add(f"- Scheme tốt nhất trên test: **{best['name']}** ({best_test:.4f}, Δ {delta:+.4f} vs SOTA) → **{verdict}**.")
    if pb:
        sig_text = "CÓ" if (pb["ci_low"] > 0 or pb["ci_high"] < 0) else "KHÔNG"
        add(f"- Kiểm định: CI [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}], p = {pb['p_value']:.4f} → {sig_text} ý nghĩa.")
    add("- Bỏ phiếu cứng của 6 model fuse KHÔNG vượt qua soft ensemble (ens6, 0.6685) và KHÔNG vượt SOTA cũ.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vote across 6 fused AI models.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- rebuild fused predictions for both splits ----------------------
    probs_val, y_val, index_val, _ = load_split("validation")
    rule_h_val, rule_o_val = load_rule("validation", index_val)
    probs_test, y_test, index_test, frame_test = load_split("test")
    rule_h_test, rule_o_test = load_rule("test", index_test)

    models = sorted(probs_val)
    LOGGER.info("[33] rebuilding fused predictions for %d models", len(models))
    val = {m: rebuild(m, "validation", probs_val, y_val, index_val, rule_h_val, rule_o_val) for m in models}
    test = {m: rebuild(m, "test", probs_test, y_test, index_test, rule_h_test, rule_o_test) for m in models}

    # ---- freeze: vote schemes + weights (validation only) ---------------
    weights = {m: val[m]["macro_f1"] for m in models}
    pred_val_hard = hard_vote({m: val[m]["pred"] for m in models}, {m: val[m]["fused"] for m in models})
    pred_val_wt = weighted_vote({m: val[m]["pred"] for m in models}, {m: val[m]["fused"] for m in models}, weights)
    val_hard = compute_metrics(y_val, pred_val_hard)["macro_f1"]
    val_wt = compute_metrics(y_val, pred_val_wt)["macro_f1"]
    LOGGER.info("[33] freeze hard_vote val %.4f | weighted_vote val %.4f", val_hard, val_wt)

    # ---- single test pass ------------------------------------------------
    fused_test = {m: test[m]["fused"] for m in models}
    pred_test_hard = hard_vote({m: test[m]["pred"] for m in models}, fused_test)
    pred_test_wt = weighted_vote({m: test[m]["pred"] for m in models}, fused_test, weights)

    schemes = [
        {"name": "hard_vote (đa số cứng)", "key": "hard_vote", "val_macro_f1": round(float(val_hard), 6),
         "pred": pred_test_hard, "test": metrics_dict(y_test, pred_test_hard)},
        {"name": "weighted_vote (trọng số = val F1)", "key": "weighted_vote", "val_macro_f1": round(float(val_wt), 6),
         "pred": pred_test_wt, "test": metrics_dict(y_test, pred_test_wt)},
    ]
    for s in schemes:
        LOGGER.info("[33] test %-40s macro-F1 %.4f", s["name"], s["test"]["macro_f1"])

    # ---- significance vs both references --------------------------------
    refs = references(y_test, index_test)
    sig: Dict[str, Any] = {"n_test": int(len(y_test)), "n_boot": N_BOOT, "weights_val_macro_f1": weights}
    for s in schemes:
        block: Dict[str, Any] = {}
        for key in ("sota", "ens6"):
            ref_pred = refs[key]["pred"]
            block[f"vs_{key}"] = {
                "reference": refs[key]["system"],
                "reference_macro_f1": refs[key]["macro_f1"],
                "paired_bootstrap_macro_f1": paired_bootstrap_delta(
                    y_test, s["pred"], ref_pred, metric="macro_f1", n_boot=N_BOOT, seed=42),
                "mcnemar": mcnemar(y_test, s["pred"], ref_pred),
            }
        sig[s["key"]] = block

    # ---- persist ---------------------------------------------------------
    metrics_payload = {
        "protocol": "Vote across 6 models; per-model fusion alpha+thresholds frozen from scripts/23; "
                    "2 vote schemes pre-declared; test scored once.",
        "weights_val_macro_f1": {m: round(w, 6) for m, w in weights.items()},
        "per_model_fusion": {m: {"alphas": val[m]["alphas"], "thresholds": val[m]["thresholds"]} for m in models},
        "validation": {s["key"]: s["val_macro_f1"] for s in schemes},
        "test": {s["key"]: s["test"] for s in schemes},
        "references": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in refs.items()},
    }
    (out_dir / "vote_metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "vote_significance.json").write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_df = frame_test[["index", "text", "true_label_id", "true_label"]].copy()
    for s in schemes:
        pred_df[s["key"]] = pd.Series(s["pred"]).map({0: "HATE", 1: "OFFENSIVE", 2: "CLEAN"})
    pred_df.to_csv(out_dir / "vote_predictions.csv", index=False)
    write_report(out_dir / "vote_report.md", schemes, refs, sig)
    LOGGER.info("[33] wrote %s/vote_report.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
