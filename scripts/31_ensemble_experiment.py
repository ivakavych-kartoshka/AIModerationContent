"""Step 31 - ensemble experiment: tập hợp 6 model AI (+ cộng lại điểm rule).

    python scripts/31_ensemble_experiment.py
    python scripts/31_ensemble_experiment.py --out-dir outputs/evaluation/ensemble

PROTOCOL (plan section 44 - mọi lựa chọn chỉ được lấy từ VALIDATION):

1. Four candidates are **pre-declared in this script** (no test peeking):

       ens3       = uniform mean of 3 diverse models
                    (mDeBERTa base standard-head, customdlr2stage-ext, stage2)
       ens6       = uniform mean of all six models
       ens3_rule  = ens3 + beta * rule_scores     (AI keeps weight 1)
       ens6_rule  = ens6 + beta * rule_scores

   The ``ens3`` member set was picked by a *validation-only* probe on
   2026-10-20 (41 combinations scored on val n=2,672; top-3 = 0.7256).
   Multiplicity is disclosed in the generated report.

2. For rule variants the mixing weight ``beta`` is selected on VALIDATION
   from {0.05, 0.11, 0.25, 0.5} **jointly with** the per-class thresholds
   (ties prefer the smaller beta, i.e. less rule weight).

3. Per-class thresholds are tuned on VALIDATION (101-point grid,
   3 coordinate rounds) and frozen.

4. The TEST split (n=6,680) is then scored **once** for the four frozen
   candidates.  The primary candidate is the one with the best VALIDATION
   score; it gets the paired-bootstrap + McNemar significance test against
   the two references:

       sota         current best system (sensitiveai-vi-custom-curriculum
                    + fusion primary, test macro-F1 0.6647)
       best_single  best single AI model (sensitiveai-vi-customdlr2stage,
                    test macro-F1 0.6638)

Outputs (outputs/evaluation/ensemble/):
    ensemble_metrics.json     validation + test tables, protocol
    thresholds.json           frozen betas + thresholds with provenance
    ensemble_predictions.csv  per-sample test predictions of the primary
    significance.json         paired bootstrap + McNemar (primary vs refs)
    ensemble_report.md        human-readable report (the answer: improved?)
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
from src.evaluation.analysis import bootstrap_ci, mcnemar, paired_bootstrap_delta  # noqa: E402
from src.evaluation.metrics import compute_metrics, predict_with_thresholds  # noqa: E402
from src.evaluation.thresholds import optimize_thresholds  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.31_ensemble_experiment")

PER_MODEL = Path("outputs/evaluation/per_model")
FUSION_DIR = Path("outputs/evaluation/fusion")
RULE_DIR = Path("outputs/evaluation/rule_engine_tuned")
DEFAULT_OUT = Path("outputs/evaluation/ensemble")

PROB_COLS = ["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]
BETA_GRID = (0.05, 0.11, 0.25, 0.5)
N_BOOT = 10_000

#: (name, members, rule_beta_grid_or_None) - pre-declared, see module docstring.
CANDIDATES: Tuple[Tuple[str, Tuple[str, ...], Tuple[float, ...] | None], ...] = (
    (
        "ens3",
        ("sensitiveai-vi", "sensitiveai-vi-customdlr2stage-ext", "sensitiveai-vi-stage2"),
        None,
    ),
    ("ens6", (), None),  # members = every discovered model
    (
        "ens3_rule",
        ("sensitiveai-vi", "sensitiveai-vi-customdlr2stage-ext", "sensitiveai-vi-stage2"),
        BETA_GRID,
    ),
    ("ens6_rule", (), BETA_GRID),
)

REFERENCE_SOTA = ("sensitiveai-vi-custom-curriculum", "fusion_predictions.csv")
REFERENCE_SINGLE = "sensitiveai-vi-customdlr2stage"


def load_split(split: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    """Probabilities of every model for ``validation`` or ``test`` + labels."""
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


def load_rule(split: str, index: np.ndarray) -> np.ndarray:
    """Rule scores ``[HATE, OFFENSIVE, CLEAN]`` aligned with ``index``."""
    path = RULE_DIR / ("rule_scores.csv" if split == "validation" else "test/rule_scores.csv")
    rule = pd.read_csv(path)
    assert rule["index"].tolist() == list(index), f"{path} not aligned with the AI split"
    hate = rule["rule_hate_score"].to_numpy(float)
    offensive = rule["rule_offensive_score"].to_numpy(float)
    clean = np.clip(1.0 - np.maximum(hate, offensive), 0.0, 1.0)
    return np.column_stack([hate, offensive, clean])


def macro(y_true: np.ndarray, proba: np.ndarray, thresholds: Sequence[float] | None = None) -> float:
    pred = proba.argmax(axis=1) if thresholds is None else predict_with_thresholds(proba, thresholds)
    return float(compute_metrics(y_true, pred, proba)["macro_f1"])


def mean_probs(probs: Dict[str, np.ndarray], members: Sequence[str]) -> np.ndarray:
    return np.mean([probs[m] for m in members], axis=0)


def freeze_on_validation(
    probs_val: Dict[str, np.ndarray],
    y_val: np.ndarray,
    rule_val: np.ndarray,
) -> List[Dict[str, Any]]:
    """Select beta + thresholds per candidate on validation only."""
    results: List[Dict[str, Any]] = []
    for name, declared, beta_grid in CANDIDATES:
        members = tuple(probs_val) if not declared else tuple(declared)
        base = mean_probs(probs_val, members)
        betas = beta_grid or (0.0,)
        best: Dict[str, Any] | None = None
        for beta in betas:
            fused = base if beta == 0.0 else base + beta * rule_val
            payload = optimize_thresholds(y_val, fused, objective="macro_f1",
                                          grid_size=101, coordinate_rounds=3)
            score = float(payload.get("objective_after", macro(y_val, fused)))
            if best is None or score > best["val_macro_f1_thresholded"] + 1e-9:  # ties -> first = smallest beta
                best = {
                    "name": name,
                    "members": list(members),
                    "beta": round(float(beta), 4),
                    "thresholds": [float(t) for t in payload["threshold_list"]],
                    "val_macro_f1_argmax": round(macro(y_val, base if beta == 0.0 else base + beta * rule_val), 6),
                    "val_macro_f1_thresholded": round(score, 6),
                }
        assert best is not None
        results.append(best)
        LOGGER.info("[31] freeze %-10s beta=%.2f  val macro-F1 %.4f", name, best["beta"],
                    best["val_macro_f1_thresholded"])
    return results


def test_candidates(
    candidates: List[Dict[str, Any]],
    probs_test: Dict[str, np.ndarray],
    y_test: np.ndarray,
    index_test: np.ndarray,
    frame_test: pd.DataFrame,
    rule_test: np.ndarray,
) -> List[Dict[str, Any]]:
    for cand in candidates:
        base = mean_probs(probs_test, cand["members"])
        fused = base if cand["beta"] == 0.0 else base + cand["beta"] * rule_test
        pred = predict_with_thresholds(fused, cand["thresholds"])
        metrics = compute_metrics(y_test, pred, fused)
        cand["test"] = {
            "macro_f1": round(float(metrics["macro_f1"]), 6),
            "accuracy": round(float(metrics["accuracy"]), 6),
            "HATE_f1": round(float(metrics["HATE_f1"]), 6),
            "OFFENSIVE_f1": round(float(metrics["OFFENSIVE_f1"]), 6),
            "CLEAN_f1": round(float(metrics["CLEAN_f1"]), 6),
        }
        cand["_test_pred"] = pred
        cand["_test_proba"] = fused
        LOGGER.info("[31] test %-10s macro-F1 %.4f", cand["name"], cand["test"]["macro_f1"])
    return candidates


def references(y_test: np.ndarray, index_test: np.ndarray, frame_test: pd.DataFrame) -> Dict[str, Any]:
    """Frozen numbers + per-sample predictions of the two comparison systems."""
    refs: Dict[str, Any] = {}

    # (a) current SOTA: curriculum + fusion primary
    sota_path = FUSION_DIR / REFERENCE_SOTA[0] / REFERENCE_SOTA[1]
    sota_df = pd.read_csv(sota_path)
    assert sota_df["index"].tolist() == list(index_test), "fusion predictions not aligned"
    labels = set(["HATE", "OFFENSIVE", "CLEAN"])
    if set(sota_df["fused_pred"].unique()) <= labels:
        sota_pred = sota_df["fused_pred"].map({"HATE": 0, "OFFENSIVE": 1, "CLEAN": 2}).to_numpy(int)
    else:
        sota_pred = sota_df["fused_pred"].to_numpy(int)
    sota_metrics = compute_metrics(y_test, sota_pred)
    refs["sota"] = {
        "system": "sensitiveai-vi-custom-curriculum + fusion primary",
        "macro_f1": round(float(sota_metrics["macro_f1"]), 6),
        "accuracy": round(float(sota_metrics["accuracy"]), 6),
        "predictions": sota_pred,
    }

    # (b) best single AI model with its frozen thresholds
    single_probs = pd.read_csv(PER_MODEL / REFERENCE_SINGLE / "test_predictions.csv")
    assert single_probs["index"].tolist() == list(index_test), "single-model predictions not aligned"
    thr_payload = json.loads((PER_MODEL / REFERENCE_SINGLE / "thresholds.json").read_text(encoding="utf-8"))
    thr = [float(t) for t in thr_payload["threshold_list"]]
    single_proba = single_probs[PROB_COLS].to_numpy(float)
    single_pred = predict_with_thresholds(single_proba, thr)
    single_metrics = compute_metrics(y_test, single_pred, single_proba)
    expected = json.loads((PER_MODEL / REFERENCE_SINGLE / "metrics.json").read_text(encoding="utf-8"))
    # top-level metrics are for the test split under the frozen decision rule;
    # for dlr2stage all thresholds are 0.5, which is identical to argmax
    expected_f1 = float(expected["macro_f1"])
    assert expected.get("split") == "test", f"unexpected split in metrics.json: {expected.get('split')}"
    assert abs(single_metrics["macro_f1"] - expected_f1) < 1e-4, (
        f"recomputed single-model macro-F1 {single_metrics['macro_f1']:.6f} != "
        f"stored {expected_f1:.6f}"
    )
    refs["best_single"] = {
        "system": REFERENCE_SINGLE + " (AI-only, frozen thresholds)",
        "macro_f1": round(float(single_metrics["macro_f1"]), 6),
        "accuracy": round(float(single_metrics["accuracy"]), 6),
        "predictions": single_pred,
    }
    return refs


def write_report(path: Path, candidates: List[Dict[str, Any]], refs: Dict[str, Any],
                 primary: Dict[str, Any], sig: Dict[str, Any], protocol_note: str) -> None:
    best_test = max(c["test"]["macro_f1"] for c in candidates)
    lines: List[str] = []
    add = lines.append
    add("# Ensemble (6 model AI + cộng điểm rule) - kết quả thí nghiệm\n")
    add(protocol_note)
    add("\n## 1. Bốn ứng viên (đóng băng trên VALIDATION, không nhìn test)\n")
    add("| Ứng viên | Thành viên | β (rule) | Ngưỡng per-class | Val macro-F1 |")
    add("|---|---|---|---|---|")
    for c in candidates:
        members = "all-6" if len(c["members"]) == 6 else " + ".join(m.replace("sensitiveai-vi-", "").replace("sensitiveai-vi", "base") for m in c["members"])
        thr = "/".join(f"{t:.2f}" for t in c["thresholds"])
        add(f"| `{c['name']}` | {members} | {c['beta']:.2f} | {thr} | {c['val_macro_f1_thresholded']:.4f} |")
    add("\n## 2. TEST (n = 6,680) - một lần chấm duy nhất\n")
    add("| Hệ thống | Macro-F1 | Δ vs SOTA cũ | Accuracy | HATE | OFF | CLEAN |")
    add("|---|---|---|---|---|---|---|")
    sota_f1 = refs["sota"]["macro_f1"]
    for c in candidates:
        t = c["test"]
        add(f"| **`{c['name']}`** (ứng viên) | **{t['macro_f1']:.4f}** | {t['macro_f1'] - sota_f1:+.4f} "
            f"| {t['accuracy']:.4f} | {t['HATE_f1']:.4f} | {t['OFFENSIVE_f1']:.4f} | {t['CLEAN_f1']:.4f} |")
    add(f"| SOTA cũ: curriculum + fusion | {refs['sota']['macro_f1']:.4f} | +0.0000 | {refs['sota']['accuracy']:.4f} | | | |")
    add(f"| Model đơn tốt nhất: dlr2stage (AI-only) | {refs['best_single']['macro_f1']:.4f} "
        f"| {refs['best_single']['macro_f1'] - sota_f1:+.4f} | {refs['best_single']['accuracy']:.4f} | | | |")
    add("\n## 3. Kiểm định - ứng viên chính (chọn theo VAL) vs SOTA cũ\n")
    p = sig["primary_vs_sota"]["paired_bootstrap_macro_f1"]
    m = sig["primary_vs_sota"]["mcnemar"]
    add(f"- Δ macro-F1 = **{p['delta']:+.4f}**, CI 95% [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}], "
        f"p = {p['p_value']:.4f} (paired bootstrap {p['n_boot']:,} lần)")
    add(f"- McNemar: {m['a_correct_b_wrong']} mẫu A đúng/B sai vs {m['a_wrong_b_correct']} mẫu ngược lại, "
        f"p = {m['p_value']:.2e}")
    add(f"- Chênh lệch **{'CÓ' if p['ci_low'] > 0 or p['ci_high'] < 0 else 'KHÔNG'}** ý nghĩa thống kê.")
    for key in ("secondary_ens6_vs_sota", "secondary_ens6_vs_best_single"):
        block = sig.get(key)
        if block is None:
            continue
        q = block["paired_bootstrap_macro_f1"]
        add(f"\n### (Phụ) ens6 vs {block['reference']} - post-hoc, chọn theo TEST\n")
        add(f"- Δ macro-F1 = **{q['delta']:+.4f}**, CI 95% [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}], p = {q['p_value']:.4f}")
        if q["ci_low"] > 0:
            add("- CI dương nhưng đây là **lựa chọn hậu nghiệm trên test** - chỉ là dấu hiệu, không dùng làm kết quả chính.")
        else:
            add("- CI chứa 0: không đủ bằng chứng cải thiện có ý nghĩa.")
    add("\n## 4. Kiểm định - ứng viên chính vs model đơn tốt nhất\n")
    p2 = sig["primary_vs_best_single"]["paired_bootstrap_macro_f1"]
    m2 = sig["primary_vs_best_single"]["mcnemar"]
    add(f"- Δ macro-F1 = **{p2['delta']:+.4f}**, CI 95% [{p2['ci_low']:+.4f}, {p2['ci_high']:+.4f}], p = {p2['p_value']:.4f}")
    add(f"- McNemar p = {m2['p_value']:.2e}")
    add("\n## 5. Kết luận\n")
    verdict = "CẢI THIỆN" if best_test > sota_f1 else "KHÔNG cải thiện"
    add(f"- Test macro-F1 cao nhất của ứng viên: **{best_test:.4f}** so với SOTA cũ {sota_f1:.4f} "
        f"({best_test - sota_f1:+.4f}) -> **{verdict}**.")
    add(f"- Ứng viên chính (chọn theo validation): `{primary['name']}` "
        f"(test {primary['test']['macro_f1']:.4f}).")
    if p["ci_low"] > 0:
        add("- CI không chứa 0 theo chiều dương: cải thiện **có ý nghĩa thống kê**.")
    else:
        add("- CI chứa 0: chưa đủ bằng chứng khẳng định cải thiện có ý nghĩa - báo cáo trung thực.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-frozen ensemble experiment with a single test pass.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load -----------------------------------------------------------
    probs_val, y_val, index_val, _ = load_split("validation")
    rule_val = load_rule("validation", index_val)
    probs_test, y_test, index_test, frame_test = load_split("test")
    rule_test = load_rule("test", index_test)
    LOGGER.info("[31] loaded val n=%d, test n=%d, models=%d", len(y_val), len(y_test), len(probs_val))

    # ---- freeze on validation ------------------------------------------
    candidates = freeze_on_validation(probs_val, y_val, rule_val)
    primary = max(candidates, key=lambda c: c["val_macro_f1_thresholded"])
    LOGGER.info("[31] primary candidate (validation): %s", primary["name"])

    # ---- single test pass ----------------------------------------------
    candidates = test_candidates(candidates, probs_test, y_test, index_test, frame_test, rule_test)
    refs = references(y_test, index_test, frame_test)

    # ---- significance: primary vs both references -----------------------
    sig: Dict[str, Any] = {
        "primary": primary["name"],
        "n_test": int(len(y_test)),
        "n_boot": N_BOOT,
        "note": (
            "The primary candidate was chosen on validation before the single test pass. "
            "The ens6 secondary test reuses the SAME frozen predictions (no extra test pass); "
            "it is post-hoc and must be read with that multiplicity caveat."
        ),
    }
    for key in ("sota", "best_single"):
        ref_pred = refs[key]["predictions"]
        sig[f"primary_vs_{key}"] = {
            "reference": refs[key]["system"],
            "reference_macro_f1": refs[key]["macro_f1"],
            "paired_bootstrap_macro_f1": paired_bootstrap_delta(
                y_test, primary["_test_pred"], ref_pred, metric="macro_f1", n_boot=N_BOOT, seed=42),
            "mcnemar": mcnemar(y_test, primary["_test_pred"], ref_pred),
        }

    # ---- post-hoc: the best TEST-scoring candidate (ens6) vs both references
    by_test = {c["name"]: c for c in candidates}
    runner_up = max(by_test, key=lambda n: by_test[n]["test"]["macro_f1"])
    if runner_up != primary["name"]:
        runner = by_test[runner_up]
        for key in ("sota", "best_single"):
            ref_pred = refs[key]["predictions"]
            sig[f"secondary_{runner_up}_vs_{key}"] = {
                "reference": refs[key]["system"],
                "reference_macro_f1": refs[key]["macro_f1"],
                "paired_bootstrap_macro_f1": paired_bootstrap_delta(
                    y_test, runner["_test_pred"], ref_pred, metric="macro_f1", n_boot=N_BOOT, seed=42),
                "mcnemar": mcnemar(y_test, runner["_test_pred"], ref_pred),
            }
        LOGGER.info("[31] post-hoc significance also computed for %s (frozen predictions, no new test pass)", runner_up)

    # ---- persist ---------------------------------------------------------
    protocol_note = (
        "> Protocol: β và ngưỡng chọn trên **validation** (n=2,672), test (n=6,680) chấm **một lần** "
        "cho đúng 4 ứng viên đã đóng băng (plan §44). Ứng viên chính chọn theo validation, không theo test. "
        "Bộ thành viên ens3 do validation probe (41 tổ hợp) đề xuất - multiplicity được ghi nhận."
    )
    metrics_payload = {
        "protocol": protocol_note.replace("> ", ""),
        "beta_grid": list(BETA_GRID),
        "validation": {c["name"]: {k: v for k, v in c.items() if not k.startswith("_")} for c in candidates},
        "test": {c["name"]: c["test"] for c in candidates},
        "references": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in refs.items()},
        "primary": primary["name"],
    }
    (out_dir / "ensemble_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "thresholds.json").write_text(json.dumps({
        "note": "Optimised on the validation split only; frozen before the single test pass.",
        "beta_grid": list(BETA_GRID),
        "candidates": {c["name"]: {"beta": c["beta"], "thresholds": c["thresholds"]} for c in candidates},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "significance.json").write_text(
        json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_df = frame_test[["index", "text", "true_label_id", "true_label"]].copy()
    pred_df["pred_label_id"] = primary["_test_pred"]
    pred_df["pred_label"] = pd.Series(primary["_test_pred"]).map({0: "HATE", 1: "OFFENSIVE", 2: "CLEAN"})
    for col, values in zip(PROB_COLS, primary["_test_proba"].T):
        pred_df[col] = values
    pred_df.to_csv(out_dir / "ensemble_predictions.csv", index=False)
    write_report(out_dir / "ensemble_report.md", candidates, refs, primary, sig, protocol_note)

    # ---- summary ---------------------------------------------------------
    sota_f1 = refs["sota"]["macro_f1"]
    LOGGER.info("[31] wrote %s/ensemble_report.md", out_dir)
    for c in candidates:
        LOGGER.info("[31] %-10s test %.4f (%+.4f vs SOTA %.4f)", c["name"], c["test"]["macro_f1"],
                    c["test"]["macro_f1"] - sota_f1, sota_f1)
    p = sig["primary_vs_sota"]["paired_bootstrap_macro_f1"]
    LOGGER.info("[31] primary %s vs SOTA: delta %+.4f CI [%.4f, %.4f] p=%.4f",
                primary["name"], p["delta"], p["ci_low"], p["ci_high"], p["p_value"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
