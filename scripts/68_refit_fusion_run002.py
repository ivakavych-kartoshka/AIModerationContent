"""Step 68 - Refit the VI hybrid fusion on the surviving curriculum checkpoint.

Why this script exists
----------------------
The historical fusion artefacts in ``outputs/evaluation/fusion/sensitiveai-vi-*``
were produced from per-model predictions whose encoder weights are no longer on
disk (only ``run_002`` survives).  Mixing those artefacts with live inference
produces contradictory numbers (different AI probabilities for the same test
split).  This script rebuilds the full VI decision-level analysis from one
checkpoint and one frozen configuration:

    checkpoint : sensitiveai-vi-custom-curriculum / run_002 / best_model
    rules      : configs/rule_engine_tuned.yaml (validation-tuned, frozen)
    protocol   : alphas + thresholds fitted on VALIDATION only, then frozen and
                 applied exactly once to TEST.

Outputs (new directory, historical artefacts untouched)
-------------------------------------------------------
    outputs/evaluation/fusion/sensitiveai-vi-custom-curriculum-run002/
        metrics.json
        fusion_predictions.csv
        numbers.json
        reliability_bins.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.constants import ID2LABEL, LABEL2ID, LABELS  # noqa: E402
from src.evaluation.analysis import (  # noqa: E402
    error_stratification,
    mcnemar,
    paired_bootstrap_delta,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.fusion.weighted_fusion import (  # noqa: E402
    MetaFusion,
    fuse_scores,
    fuse_scores_per_channel,
    rule_pseudo_proba,
    score_alpha_grid,
    score_alpha_grid_per_channel,
    tune_thresholds_at_alpha,
    tune_thresholds_at_alphas,
)
from src.inference.predictor import CheckpointPredictor  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUT_DIR = EVAL_DIR / "fusion" / "sensitiveai-vi-custom-curriculum-run002"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = PROJECT_ROOT / "outputs" / "reports" / "sensitiveai-vi-custom-curriculum" / "run_002" / "best_model"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "validation.csv"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
RULE_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "rule_engine_tuned"

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

LAB = {"HATE": 0, "OFFENSIVE": 1, "CLEAN": 2}


def load_split(csv_path: Path, rule_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    rules = pd.read_csv(rule_path)
    df = df.reset_index().rename(columns={"index": "row"})
    rules = rules.rename(columns={"index": "row"})
    merged = df.merge(rules[["row", "true_label", "rule_hate_score", "rule_offensive_score", "pred_label"]],
                      on="row", how="inner")
    if len(merged) != len(df):
        raise RuntimeError(f"{csv_path.name}: {len(df)} rows but merged {len(merged)}")
    y_csv = merged["label"] if pd.api.types.is_numeric_dtype(merged["label"]) else merged["label"].map(LAB)
    y_rule = merged["true_label"] if pd.api.types.is_numeric_dtype(merged["true_label"]) else merged["true_label"].map(LAB)
    if (y_csv.to_numpy(dtype=int) != y_rule.to_numpy(dtype=int)).any():
        raise RuntimeError(f"{csv_path.name}: label order disagrees with rule scores")
    return {
        "texts": merged["text"].astype(str).tolist(),
        "y": y_rule.to_numpy(dtype=int),
        "rule_hate": merged["rule_hate_score"].to_numpy(dtype=float),
        "rule_offensive": merged["rule_offensive_score"].to_numpy(dtype=float),
        "rule_pred": merged["pred_label"].map(LAB).to_numpy(dtype=int),
    }


def infer(predictor: CheckpointPredictor, texts: list[str]) -> np.ndarray:
    return np.asarray(predictor.predict_proba(texts), dtype=float)


def metric_block(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    m = compute_metrics(y, pred, proba)
    out = {"n": int(len(y)), "accuracy": round(float(m["accuracy"]), 4),
           "macro_f1": round(float(m["macro_f1"]), 4)}
    for label in LABELS:
        out[f"{label}_precision"] = round(float(m[f"{label}_precision"]), 4)
        out[f"{label}_recall"] = round(float(m[f"{label}_recall"]), 4)
        out[f"{label}_f1"] = round(float(m[f"{label}_f1"]), 4)
        out[f"{label}_support"] = int(m[f"{label}_support"])
    out["confusion_matrix"] = np.asarray(m["confusion_matrix"], dtype=int).tolist()
    return out


def decide(scores: np.ndarray, thresholds: dict[str, float] | list[float]) -> np.ndarray:
    if isinstance(thresholds, dict):
        tau = np.array([thresholds[label] for label in LABELS], dtype=float)
    else:
        tau = np.asarray(thresholds, dtype=float)
    above = scores >= tau[None, :]
    out = np.argmax(np.where(above, scores, -np.inf), axis=1)
    no_hit = ~above.any(axis=1)
    out[no_hit] = np.argmax(scores[no_hit], axis=1)
    return out.astype(int)


def calibration(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        count = int(mask.sum())
        acc = float((pred[mask] == y[mask]).mean()) if count else 0.0
        avg = float(conf[mask].mean()) if count else 0.0
        ece += count / len(y) * abs(acc - avg)
        bins.append({"low": float(edges[i]), "high": float(edges[i + 1]), "count": count,
                     "confidence": round(avg, 4), "accuracy": round(acc, 4)})
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    p_true = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    nll = float(-np.mean(np.log(p_true)))
    return {"ece": round(ece, 4), "brier": round(brier, 4), "nll": round(nll, 4),
            "mean_confidence": round(float(conf.mean()), 4), "bins": bins}


def compare(y: np.ndarray, a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    boot = paired_bootstrap_delta(y, a, b)
    mc = mcnemar(y, a, b)
    return {
        "a": name_a, "b": name_b,
        "delta_macro_f1": round(float(boot["delta"]), 4),
        "ci_low": round(float(boot["ci_low"]), 4),
        "ci_high": round(float(boot["ci_high"]), 4),
        "bootstrap_p": round(float(boot["p_value"]), 4),
        "a_right_b_wrong": int(mc["a_correct_b_wrong"]),
        "a_wrong_b_right": int(mc["a_wrong_b_correct"]),
        "mcnemar_p": round(float(mc["p_value"]), 4),
        "accuracy_a": round(float(np.mean(a == y)), 4),
        "accuracy_b": round(float(np.mean(b == y)), 4),
    }


def main() -> int:
    print("[1/6] Loading splits and rule scores...")
    val = load_split(VAL_CSV, RULE_DIR / "rule_scores.csv")
    test = load_split(TEST_CSV, RULE_DIR / "test" / "rule_scores.csv")
    print(f"    validation {len(val['y'])} / test {len(test['y'])}")

    print("[2/6] Running checkpoint inference (validation + test)...")
    predictor = CheckpointPredictor(str(CKPT), head="custom", max_length=256, batch_size=32)
    val["p_ai"] = infer(predictor, val["texts"])
    test["p_ai"] = infer(predictor, test["texts"])

    print("[3/6] Tuning on validation only...")
    global_alpha, global_trace = score_alpha_grid(
        val["y"], val["p_ai"], val["rule_hate"], val["rule_offensive"], ALPHA_GRID
    )
    global_thresholds = tune_thresholds_at_alpha(
        val["y"], val["p_ai"], val["rule_hate"], val["rule_offensive"], global_alpha
    )
    per_alphas, per_trace = score_alpha_grid_per_channel(
        val["y"], val["p_ai"], val["rule_hate"], val["rule_offensive"], ALPHA_GRID, rounds=3
    )
    per_thresholds = tune_thresholds_at_alphas(
        val["y"], val["p_ai"], val["rule_hate"], val["rule_offensive"], per_alphas
    )
    ai_thresholds = tune_thresholds_at_alpha(
        val["y"], val["p_ai"], val["rule_hate"], val["rule_offensive"], 1.0
    )
    meta = MetaFusion().fit(val["p_ai"], val["rule_hate"], val["rule_offensive"], val["y"])

    print(f"    global alpha={global_alpha} thresholds={global_thresholds['thresholds']}")
    print(f"    per-channel alphas={per_alphas} thresholds={per_thresholds['thresholds']}")
    print(f"    AI-only thresholds={ai_thresholds['thresholds']}")

    print("[4/6] Scoring the test split once...")
    p_ai = test["p_ai"]
    rh, ro = test["rule_hate"], test["rule_offensive"]
    y = test["y"]

    tau_ai = ai_thresholds["thresholds"]
    tau_g = global_thresholds["thresholds"]
    tau_p = per_thresholds["thresholds"]

    global_scores = fuse_scores(p_ai, rh, ro, float(global_alpha))
    per_scores = fuse_scores_per_channel(p_ai, rh, ro, per_alphas)
    rule_scores = rule_pseudo_proba(rh, ro)

    from src.evaluation.metrics import apply_decision_rule

    systems = {
        "ai_only_argmax": (apply_decision_rule(p_ai, "argmax"), p_ai),
        "ai_only_thresholds": (decide(p_ai, tau_ai), p_ai),
        "rule_only_argmax": (apply_decision_rule(rule_scores, "argmax"), rule_scores),
        "rule_only_thresholds": (test["rule_pred"], rule_scores),
        "fusion_global_argmax": (apply_decision_rule(global_scores, "argmax"), global_scores),
        "fusion_global_thresholds": (decide(global_scores, tau_g), global_scores),
        "fusion_per_channel_argmax": (apply_decision_rule(per_scores, "argmax"), per_scores),
        "fusion_per_channel_thresholds": (decide(per_scores, tau_p), per_scores),
        "meta_classifier": (apply_decision_rule(
            meta.predict_proba(p_ai, rh, ro), "argmax"), meta.predict_proba(p_ai, rh, ro)),
    }

    metrics = {name: metric_block(y, pred, proba) for name, (pred, proba) in systems.items()}
    primary = systems["fusion_per_channel_thresholds"][0]
    ai_argmax = systems["ai_only_argmax"][0]
    ai_thr = systems["ai_only_thresholds"][0]

    print("[5/6] Significance, calibration, provenance, strata...")
    significance = {
        "hybrid_vs_ai_argmax": compare(y, primary, ai_argmax, "hybrid", "ai_argmax"),
        "hybrid_vs_ai_thresholds": compare(y, primary, ai_thr, "hybrid", "ai_thresholds"),
    }

    calib = {
        "ai_only_argmax": calibration(p_ai, y),
        "hybrid_per_channel_thresholds": calibration(per_scores, y),
        "uniform_mix_global_0.70": calibration(0.70 * p_ai + 0.30 * (1.0 / 3.0), y),
        "uniform_mix_per_channel": calibration(
            np.asarray(per_alphas).reshape(1, 3) * p_ai
            + (1.0 - np.asarray(per_alphas).reshape(1, 3)) * (1.0 / 3.0), y),
    }

    # provenance of changed decisions vs AI-only argmax
    changed = np.where(primary != ai_argmax)[0]
    corrections = changed[ai_argmax[changed] != y[changed]]
    regressions = changed[ai_argmax[changed] == y[changed]]

    def rule_dir(i: int) -> str:
        if rh[i] == 0 and ro[i] == 0:
            return "silent"
        return "HATE" if rh[i] >= ro[i] else "OFFENSIVE"

    transitions = {}
    for i in changed:
        key = f"{ID2LABEL[int(ai_argmax[i])]}->{ID2LABEL[int(primary[i])]}"
        transitions[key] = transitions.get(key, 0) + 1

    provenance = {
        "changed": int(len(changed)),
        "corrections": int(len(corrections)),
        "regressions": int(len(regressions)),
        "transitions": transitions,
        "corrections_by_rule_direction": {
            d: int(sum(rule_dir(i) == d for i in corrections)) for d in ("HATE", "OFFENSIVE", "silent")
        },
        "regressions_by_rule_direction": {
            d: int(sum(rule_dir(i) == d for i in regressions)) for d in ("HATE", "OFFENSIVE", "silent")
        },
        "hate_recoveries": int(sum(
            ID2LABEL[int(primary[i])] == "HATE" and ID2LABEL[int(ai_argmax[i])] != "HATE" for i in corrections)),
        "hate_recoveries_with_rule_hate_direction": int(sum(
            ID2LABEL[int(primary[i])] == "HATE" and ID2LABEL[int(ai_argmax[i])] != "HATE"
            and rule_dir(i) == "HATE" for i in corrections)),
        "median_rule_hate_on_hate_recoveries": round(float(np.median([rh[i] for i in corrections
            if ID2LABEL[int(primary[i])] == "HATE" and ID2LABEL[int(ai_argmax[i])] != "HATE"])) , 4)
            if any(ID2LABEL[int(primary[i])] == "HATE" and ID2LABEL[int(ai_argmax[i])] != "HATE"
                   for i in corrections) else None,
        "clean_deflag": int(sum(
            ID2LABEL[int(primary[i])] == "CLEAN" and ID2LABEL[int(ai_argmax[i])] != "CLEAN" for i in corrections)),
        "clean_deflag_with_silent_rule": int(sum(
            ID2LABEL[int(primary[i])] == "CLEAN" and ID2LABEL[int(ai_argmax[i])] != "CLEAN" and rule_dir(i) == "silent"
            for i in corrections)),
    }

    # surface strata (overlapping tags)
    frame_ai = pd.DataFrame({"text": test["texts"], "true_label_id": y, "pred_label_id": ai_argmax})
    frame_hy = pd.DataFrame({"text": test["texts"], "true_label_id": y, "pred_label_id": primary})
    strata = {
        "ai_only_argmax": error_stratification(frame_ai),
        "hybrid": error_stratification(frame_hy),
    }

    rule_stats = {
        "silent_fraction": round(float(np.mean((rh == 0) & (ro == 0))), 4),
        "non_silent_fraction": round(float(np.mean((rh > 0) | (ro > 0))), 4),
        "hate_direction_fraction": round(float(np.mean((rh > 0) & (rh >= ro))), 4),
        "offensive_direction_fraction": round(float(np.mean((ro > 0) & (ro > rh))), 4),
    }

    numbers = {
        "model": "sensitiveai-vi-custom-curriculum",
        "checkpoint": str(CKPT.relative_to(PROJECT_ROOT)),
        "protocol": "alphas + thresholds selected on validation, frozen before the single test pass",
        "n_validation": int(len(val["y"])),
        "n_test": int(len(y)),
        "alpha_grid": ALPHA_GRID,
        "global_alpha": float(global_alpha),
        "global_thresholds": tau_g,
        "per_channel_alphas": per_alphas,
        "per_channel_thresholds": tau_p,
        "ai_only_thresholds": tau_ai,
        "validation": {
            "global": {"alpha": float(global_alpha), "trace": global_trace,
                       "thresholds": global_thresholds},
            "per_channel": {"alphas": per_alphas, "trace": per_trace, "thresholds": per_thresholds},
            "ai_only_thresholds": ai_thresholds,
        },
        "test": metrics,
        "significance": significance,
        "calibration": calib,
        "provenance": provenance,
        "rule_stats": rule_stats,
        "surface_strata": strata,
        "meta_coefficients": meta.coefficients(),
    }
    (OUT_DIR / "numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(json.dumps(
        {"model": numbers["model"], "checkpoint": numbers["checkpoint"],
         "per_channel_alphas": per_alphas, "per_channel_thresholds": tau_p,
         "global_alpha": float(global_alpha), "global_thresholds": tau_g,
         "ai_only_thresholds": tau_ai, "test": metrics, "significance": significance},
        indent=2), encoding="utf-8")

    frame = pd.DataFrame({
        "index": np.arange(len(y)),
        "text": test["texts"],
        "true_label": [ID2LABEL[int(i)] for i in y],
        "ai_pred": [ID2LABEL[int(i)] for i in ai_argmax],
        "rule_pred": [ID2LABEL[int(i)] for i in test["rule_pred"]],
        "fused_pred": [ID2LABEL[int(i)] for i in primary],
        "prob_HATE": np.round(p_ai[:, 0], 6),
        "prob_OFFENSIVE": np.round(p_ai[:, 1], 6),
        "prob_CLEAN": np.round(p_ai[:, 2], 6),
        "rule_hate_score": np.round(rh, 6),
        "rule_offensive_score": np.round(ro, 6),
        "fused_hate": np.round(per_scores[:, 0], 6),
        "fused_offensive": np.round(per_scores[:, 1], 6),
        "fused_clean": np.round(per_scores[:, 2], 6),
        "true_label_id": y,
        "pred_label_id": primary,
    })
    frame.to_csv(OUT_DIR / "fusion_predictions.csv", index=False, encoding="utf-8")
    (OUT_DIR / "reliability_bins.json").write_text(json.dumps(
        {k: v["bins"] for k, v in calib.items()}, indent=2), encoding="utf-8")

    print("[6/6] Done.")
    print("\n=== Test variants ===")
    for name, m in metrics.items():
        print(f"  {name:32s} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
    print("\n=== Significance ===")
    for key, s in significance.items():
        print(f"  {key}: {s}")
    print("\n=== Calibration ===")
    for key, c in calib.items():
        print(f"  {key:32s} ECE={c['ece']:.4f} Brier={c['brier']:.4f} "
              f"NLL={c['nll']:.4f} conf={c['mean_confidence']:.4f}")
    print("\n=== Provenance ===")
    print(json.dumps(provenance, indent=1))
    print(f"\nSaved {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
