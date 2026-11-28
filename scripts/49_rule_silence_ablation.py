"""Rule-silence audit (revision protocol, Phase 2.5.1): CURRENT vs NEUTRAL.

Reads the frozen test artefacts of the primary fusion only - nothing is trained,
no existing result file is overwritten.  Writes a single new JSON:

    outputs/evaluation/fusion/sensitiveai-vi-custom/rule_silence.json

The three protocol configurations:

    CURRENT     silence -> rule pseudo-probability (0, 0, 1), i.e. the frozen
                per-channel alphas (0.1, 0.8, 1.0) are applied to every comment.
    NEUTRAL     silence -> the lexicon contributes nothing, so the neural scores
                pass through unchanged (alpha = 1 on the silent rows).
    CALIBRATED  silence -> CLEAN evidence only where a separately validated CLEAN
                signal exists.  The rule engine has no such component (its CLEAN
                score is the derived 1 - max(r_H, r_O)), so this configuration is
                reported as PENDING instead of being approximated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import apply_decision_rule, compute_metrics
from src.fusion.weighted_fusion import fuse_scores_per_channel, rule_pseudo_proba

ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = ROOT / "outputs/evaluation/fusion/sensitiveai-vi-custom"
ALPHA_ORDER = ["HATE", "OFFENSIVE", "CLEAN"]


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict) -> dict:
    """Confusion matrix plus the two error rates the protocol asks for."""
    labels = ALPHA_ORDER
    cm = metrics["confusion_matrix"]
    out = {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "confusion_matrix": cm}
    for lab in labels:
        out[f"{lab}_precision"] = metrics[f"{lab}_precision"]
        out[f"{lab}_recall"] = metrics[f"{lab}_recall"]
        out[f"{lab}_f1"] = metrics[f"{lab}_f1"]
    # CLEAN false-positive rate: clean comments flagged as toxic.
    clean_id = 2
    out["CLEAN_fpr"] = float((y_pred[y_true == clean_id] != clean_id).mean())
    # HATE / OFFENSIVE false-negative rate: toxic comments passed as clean.
    out["HATE_fnr"] = float((y_pred[y_true == 0] == clean_id).mean())
    out["OFFENSIVE_fnr"] = float((y_pred[y_true == 1] == clean_id).mean())
    return out


def main() -> None:
    df = pd.read_csv(FUSION_DIR / "fusion_predictions.csv")
    meta = json.loads((FUSION_DIR / "metrics.json").read_text(encoding="utf-8"))

    y_true = df["true_label"].map({k: i for i, k in enumerate(ALPHA_ORDER)}).to_numpy()
    p_ai = df[["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]].to_numpy(dtype=float)
    r_h = df["rule_hate_score"].to_numpy(dtype=float)
    r_o = df["rule_offensive_score"].to_numpy(dtype=float)

    alphas = np.array([meta["per_channel_alphas"][k] for k in ALPHA_ORDER], dtype=float)
    thr_payload = meta["per_channel_thresholds"]
    thresholds = np.array([thr_payload[k] for k in ALPHA_ORDER], dtype=float)
    ai_pred = df["ai_pred"].map({k: i for i, k in enumerate(ALPHA_ORDER)}).to_numpy()
    frozen_pred = df["fused_pred"].map({k: i for i, k in enumerate(ALPHA_ORDER)}).to_numpy()

    silent = (r_h == 0.0) & (r_o == 0.0)

    # ---- CURRENT: reproduce the frozen primary variant exactly --------------
    scores_current = fuse_scores_per_channel(p_ai, r_h, r_o, alphas)
    pred_current = apply_decision_rule(scores_current, "thresholds", thresholds)
    reference_macro = meta["test"]["fusion_per_channel_thresholds"]["macro_f1"]
    reproduced = abs(float(compute_metrics(y_true, pred_current)["macro_f1"]) - reference_macro) < 1e-9
    agrees_with_frozen_pred = bool((pred_current == frozen_pred).all())
    if not (reproduced and agrees_with_frozen_pred):
        raise SystemExit(
            "Fidelity check failed: this script does not reproduce the shipped primary "
            f"variant (macro_f1 match={reproduced}, prediction match={agrees_with_frozen_pred})."
        )

    # ---- NEUTRAL: no contribution where the lexicon is silent --------------
    rule = rule_pseudo_proba(r_h, r_o)
    alpha_rows = np.tile(alphas, (len(df), 1))
    alpha_rows[silent] = 1.0  # silence -> pure neural scores
    scores_neutral = alpha_rows * p_ai + (1.0 - alpha_rows) * rule
    pred_neutral = apply_decision_rule(scores_neutral, "thresholds", thresholds)

    report = {
        "protocol_note": (
            "Frozen test scores, frozen validation-selected alphas and thresholds. "
            "No parameter was re-tuned; both configurations use the alphas and "
            "thresholds of the shipped primary variant, so the difference isolates "
            "the rule-silence convention."
        ),
        "test_size": int(len(df)),
        "silent_fraction": float(silent.mean()),
        "alphas": dict(zip(ALPHA_ORDER, alphas.tolist())),
        "thresholds": dict(thr_payload),
        "fidelity": {
            "reproduced_macro_f1": reproduced,
            "reproduced_predictions": agrees_with_frozen_pred,
        },
        "CALIBRATED": "PENDING - the rule engine has no validated CLEAN signal; "
                      "its CLEAN score is the derived 1 - max(r_H, r_O).",
    }

    for name, pred in (("CURRENT", pred_current), ("NEUTRAL", pred_neutral)):
        m = compute_metrics(y_true, pred)
        block = per_class_report(y_true, pred, m)
        block["correct"] = int((pred == y_true).sum())
        block["corrections_vs_ai"] = int(((ai_pred != y_true) & (pred == y_true)).sum())
        block["regressions_vs_ai"] = int(((ai_pred == y_true) & (pred != y_true)).sum())
        block["flips_vs_current"] = int((pred != pred_current).sum())
        report[name] = block

    # Where do the two configurations disagree, and who is right?
    changed = pred_current != pred_neutral
    report["disagreement"] = {
        "n_changed": int(changed.sum()),
        "neutral_wins": int((changed & (pred_neutral == y_true) & (pred_current != y_true)).sum()),
        "current_wins": int((changed & (pred_current == y_true) & (pred_neutral != y_true)).sum()),
        "both_wrong": int((changed & (pred_current != y_true) & (pred_neutral != y_true)).sum()),
        "all_on_silent_rows": bool(
            changed.sum() == 0 or bool(silent[changed].all())
        ),
    }
    # The regressions that Sect. 6.3 attributes to "threshold-driven" decisions
    # must all sit on silent rows; this checks that attribution claim.
    ai_right = ai_pred == y_true
    report["regression_check"] = {
        "current_regressions_on_silent_rows": int(((ai_right) & (pred_current != y_true) & silent).sum()),
        "current_regressions_total": int(((ai_right) & (pred_current != y_true)).sum()),
    }

    # Paired significance on the same frozen predictions: does the convention
    # actually move the metric, or is the 31-comment disagreement noise?
    from src.evaluation.analysis import mcnemar, paired_bootstrap_delta

    report["significance"] = {
        "macro_f1_neutral_minus_current": paired_bootstrap_delta(
            y_true, pred_neutral, pred_current, "macro_f1", n_boot=10000, seed=42
        ),
        "accuracy_neutral_minus_current": paired_bootstrap_delta(
            y_true, pred_neutral, pred_current, "accuracy", n_boot=10000, seed=42
        ),
        "mcnemar_current_vs_neutral": mcnemar(y_true, pred_current, pred_neutral),
    }

    out = FUSION_DIR / "rule_silence.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"silent rows: {silent.sum()}/{len(df)} ({100 * silent.mean():.1f}%)")
    print(f"fidelity: macro-F1 reproduced={reproduced}, predictions reproduced={agrees_with_frozen_pred}")
    for name in ("CURRENT", "NEUTRAL"):
        b = report[name]
        print(
            f"{name:8s} macro-F1={b['macro_f1']:.4f} acc={b['accuracy']:.4f} "
            f"HATE={b['HATE_f1']:.4f} OFF={b['OFFENSIVE_f1']:.4f} CLEAN={b['CLEAN_f1']:.4f} "
            f"corr={b['corrections_vs_ai']} regr={b['regressions_vs_ai']}"
        )
    print("disagreement:", report["disagreement"])
    print("threshold-regression check:", report["regression_check"])
    sig = report["significance"]
    print(
        "neutral - current macro-F1: delta=%.5f CI=[%.4f, %.4f] p=%.3f"
        % (
            sig["macro_f1_neutral_minus_current"]["delta"],
            sig["macro_f1_neutral_minus_current"]["ci_low"],
            sig["macro_f1_neutral_minus_current"]["ci_high"],
            sig["macro_f1_neutral_minus_current"]["p_value"],
        )
    )
    print(
        "neutral - current accuracy: delta=%.5f CI=[%.4f, %.4f] p=%.3f"
        % (
            sig["accuracy_neutral_minus_current"]["delta"],
            sig["accuracy_neutral_minus_current"]["ci_low"],
            sig["accuracy_neutral_minus_current"]["ci_high"],
            sig["accuracy_neutral_minus_current"]["p_value"],
        )
    )
    print("mcnemar current vs neutral:", sig["mcnemar_current_vs_neutral"])
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
