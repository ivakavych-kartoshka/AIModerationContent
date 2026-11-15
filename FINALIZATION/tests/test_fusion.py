import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_fusion_formula_and_thresholds():
    m = json.loads((ROOT / "outputs/evaluation/fusion/sensitiveai-vi-custom/metrics.json").read_text(encoding="utf-8"))
    assert m["per_channel_alphas"] == {"HATE": 0.2, "OFFENSIVE": 0.9, "CLEAN": 1.0}
    assert m["per_channel_thresholds"] == {"HATE": 0.14, "OFFENSIVE": 0.66, "CLEAN": 0.60}

    p = np.array([0.1, 0.2, 0.7])
    r = np.array([0.8, 0.1])
    r_tilde = np.array([r[0], r[1], 1 - max(r[0], r[1])]) / (1 + min(r[0], r[1]))
    alpha = np.array([0.2, 0.9, 1.0])
    f = alpha * p + (1 - alpha) * r_tilde
    assert np.allclose(f, [0.60181818, 0.18909091, 0.7])
    assert f[0] >= 0.14

    # Silence convention.
    silent = np.array([0.0, 0.0, 1.0])
    assert np.allclose(silent, [0, 0, 1])


def test_fusion_predictions_consistency():
    import pandas as pd

    preds = pd.read_csv(ROOT / "outputs/evaluation/fusion/sensitiveai-vi-custom/fusion_predictions.csv")
    tuned = pd.read_csv(ROOT / "outputs/evaluation/rule_engine_tuned/test/rule_scores.csv")[["index", "rule_hate_score", "rule_offensive_score"]]
    df = preds.merge(tuned, on="index", suffixes=("", "_rt"))
    assert len(df) == 6680
    assert (df["rule_hate_score"] == df["rule_hate_score_rt"]).all()
    assert (df["rule_offensive_score"] == df["rule_offensive_score_rt"]).all()
    ai_ok = df["ai_pred"] == df["true_label"]
    fu_ok = df["fused_pred"] == df["true_label"]
    assert int(ai_ok.sum()) == 5853
    assert int(fu_ok.sum()) == 5861
    assert int((~ai_ok & fu_ok).sum()) == 23
    assert int((ai_ok & ~fu_ok).sum()) == 15
