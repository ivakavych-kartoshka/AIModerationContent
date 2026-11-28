#!/usr/bin/env python
"""Automated reproducibility assertions for the ICCIES 2027 paper.

Run:
    $env:PYTHONIOENCODING="utf-8"
    python scripts/audit_iccies2027_assertions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "outputs/evaluation/fusion/iccies2027/numbers.json"
PRED = ROOT / "outputs/evaluation/fusion/sensitiveai-vi-custom/fusion_predictions.csv"
RULE = ROOT / "outputs/evaluation/rule_engine_tuned/test/rule_scores.csv"


def fail(msg: str) -> None:
    print("FAIL:", msg)
    sys.exit(1)


def main() -> int:
    n = json.loads(NUMBERS.read_text(encoding="utf-8"))
    preds = pd.read_csv(PRED, encoding="utf-8")
    tuned = pd.read_csv(RULE, encoding="utf-8")[["index", "rule_hate_score", "rule_offensive_score"]]
    df = preds.merge(tuned, on="index", suffixes=("", "_rt"))

    # Rule-score fidelity
    if not np.allclose(df["rule_hate_score"], df["rule_hate_score_rt"]):
        fail("shipped rule scores differ from tuned rule engine")
    if not np.allclose(df["rule_offensive_score"], df["rule_offensive_score_rt"]):
        fail("shipped offensive rule scores differ from tuned rule engine")

    # Row counts / labels
    assert len(df) == 6680, len(df)
    assert (df["true_label"].value_counts().to_dict() == {"CLEAN": 5548, "HATE": 688, "OFFENSIVE": 444})
    assert df["text"].nunique() == 6571

    # Numerical claims
    assert n["confusion"]["ai_correct"] == 5853
    assert n["confusion"]["fusion_correct"] == 5861
    assert n["confusion"]["fixed"] == 23
    assert n["confusion"]["broken"] == 15
    assert n["confusion"]["both_wrong"] == 804
    assert n["fixed_transitions"] == {
        "CLEAN<-HATE": 6,
        "CLEAN<-OFFENSIVE": 1,
        "HATE<-CLEAN": 9,
        "HATE<-OFFENSIVE": 7,
    }
    assert n["broken_transitions"] == {
        "CLEAN->HATE": 6,
        "HATE->CLEAN": 5,
        "OFFENSIVE->HATE": 4,
    }
    assert n["rule_channel"]["coverage_count"] == 1052
    assert n["rule_channel"]["coverage_pct"] == 15.75
    assert n["rule_channel"]["precision_on_firing_pct"] == 58.75
    assert n["opportunities"]["missed_corrections"] == 132
    assert n["regressions"]["broken_with_rule_fire"] == 10
    assert n["regressions"]["broken_to_HATE"] == 10
    assert n["cue_taxonomy_hate_corrections"] == {
        "n": 16, "teencode": 10, "explicit": 5, "derogatory": 8,
        "no_cue": 1, "multi_label": True,
    }

    # Recompute confusion from prediction file
    ai_ok = df["ai_pred"] == df["true_label"]
    fu_ok = df["fused_pred"] == df["true_label"]
    assert int(ai_ok.sum()) == 5853
    assert int(fu_ok.sum()) == 5861
    assert int((~ai_ok & fu_ok).sum()) == 23
    assert int((ai_ok & ~fu_ok).sum()) == 15

    # Fusion hyperparameters canonical
    assert n["hyperparameters"]["per_channel_alphas"] == {"HATE": 0.2, "OFFENSIVE": 0.9, "CLEAN": 1.0}
    assert n["hyperparameters"]["per_channel_thresholds"] == {"HATE": 0.14, "OFFENSIVE": 0.66, "CLEAN": 0.6}

    print("PASS: all ICCIES 2027 reproducibility assertions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY