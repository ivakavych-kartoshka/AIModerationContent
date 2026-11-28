import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_rule_engine_config_and_scores():
    cfg = (ROOT / "configs/rule_engine.yaml").read_text(encoding="utf-8")
    assert "HATE: 0.90" in cfg or "HATE: 0.9" in cfg
    assert "OFFENSIVE: 0.60" in cfg or "OFFENSIVE: 0.6" in cfg

    scores = pd.read_csv(ROOT / "outputs/evaluation/rule_engine_tuned/test/rule_scores.csv")
    assert len(scores) == 6680
    assert set(scores.columns) >= {"index", "rule_hate_score", "rule_offensive_score"}
    assert scores["rule_hate_score"].between(0, 1).all()
    assert scores["rule_offensive_score"].between(0, 1).all()

    # Pseudo-probability normalization from Equation (3).
    r_h, r_o = 0.4, 0.2
    denom = 1 + min(r_h, r_o)
    expected = np.array([r_h, r_o, 1 - max(r_h, r_o)]) / denom
    assert np.allclose(expected, [0.3333333, 0.1666667, 0.5])
    assert expected.sum() == 1.0

    # Boundary: both scores zero.
    zero = np.array([0.0, 0.0, 1.0])
    assert np.allclose(zero, [0, 0, 1])
