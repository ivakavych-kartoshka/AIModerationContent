import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_numbers():
    n = json.loads((ROOT / "outputs/evaluation/fusion/iccies2027/numbers.json").read_text(encoding="utf-8"))
    assert n["confusion"]["ai_correct"] == 5853
    assert n["confusion"]["fusion_correct"] == 5861
    assert n["confusion"]["fixed"] == 23
    assert n["confusion"]["broken"] == 15
    assert n["confusion"]["both_wrong"] == 804
    assert n["fixed_transitions"] == {"CLEAN<-HATE": 6, "CLEAN<-OFFENSIVE": 1, "HATE<-CLEAN": 9, "HATE<-OFFENSIVE": 7}
    assert n["broken_transitions"] == {"CLEAN->HATE": 6, "HATE->CLEAN": 5, "OFFENSIVE->HATE": 4}
    assert n["rule_channel"]["coverage_count"] == 1052
    assert n["rule_channel"]["coverage_pct"] == 15.75
    assert n["rule_channel"]["precision_on_firing_pct"] == 58.75
    assert n["opportunities"]["missed_corrections"] == 132
    assert n["regressions"]["broken_with_rule_fire"] == 10
    assert n["regressions"]["broken_to_HATE"] == 10
    assert n["cue_taxonomy_hate_corrections"] == {"n": 16, "teencode": 10, "explicit": 5, "derogatory": 8, "no_cue": 1, "multi_label": True}
    assert n["hyperparameters"]["per_channel_alphas"] == {"HATE": 0.2, "OFFENSIVE": 0.9, "CLEAN": 1.0}
    assert n["hyperparameters"]["per_channel_thresholds"] == {"HATE": 0.14, "OFFENSIVE": 0.66, "CLEAN": 0.6}


def test_backbone_values():
    n = json.loads((ROOT / "outputs/evaluation/fusion/iccies2027/numbers.json").read_text(encoding="utf-8"))
    assert n["backbones"]["sensitiveai-vi-custom"]["ai_macro_f1"] == 0.6575
    assert n["backbones"]["sensitiveai-vi-custom"]["fusion_macro_f1"] == 0.6602
    assert n["significance"]["sensitiveai-vi-custom"]["p_bootstrap"] == 0.2942
    assert n["significance"]["sensitiveai-vi-custom"]["mcnemar_b"] == 15
    assert n["significance"]["sensitiveai-vi-custom"]["mcnemar_c"] == 23
