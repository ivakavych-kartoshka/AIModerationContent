# Fusion report - sensitiveai-vi-custom (run_006)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.9**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.2, 'OFFENSIVE': 0.9, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.5, 'OFFENSIVE': 0.5, 'CLEAN': 0.5}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8738 | **0.6627** | 0.5863 | 0.4626 | 0.9392 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8750 | **0.6604** | 0.5891 | 0.4523 | 0.9398 |
| fusion_thresholds | 0.8750 | **0.6604** | 0.5891 | 0.4523 | 0.9398 |
| fusion_per_channel_thresholds | 0.8747 | **0.6617** | 0.5975 | 0.4477 | 0.9399 |
| fusion_meta_classifier | 0.8515 | **0.6510** | 0.5903 | 0.4350 | 0.9279 |

AI-only as reported by the evaluator (rule=`argmax`): macro-F1 0.6627, HATE 0.5863, OFFENSIVE 0.4626.
