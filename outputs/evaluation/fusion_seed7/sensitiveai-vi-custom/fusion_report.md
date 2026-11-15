# Fusion report - sensitiveai-vi-custom (run_005)

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
| ai_only_argmax | 0.8744 | **0.6717** | 0.6097 | 0.4667 | 0.9385 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8753 | **0.6713** | 0.6117 | 0.4636 | 0.9385 |
| fusion_thresholds | 0.8753 | **0.6713** | 0.6117 | 0.4636 | 0.9385 |
| fusion_per_channel_thresholds | 0.8750 | **0.6676** | 0.6133 | 0.4505 | 0.9390 |
| fusion_meta_classifier | 0.8524 | **0.6548** | 0.5966 | 0.4423 | 0.9255 |

AI-only as reported by the evaluator (rule=`argmax`): macro-F1 0.6717, HATE 0.6097, OFFENSIVE 0.4667.
