# Fusion report - visobert-published (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.8**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 1.0, 'OFFENSIVE': 1.0, 'CLEAN': 0.7}
- frozen thresholds = {'HATE': 0.58, 'OFFENSIVE': 0.34, 'CLEAN': 0.57}
- primary variant (chosen on validation) = `fusion_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8795 | **0.6754** | 0.6160 | 0.4688 | 0.9416 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8823 | **0.6675** | 0.6206 | 0.4392 | 0.9426 |
| fusion_thresholds | 0.8796 | **0.6719** | 0.6017 | 0.4718 | 0.9421 |
| fusion_per_channel_thresholds | 0.8807 | **0.6744** | 0.6048 | 0.4761 | 0.9425 |
| fusion_meta_classifier | 0.8516 | **0.6550** | 0.5985 | 0.4385 | 0.9279 |

AI-only as reported by the evaluator (rule=`argmax`): macro-F1 0.6754, HATE 0.6160, OFFENSIVE 0.4688.
