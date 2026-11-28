# Fusion report - sensitiveai-vi-custom-curriculum (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.9**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.5, 'OFFENSIVE': 1.0, 'CLEAN': 0.9}
- frozen thresholds = {'HATE': 0.64, 'OFFENSIVE': 0.27, 'CLEAN': 0.0}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8741 | **0.6602** | 0.5876 | 0.4535 | 0.9395 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8747 | **0.6611** | 0.5881 | 0.4556 | 0.9397 |
| fusion_thresholds | 0.8753 | **0.6610** | 0.5893 | 0.4535 | 0.9402 |
| fusion_per_channel_thresholds | 0.8765 | **0.6618** | 0.5927 | 0.4518 | 0.9409 |
| fusion_meta_classifier | 0.8603 | **0.6531** | 0.5733 | 0.4540 | 0.9319 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.6590, HATE 0.5847, OFFENSIVE 0.4535.
