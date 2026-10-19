# Fusion report - sensitiveai-vi-custom-curriculum (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.9**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.2, 'OFFENSIVE': 1.0, 'CLEAN': 0.9}
- frozen thresholds = {'HATE': 0.64, 'OFFENSIVE': 0.27, 'CLEAN': 0.0}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8741 | **0.6602** | 0.5876 | 0.4535 | 0.9395 |
| rule_only_frozen_thresholds | 0.8398 | **0.5335** | 0.4323 | 0.2418 | 0.9263 |
| rule_only_argmax | 0.8515 | **0.4744** | 0.4947 | 0.0000 | 0.9284 |
| fusion_argmax | 0.8747 | **0.6611** | 0.5881 | 0.4556 | 0.9397 |
| fusion_thresholds | 0.8754 | **0.6618** | 0.5898 | 0.4553 | 0.9402 |
| fusion_per_channel_thresholds | 0.8769 | **0.6647** | 0.5920 | 0.4612 | 0.9408 |
| fusion_meta_classifier | 0.8618 | **0.6538** | 0.5765 | 0.4517 | 0.9330 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.6590, HATE 0.5847, OFFENSIVE 0.4535.
