# Fusion report - sensitiveai-vi-custom (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.9**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.2, 'OFFENSIVE': 0.8, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.5, 'OFFENSIVE': 0.5, 'CLEAN': 0.5}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8762 | **0.6575** | 0.5830 | 0.4493 | 0.9403 |
| rule_only_frozen_thresholds | 0.8398 | **0.5335** | 0.4323 | 0.2418 | 0.9263 |
| rule_only_argmax | 0.8515 | **0.4744** | 0.4947 | 0.0000 | 0.9284 |
| fusion_argmax | 0.8769 | **0.6583** | 0.5851 | 0.4490 | 0.9408 |
| fusion_thresholds | 0.8769 | **0.6583** | 0.5851 | 0.4490 | 0.9408 |
| fusion_per_channel_thresholds | 0.8775 | **0.6604** | 0.5904 | 0.4499 | 0.9407 |
| fusion_meta_classifier | 0.8638 | **0.6529** | 0.5856 | 0.4382 | 0.9348 |

AI-only as reported by the evaluator (rule=`argmax`): macro-F1 0.6575, HATE 0.5830, OFFENSIVE 0.4493.
