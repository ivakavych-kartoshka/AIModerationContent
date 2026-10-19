# Fusion report - sensitiveai-vi-customdlr2stage (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.8**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.6, 'OFFENSIVE': 1.0, 'CLEAN': 0.6}
- frozen thresholds = {'HATE': 0.0, 'OFFENSIVE': 0.53, 'CLEAN': 0.0}
- primary variant (chosen on validation) = `fusion_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8754 | **0.6638** | 0.5943 | 0.4561 | 0.9409 |
| rule_only_frozen_thresholds | 0.8398 | **0.5335** | 0.4323 | 0.2418 | 0.9263 |
| rule_only_argmax | 0.8515 | **0.4744** | 0.4947 | 0.0000 | 0.9284 |
| fusion_argmax | 0.8754 | **0.6633** | 0.5932 | 0.4559 | 0.9408 |
| fusion_thresholds | 0.8754 | **0.6637** | 0.5938 | 0.4568 | 0.9405 |
| fusion_per_channel_thresholds | 0.8762 | **0.6645** | 0.5948 | 0.4575 | 0.9410 |
| fusion_meta_classifier | 0.8684 | **0.6631** | 0.5986 | 0.4537 | 0.9369 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.6638, HATE 0.5943, OFFENSIVE 0.4561.
