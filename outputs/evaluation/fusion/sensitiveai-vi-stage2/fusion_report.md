# Fusion report - sensitiveai-vi-stage2 (run_002)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.95**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.5, 'OFFENSIVE': 1.0, 'CLEAN': 0.7}
- frozen thresholds = {'HATE': 0.8, 'OFFENSIVE': 0.24, 'CLEAN': 0.08}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8633 | **0.6604** | 0.6088 | 0.4382 | 0.9344 |
| rule_only_frozen_thresholds | 0.8398 | **0.5335** | 0.4323 | 0.2418 | 0.9263 |
| rule_only_argmax | 0.8515 | **0.4744** | 0.4947 | 0.0000 | 0.9284 |
| fusion_argmax | 0.8651 | **0.6619** | 0.6114 | 0.4388 | 0.9354 |
| fusion_thresholds | 0.8680 | **0.6547** | 0.5817 | 0.4438 | 0.9385 |
| fusion_per_channel_thresholds | 0.8678 | **0.6520** | 0.5834 | 0.4319 | 0.9407 |
| fusion_meta_classifier | 0.8482 | **0.6477** | 0.5996 | 0.4180 | 0.9256 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.6604, HATE 0.6088, OFFENSIVE 0.4382.
