# Fusion report - sensitiveai-vi-customdlr2stage-ext (run_002)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **1.0**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 1.0, 'OFFENSIVE': 1.0, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.5, 'OFFENSIVE': 0.5, 'CLEAN': 0.5}
- primary variant (chosen on validation) = `fusion_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8783 | **0.6536** | 0.5843 | 0.4350 | 0.9415 |
| rule_only_frozen_thresholds | 0.8398 | **0.5335** | 0.4323 | 0.2418 | 0.9263 |
| rule_only_argmax | 0.8515 | **0.4744** | 0.4947 | 0.0000 | 0.9284 |
| fusion_argmax | 0.8783 | **0.6536** | 0.5843 | 0.4350 | 0.9415 |
| fusion_thresholds | 0.8783 | **0.6536** | 0.5843 | 0.4350 | 0.9415 |
| fusion_per_channel_thresholds | 0.8783 | **0.6536** | 0.5843 | 0.4350 | 0.9415 |
| fusion_meta_classifier | 0.8651 | **0.6499** | 0.5824 | 0.4322 | 0.9351 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.6536, HATE 0.5843, OFFENSIVE 0.4350.
