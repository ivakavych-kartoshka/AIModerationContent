# Fusion report - sensitiveai-vi-customdlr2stage (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.9**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.7, 'OFFENSIVE': 0.5, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.5, 'OFFENSIVE': 0.5, 'CLEAN': 0.5}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8754 | **0.6638** | 0.5943 | 0.4561 | 0.9409 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8756 | **0.6640** | 0.5943 | 0.4567 | 0.9410 |
| fusion_thresholds | 0.8756 | **0.6640** | 0.5943 | 0.4567 | 0.9410 |
| fusion_per_channel_thresholds | 0.8756 | **0.6637** | 0.5953 | 0.4552 | 0.9407 |
| fusion_meta_classifier | 0.8638 | **0.6576** | 0.5863 | 0.4519 | 0.9345 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.6638, HATE 0.5943, OFFENSIVE 0.4561.
