# Fusion report - sensitiveai-vi-custom (run_001)

- Rule scores: `outputs/evaluation/rule_ablations/no_fuzzy`
- mix weight alpha (AI) = **0.9**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.2, 'OFFENSIVE': 0.9, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.5, 'OFFENSIVE': 0.5, 'CLEAN': 0.5}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8762 | **0.6575** | 0.5830 | 0.4493 | 0.9403 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8769 | **0.6583** | 0.5851 | 0.4490 | 0.9408 |
| fusion_thresholds | 0.8769 | **0.6583** | 0.5851 | 0.4490 | 0.9408 |
| fusion_per_channel_thresholds | 0.8774 | **0.6602** | 0.5906 | 0.4494 | 0.9407 |
| fusion_meta_classifier | 0.8624 | **0.6511** | 0.5815 | 0.4379 | 0.9340 |

AI-only as reported by the evaluator (rule=`argmax`): macro-F1 0.6575, HATE 0.5830, OFFENSIVE 0.4493.
