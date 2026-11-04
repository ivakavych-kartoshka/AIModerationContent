# Fusion report - sensitiveai-vi (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.9**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.3, 'OFFENSIVE': 1.0, 'CLEAN': 0.95}
- frozen thresholds = {'HATE': 0.26, 'OFFENSIVE': 0.41, 'CLEAN': 0.59}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8734 | **0.6545** | 0.5864 | 0.4367 | 0.9405 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8746 | **0.6563** | 0.5891 | 0.4389 | 0.9411 |
| fusion_thresholds | 0.8740 | **0.6575** | 0.5934 | 0.4385 | 0.9407 |
| fusion_per_channel_thresholds | 0.8687 | **0.6494** | 0.5827 | 0.4271 | 0.9386 |
| fusion_meta_classifier | 0.8641 | **0.6524** | 0.5947 | 0.4267 | 0.9359 |

AI-only as reported by the evaluator (rule=`argmax`): macro-F1 0.6545, HATE 0.5864, OFFENSIVE 0.4367.
