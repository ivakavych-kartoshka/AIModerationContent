# Fusion report - visobert-base (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.95**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.5, 'OFFENSIVE': 1.0, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.62, 'OFFENSIVE': 0.22, 'CLEAN': 0.66}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8716 | **0.6548** | 0.5836 | 0.4439 | 0.9369 |
| rule_only_frozen_thresholds | 0.8421 | **0.5258** | 0.4102 | 0.2389 | 0.9284 |
| rule_only_argmax | 0.8494 | **0.4714** | 0.4867 | 0.0000 | 0.9274 |
| fusion_argmax | 0.8726 | **0.6558** | 0.5899 | 0.4407 | 0.9369 |
| fusion_thresholds | 0.8660 | **0.6500** | 0.5655 | 0.4487 | 0.9360 |
| fusion_per_channel_thresholds | 0.8738 | **0.6622** | 0.6109 | 0.4370 | 0.9388 |
| fusion_meta_classifier | 0.8518 | **0.6483** | 0.5964 | 0.4223 | 0.9262 |
