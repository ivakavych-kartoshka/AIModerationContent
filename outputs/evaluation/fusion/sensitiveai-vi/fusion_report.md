# Fusion report - sensitiveai-vi (run_001)

- Rule scores: `outputs/evaluation/rule_engine_tuned`
- mix weight alpha (AI) = **0.95**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.2, 'OFFENSIVE': 0.8, 'CLEAN': 0.95}
- frozen thresholds = {'HATE': 0.5, 'OFFENSIVE': 0.5, 'CLEAN': 0.5}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8734 | **0.6545** | 0.5864 | 0.4367 | 0.9405 |
| rule_only_frozen_thresholds | 0.8398 | **0.5335** | 0.4323 | 0.2418 | 0.9263 |
| rule_only_argmax | 0.8515 | **0.4744** | 0.4947 | 0.0000 | 0.9284 |
| fusion_argmax | 0.8735 | **0.6547** | 0.5859 | 0.4377 | 0.9406 |
| fusion_thresholds | 0.8735 | **0.6547** | 0.5859 | 0.4377 | 0.9406 |
| fusion_per_channel_thresholds | 0.8717 | **0.6511** | 0.5910 | 0.4226 | 0.9398 |
| fusion_meta_classifier | 0.8683 | **0.6553** | 0.6026 | 0.4249 | 0.9384 |

AI-only as reported by the evaluator (rule=`argmax`): macro-F1 0.6545, HATE 0.5864, OFFENSIVE 0.4367.
