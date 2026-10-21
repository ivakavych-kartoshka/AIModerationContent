# Fusion report - sensitiveai-vi-custom-curriculum (run_001)

- Rule scores: `outputs/evaluation/rule_engine_nb`
- mix weight alpha (AI) = **0.7**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.1, 'OFFENSIVE': 0.8, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.5, 'OFFENSIVE': 0.21, 'CLEAN': 0.58}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2672, test n=6680

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.8741 | **0.6602** | 0.5876 | 0.4535 | 0.9395 |
| rule_only_frozen_thresholds | 0.8612 | **0.5805** | 0.4923 | 0.3142 | 0.9349 |
| rule_only_argmax | 0.8612 | **0.4949** | 0.5240 | 0.0310 | 0.9296 |
| fusion_argmax | 0.8756 | **0.6615** | 0.5897 | 0.4547 | 0.9402 |
| fusion_thresholds | 0.8754 | **0.6619** | 0.5906 | 0.4548 | 0.9404 |
| fusion_per_channel_thresholds | 0.8775 | **0.6677** | 0.6084 | 0.4535 | 0.9412 |
| fusion_meta_classifier | 0.8653 | **0.6630** | 0.6003 | 0.4539 | 0.9348 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.6590, HATE 0.5847, OFFENSIVE 0.4535.
