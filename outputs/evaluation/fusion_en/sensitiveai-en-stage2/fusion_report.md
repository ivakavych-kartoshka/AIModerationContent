# Fusion report - sensitiveai-en-stage2 (run_001)

- Rule scores: `outputs/evaluation/rule_engine_en`
- mix weight alpha (AI) = **1.0**  (0 = rule only, 1 = AI only)
- per-channel alphas (AI weight per label) = {'HATE': 0.8, 'OFFENSIVE': 1.0, 'CLEAN': 1.0}
- frozen thresholds = {'HATE': 0.29, 'OFFENSIVE': 0.48, 'CLEAN': 0.5}
- primary variant (chosen on validation) = `fusion_per_channel_thresholds`
- validation n=2477, test n=2478

Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.

## Test metrics

| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |
|---|---|---|---|---|---|
| ai_only_argmax | 0.9080 | **0.7684** | 0.4542 | 0.9436 | 0.9072 |
| rule_only_frozen_thresholds | 0.7058 | **0.5036** | 0.1782 | 0.8063 | 0.5264 |
| rule_only_argmax | 0.6755 | **0.4973** | 0.2009 | 0.7781 | 0.5130 |
| fusion_argmax | 0.9080 | **0.7684** | 0.4542 | 0.9436 | 0.9072 |
| fusion_thresholds | 0.9080 | **0.7722** | 0.4656 | 0.9440 | 0.9070 |
| fusion_per_channel_thresholds | 0.9096 | **0.7662** | 0.4444 | 0.9451 | 0.9091 |
| fusion_meta_classifier | 0.8442 | **0.7255** | 0.3718 | 0.8961 | 0.9085 |

AI-only as reported by the evaluator (rule=`thresholds`): macro-F1 0.7722, HATE 0.4656, OFFENSIVE 0.9440.
