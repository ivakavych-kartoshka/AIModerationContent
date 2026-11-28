# Paper-ready summary for ICCIES

## Focus model
- Model: **sensitiveai-vi-custom**
- Primary fusion variant: **fusion_per_channel_thresholds**

## Quantitative
- Test size: 6680
- AI-only correct: 5853 (87.62%)
- Fusion correct: 5862 (87.75%)
- AI_wrong -> Fuse_right: 29
- AI_right -> Fuse_wrong: 20
- Net gain: 9

## Explainability-oriented metrics
- Rule coverage: 51.45% of test samples
- Among AI misses corrected by fusion: 89.66% had explicit rule support

## Suggested framing
Title suggestion:
> *"An Explainable Hybrid Approach for Vietnamese Hate Speech Detection: Combining Lexicon-based Rules with Transformer Models"*

