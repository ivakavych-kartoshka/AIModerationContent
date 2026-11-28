# Paper-ready summary for ICCIES

## Focus model
- Strongest by AI macro-F1: **sensitiveai-vi-customdlr2stage**
- Primary fusion variant: **fusion_thresholds**

## Quantitative
- Test size: 6680
- AI-only correct: 5848 (87.54%)
- Fusion correct: 5848 (87.54%)
- AI_wrong -> Fuse_right: 3
- AI_right -> Fuse_wrong: 3
- Net gain: 0

## Explainability-oriented metrics
- Rule coverage: 51.45% of test samples
- Among AI misses corrected by fusion: 33.33% had explicit rule support

## Suggested framing
Title suggestion:
> *"An Explainable Hybrid Approach for Vietnamese Hate Speech Detection: Combining Lexicon-based Rules with a Strong Transformer Model"*

Key points:
- Use only this model as primary case study
- Emphasize interpretability over small accuracy delta
- Highlight obfuscation/GenZ/slang cases
- Use case studies as concrete evidence
