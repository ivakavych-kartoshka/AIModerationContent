# Hybrid fusion - Rule Engine + AI

Rule scores from the frozen rule-engine evaluation; alpha / thresholds / meta tuned on validation, test scored once (Rule_Engine_Plan.md sections 40-44).
`primary` is the variant chosen on validation macro-F1; `Δ vs AI` is primary minus AI-only (argmax).

| model | alpha | AI macro-F1 | rule macro-F1 | fusion(argmax) | fusion(thresholds) | fusion(per-channel) | fusion(meta) | primary (val-chosen) | Δ vs AI |
|---|---|---|---|---|---|---|---|---|---|
| sensitiveai-vi-custom-curriculum | 0.7 | 0.6602 | 0.5805 | 0.6615 | 0.6619 | 0.6677 | 0.6630 | 0.6677 (`fusion_per_channel_thresholds`) | +0.0075 |
