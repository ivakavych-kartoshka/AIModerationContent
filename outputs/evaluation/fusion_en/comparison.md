# Hybrid fusion - Rule Engine + AI

Rule scores from the frozen rule-engine evaluation; alpha / thresholds / meta tuned on validation, test scored once (Rule_Engine_Plan.md sections 40-44).
`primary` is the variant chosen on validation macro-F1; `Δ vs AI` is primary minus AI-only (argmax).

| model | alpha | AI macro-F1 | rule macro-F1 | fusion(argmax) | fusion(thresholds) | fusion(per-channel) | fusion(meta) | primary (val-chosen) | Δ vs AI |
|---|---|---|---|---|---|---|---|---|---|
| sensitiveai-en-stage2 | 1.0 | 0.7684 | 0.5036 | 0.7684 | 0.7722 | 0.7662 | 0.7255 | 0.7662 (`fusion_per_channel_thresholds`) | -0.0022 |
