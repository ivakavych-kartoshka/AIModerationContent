# Hybrid fusion - Rule Engine + AI

Rule scores from the frozen rule-engine evaluation; alpha / thresholds / meta tuned on validation, test scored once (Rule_Engine_Plan.md sections 40-44).
`primary` is the variant chosen on validation macro-F1; `Δ vs AI` is primary minus AI-only (argmax).

| model | alpha | AI macro-F1 | rule macro-F1 | fusion(argmax) | fusion(thresholds) | fusion(per-channel) | fusion(meta) | primary (val-chosen) | Δ vs AI |
|---|---|---|---|---|---|---|---|---|---|
| sensitiveai-vi | 0.9 | 0.6545 | 0.5258 | 0.6563 | 0.6575 | 0.6494 | 0.6524 | 0.6494 (`fusion_per_channel_thresholds`) | -0.0051 |
| sensitiveai-vi-custom | 0.9 | 0.6575 | 0.5258 | 0.6583 | 0.6583 | 0.6602 | 0.6513 | 0.6602 (`fusion_per_channel_thresholds`) | +0.0027 |
| sensitiveai-vi-customdlr2stage | 0.9 | 0.6638 | 0.5258 | 0.6640 | 0.6640 | 0.6637 | 0.6576 | 0.6637 (`fusion_per_channel_thresholds`) | -0.0000 |
| sensitiveai-vi-custom-curriculum | 0.9 | 0.6602 | 0.5258 | 0.6611 | 0.6610 | 0.6618 | 0.6531 | 0.6618 (`fusion_per_channel_thresholds`) | +0.0016 |
