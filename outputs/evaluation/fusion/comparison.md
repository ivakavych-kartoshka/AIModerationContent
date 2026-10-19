# Hybrid fusion - Rule Engine + AI

Rule scores from the frozen rule-engine evaluation; alpha / thresholds / meta tuned on validation, test scored once (Rule_Engine_Plan.md sections 40-44).
`primary` is the variant chosen on validation macro-F1; `Δ vs AI` is primary minus AI-only (argmax).

| model | alpha | AI macro-F1 | rule macro-F1 | fusion(argmax) | fusion(thresholds) | fusion(per-channel) | fusion(meta) | primary (val-chosen) | Δ vs AI |
|---|---|---|---|---|---|---|---|---|---|
| sensitiveai-vi | 0.95 | 0.6545 | 0.5335 | 0.6547 | 0.6547 | 0.6511 | 0.6553 | 0.6511 (`fusion_per_channel_thresholds`) | -0.0034 |
| sensitiveai-vi-custom | 0.9 | 0.6575 | 0.5335 | 0.6583 | 0.6583 | 0.6604 | 0.6529 | 0.6604 (`fusion_per_channel_thresholds`) | +0.0029 |
| sensitiveai-vi-custom-curriculum | 0.9 | 0.6602 | 0.5335 | 0.6611 | 0.6618 | 0.6647 | 0.6538 | 0.6647 (`fusion_per_channel_thresholds`) | +0.0045 |
| sensitiveai-vi-customdlr2stage | 0.8 | 0.6638 | 0.5335 | 0.6633 | 0.6637 | 0.6645 | 0.6631 | 0.6637 (`fusion_thresholds`) | -0.0001 |
| sensitiveai-vi-customdlr2stage-ext | 1.0 | 0.6536 | 0.5335 | 0.6536 | 0.6536 | 0.6536 | 0.6499 | 0.6536 (`fusion_thresholds`) | +0.0000 |
| sensitiveai-vi-stage2 | 0.95 | 0.6604 | 0.5335 | 0.6619 | 0.6547 | 0.6520 | 0.6477 | 0.6520 (`fusion_per_channel_thresholds`) | -0.0085 |
