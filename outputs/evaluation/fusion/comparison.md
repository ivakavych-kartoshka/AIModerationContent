# Hybrid fusion - Rule Engine + AI

Rule scores from the frozen rule-engine evaluation; alpha / thresholds / meta tuned on validation, test scored once (Rule_Engine_Plan.md sections 40-44).
`primary` is the variant chosen on validation macro-F1; `Δ vs AI` is primary minus AI-only (argmax).

| model | alpha | AI macro-F1 | rule macro-F1 | fusion(argmax) | fusion(thresholds) | fusion(per-channel) | fusion(meta) | primary (val-chosen) | Δ vs AI |
|---|---|---|---|---|---|---|---|---|---|
| visobert-published | 0.8 | 0.6754 | 0.5258 | 0.6675 | 0.6719 | 0.6744 | 0.6550 | 0.6719 (`fusion_thresholds`) | -0.0036 |
