# 03 Data Leakage Report

## Findings
- ViHSD exact cross-split duplicates: 895 of 6680 test rows (13.4%).
- Distinct ViHSD test texts in bontori train: 620 (679 rows), including 72 toxic texts.
- ViCM train overlap: 2 distinct texts.
- Original lexicon claims training-only sources, but bontori train contains ViHSD test material.

## Leakage-free lexicon
- Original entries: 668.
- Leakage-free entries: 671.
- Added: 3 (`bá dơ`, `mà đéo`, `quái thai`).
- Removed: 0.
- Label changes: 1 (`chịch`: OFFENSIVE -> HATE).

## Re-evaluation
- Standalone tuned rule engine with leak-free lexicon:
  - Validation macro-F1: 0.5398 (thresholds HATE 0.76, OFFENSIVE 0.00, CLEAN 0.64).
  - Test macro-F1: 0.5328 (accuracy 0.8440).
- Original tuned rule engine:
  - Validation macro-F1: 0.5428 (thresholds HATE 0.81, OFFENSIVE 0.00, CLEAN 0.64).
  - Test macro-F1: 0.5258 (accuracy 0.8421).
- Leak-free fusion integration completed in separate outputs:
  - Output dir: outputs/evaluation/fusion_leakfree.
  - AI-only: 5853 correct, macro-F1 0.6575.
  - Hybrid: 5863 correct, 23 corrections, 13 regressions, macro-F1 0.6612.
  - Canonical original hybrid remains 5861 correct, 23 corrections, 15 regressions.
  - Alphas/thresholds unchanged: HATE 0.2/0.14, OFFENSIVE 0.9/0.66, CLEAN 1.0/0.60.

## Status
- Leakage risk: QUANTIFIED.
- Mitigation: PARTIAL for standalone; leak-free fusion integration now completed in separate outputs, but canonical manuscript numbers remain original unless author approves.

## Leak-free fusion sensitivity (completed)
- Leak-free rule scores change 35 of 6680 hybrid decisions relative to original rule scores.
- Original hybrid: 5861 correct, 23 corrections, 15 regressions.
- Leak-free sensitivity: 5857 correct, 23 corrections, 19 regressions.
- Impact is small but not negligible; full protocol re-run is still required for a canonical claim.
