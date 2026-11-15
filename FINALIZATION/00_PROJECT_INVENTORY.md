# 00 Project Inventory

## Repository state
- Branch: `sensitiveai-vi`
- Commit: `62e5f079a7f7d60160c503c75d678ec2be56520f`
- Working tree contains untracked research artifacts; no commits or pushes were made.

## Core artifacts
- Manuscript source: `papers/iccies2027/`
- Canonical numbers: `outputs/evaluation/fusion/iccies2027/numbers.json`
- Fusion predictions: `outputs/evaluation/fusion/sensitiveai-vi-custom/fusion_predictions.csv`
- Fusion metrics: `outputs/evaluation/fusion/sensitiveai-vi-custom/metrics.json`
- Rule scores: `outputs/evaluation/rule_engine_tuned/test/rule_scores.csv`
- Rule thresholds: `outputs/evaluation/rule_engine_tuned/thresholds.json`
- Significance: `outputs/evaluation/fusion/fusion_significance.json`
- Frozen splits: `data/splits/{train,validation,test}.csv`
- Original lexicon: `data/processed/lexicon/verified_lexicon.csv` (668 entries)
- Leakage-free lexicon: `data/processed/lexicon/leakage_free/verified_lexicon.csv` (671 entries)
- Figures: `papers/iccies2027/figures/{architecture,results}.pdf`
- Tests: `FINALIZATION/tests/`
- Final PDF: `FINALIZATION/submission/iccies2027_final.pdf`

## Scripts
- `scripts/47_iccies_numbers.py`: canonical quantitative claims.
- `scripts/48_paper_figures.py`: regenerated figures from frozen artifacts.
- `scripts/audit_iccies2027_assertions.py`: reproducibility assertions.
- `scripts/17_evaluate_rule_engine.py`: rule-engine tuning and test evaluation.

## Status
- Numerical integrity: PASS.
- Figure regeneration: PASS.
- Leakage audit: PARTIAL (lexicon comparison available; fusion re-evaluation with leak-free lexicon not fully integrated).
- Multi-seed training: NOT PERFORMED (documented limitation).
- Citation/prose issues: UNRESOLVED pending human author action.
