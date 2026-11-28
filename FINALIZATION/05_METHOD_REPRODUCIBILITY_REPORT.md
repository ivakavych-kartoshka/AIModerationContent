# 05 Method Reproducibility Report

## Equations
- Equation (1) match score: verified against rule engine scoring.
- Equation (2) saturating aggregation: verified.
- Equation (3) pseudo-probability normalization: verified; zero-input boundary yields (0,0,1).
- Equation (4) fusion: verified; thresholds HATE 0.14, OFFENSIVE 0.66, CLEAN 0.60; alphas 0.2/0.9/1.0.

## Rule engine
- Unicode NFC, invisible removal, whitespace collapse, repeat reduction, lowercasing.
- Matching views: accentless, leet, repeat-collapsed, separator-stripped.
- Match types: exact, accentless_alias, leet, repeat_collapsed, separator_inserted, obfuscated_combined, fuzzy.
- Severity tuned: HATE 0.95, OFFENSIVE 0.60.
- Tuned factors: accentless alias 0.0, repeat-collapsed 0.4, combined obfuscation 0.3, negation 1.0, context-required 1.0.
- K=10 saturation.
- Fuzzy matching: 0 matches on the test split.

## Build
- XeLaTeX required and enforced.
- Final PDF: 15 pages.
- Tests: 9 passed.
