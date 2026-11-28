# 02 Numerical Validation Report

## Canonical results
- AI-only accuracy: 0.8762; macro-F1: 0.6575.
- Primary hybrid accuracy: 0.8774; macro-F1: 0.6602.
- AI-only correct: 5853; hybrid correct: 5861; net +8.
- Corrections: 23; regressions: 15; changed decisions: 38; both wrong: 804.

## Correction matrix
- OFFENSIVE -> HATE: 7
- CLEAN -> HATE: 9
- HATE -> CLEAN: 6
- OFFENSIVE -> CLEAN: 1

## Regression matrix
- OFFENSIVE -> HATE: 4
- CLEAN -> HATE: 6
- HATE -> CLEAN: 5

## Rule channel
- Coverage: 1052 comments, 15.75%.
- Precision on firing: 58.75%.
- Recall on non-CLEAN: 54.59%.
- False alarms on CLEAN: 7.82%.
- Missed corrections: 132.
- Rule-driven regressions: 10.
- Threshold-driven regressions: 5.

## Significance
- Focused backbone: delta(AI-hybrid) = -0.0027; 95% CI [-0.0079, +0.0027]; p_bootstrap=0.2942.
- McNemar: b=15, c=23, p=0.2559.

## Verification
- `scripts/audit_iccies2027_assertions.py`: PASS.
- `FINALIZATION/tests`: 9 passed.
