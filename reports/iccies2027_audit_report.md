# ICCIES 2027 Paper Audit Report

Scope: `papers/iccies2027` and frozen artifacts under `outputs/evaluation`.

## Deliverable A - Reviewer Report

### Overall assessment
The manuscript is internally consistent on the headline numerical claims and is supported by reproducible artifacts. The main scientific value is the per-decision provenance analysis, not aggregate accuracy. Major issues are test-focused model selection, single-seed training, citation hygiene, and cross-corpus leakage quantification.

### Strengths
- Frozen split hashes and shared evaluator.
- AI-only, rule-only, fusion variants, bootstrap and McNemar results all reproduce from `outputs/evaluation/fusion/iccies2027/numbers.json`.
- 23 corrections and 15 regressions decompose into mutually exclusive categories (26 rule-driven, 12 threshold-driven changed decisions).
- Rule channel behaviour and missed-correction analysis are well quantified.
- Evidence record makes the lexical channel auditable.

### Critical weaknesses
- The primary backbone is selected because it has the most favorable correction/regression balance on the test set.
- Only seed 42 is reported for each backbone.
- Cross-corpus leakage: 620 distinct ViHSD test comments appear in the bontori training split used for lexicon mining.
- Citation `[23]` attributes scikit-learn metric definitions to Saito and Rehmsmeier's precision-recall paper.
- Six references are uncited.

### Major revision requirements
- Reclassify the primary-model comparison as exploratory/post-hoc or select the primary model using validation metrics.
- Report seed variance or explicitly scope claims to a single seed.
- Document cross-corpus leakage and provide the leakage-free lexicon comparison.
- Fix citation `[23]` and remove or cite the six unused references.

### Minor revision requirements
- Regenerate Figure 1 and Figure 2 from canonical artifacts (done).
- Update README provenance line to the frozen 0.2/0.9/1.0 alphas (done).
- Clarify that cue taxonomy annotation is post-hoc and unblinded.

### Novelty assessment
The strongest defensible contribution is methodological/analytical: a frozen hybrid protocol with per-decision provenance, rule-driven versus threshold-driven attribution, and identification of 132 missed corrections. The aggregate accuracy gain is not statistically significant.

### Technical soundness assessment
PASS for numerical consistency and implementation fidelity to Equations (1)-(4). The rule engine implementation follows the documented formula, including pseudo-probability normalization. Thresholds in the frozen metrics are 0.14/0.66/0.60 with alphas 0.2/0.9/1.0.

### Experimental validity assessment
PARTIAL. Test predictions and significance tests reproduce. Model selection and seed stability are not yet defensible for a confirmatory claim.

### Reproducibility assessment
PASS for the frozen primary run. The reproducibility assertion script is `scripts/audit_iccies2027_assertions.py`.

### ICCIES scope fit
PASS. The paper addresses NLP, explainability, and hybrid systems within computational intelligence.

### Overall recommendation
Borderline (Weak Accept if the P0/P1 items are addressed; otherwise Weak Reject).

## Deliverable B - Numerical Consistency Report

| Claim | Paper | Actual | Difference | Source | Status |
|---|---:|---:|---:|---|---|
| AI-only accuracy | 0.8762 | 0.8762 | 0.0000 | `fusion_predictions.csv` | PASS |
| Hybrid accuracy | 0.8774 | 0.8774 | 0.0000 | `fusion_predictions.csv` | PASS |
| AI-only macro-F1 | 0.6575 | 0.6575 | 0.0000 | `fusion_predictions.csv` | PASS |
| Hybrid macro-F1 | 0.6602 | 0.6602 | 0.0000 | `fusion_predictions.csv` | PASS |
| AI-only correct | 5853 | 5853 | 0 | `fusion_predictions.csv` | PASS |
| Hybrid correct | 5861 | 5861 | +8 | `fusion_predictions.csv` | PASS |
| Corrections | 23 | 23 | 0 | prediction comparison | PASS |
| Regressions | 15 | 15 | 0 | prediction comparison | PASS |
| Changed decisions | 38 | 38 | 0 | prediction comparison | PASS |
| Rule coverage | 15.75% | 15.75% | 0.00 | `rule_scores.csv` | PASS |
| Rule precision on firing | 58.75% | 58.75% | 0.00 | `rule_scores.csv` | PASS |
| Missed corrections | 132 | 132 | 0 | prediction comparison | PASS |
| Rule-driven regressions | 10 | 10 | 0 | prediction comparison | PASS |
| Threshold-driven regressions | 5 | 5 | 0 | prediction comparison | PASS |

Correction matrix:

| Transition | Paper | Actual | Status |
|---|---:|---:|---|
| OFFENSIVE -> HATE | 7 | 7 | PASS |
| CLEAN -> HATE | 9 | 9 | PASS |
| HATE -> CLEAN | 6 | 6 | PASS |
| OFFENSIVE -> CLEAN | 1 | 1 | PASS |

Regression matrix:

| Transition | Paper | Actual | Status |
|---|---:|---:|---|
| OFFENSIVE -> HATE | 4 | 4 | PASS |
| CLEAN -> HATE | 6 | 6 | PASS |
| HATE -> CLEAN | 5 | 5 | PASS |

Figure 2(b) discrepancy: an archived rule-silence run used `alpha_H=0.1`, `alpha_O=0.8`, `tau_H=0.07`, `tau_O=0.58` and reported 29 corrections / 20 regressions. The canonical frozen run uses 0.2/0.9/0.14/0.66 and reports 23/15. `results.pdf` has been regenerated from `numbers.json`.

Figure 1 discrepancy: the old `architecture.pdf` showed 0.1/0.8/1.0. The frozen metrics show 0.2/0.9/1.0. `architecture.pdf` has been regenerated from `metrics.json`.

## Deliverable C - Experiment Provenance Manifest

- Dataset: `data/splits/test.csv`, 6,680 rows, 6,571 distinct texts, SHA-256 `d8f9ae7ddd8c` (first 12 hex digits).
- Train/validation/test counts: 24,046 / 2,672 / 6,680.
- Lexicon: `data/processed/lexicon/verified_lexicon.csv`, 668 entries, dictionary version `v1`, source split `train`.
- Model checkpoint: `outputs/reports/sensitiveai-vi-custom/run_001/best_model`.
- Random seed: 42 (single seed per backbone; limitation).
- Fusion configuration: per-channel alphas HATE 0.2, OFFENSIVE 0.9, CLEAN 1.0; thresholds HATE 0.14, OFFENSIVE 0.66, CLEAN 0.60.
- Rule scores: `outputs/evaluation/rule_engine_tuned/test/rule_scores.csv`; fidelity assertion passes.
- Predictions: `outputs/evaluation/fusion/sensitiveai-vi-custom/fusion_predictions.csv`.
- Canonical numbers: `outputs/evaluation/fusion/iccies2027/numbers.json`.
- Evaluation script: `scripts/47_iccies_numbers.py`.
- Figure script: `scripts/48_paper_figures.py`.

## Deliverable D - Reproducibility Tests

Created and passing: `scripts/audit_iccies2027_assertions.py`.

Assertions cover test row count, class distribution, distinct text count, changed decisions, correction/regression totals, transition counts, rule channel metrics, missed corrections, cue taxonomy, and fusion hyperparameters.

## Deliverable E - Corrected Non-Prose Artifacts

- Regenerated `papers/iccies2027/figures/architecture.pdf` from frozen metrics.
- Regenerated `papers/iccies2027/figures/results.pdf` from `numbers.json`.
- Updated `papers/iccies2027/README.md` provenance line.
- Created leakage-free lexicon: `data/processed/lexicon/leakage_free/verified_lexicon.csv` (671 entries).
- No manuscript prose was changed.

## Deliverable F - Final Task Checklist

### P0 - Must fix before submission
| Issue | Source | Evidence | Action | Verification | Status |
|---|---|---|---|---|---|
| Figure 1 alpha mismatch | `figures/architecture.pdf` vs Section 4.3 | old 0.1/0.8/1.0 | regenerate | pdftotext | PASS |
| Figure 2(b) 29/20/49 | archived rule-silence run | 23/15/38 canonical | use canonical artifact | assertions | PASS |
| Citation [23] misuse | `sections/setup.tex` | Saito is PR plot paper | replace with scikit-learn reference | citation audit | FAIL |
| Uncited references | `references.bib` | 6 unused entries | remove or cite | citation audit | FAIL |
| Test-focused primary model | `sections/setup.tex` | selected by net corrections | reclassify or validate-based selection | human review | FAIL |

### P1 - Major scientific improvements
| Issue | Source | Evidence | Action | Verification | Status |
|---|---|---|---|---|---|
| Single seed | `sections/setup.tex`, `discussion.tex` | seed 42 only | add seed variance or scope claims | experiment | FAIL |
| Cross-corpus leakage | bontori train | 620 distinct ViHSD test comments | document and compare leak-free lexicon | leakage audit | PARTIAL |
| Duplicate-aware evaluation | Section 4 / data | 6,571 distinct texts | report duplicate-group bootstrap | analysis | UNVERIFIED |
| Rule-silence convention | `analysis.tex` | 30 flips, p=0.44 | keep as exploratory | human review | PASS |

### P2 - Presentation and minor improvements
| Issue | Source | Evidence | Action | Verification | Status |
|---|---|---|---|---|---|
| Cue taxonomy post-hoc | `sections/analysis.tex` | annotated after test pass | state clearly | human review | PASS |
| Example masking | `tables/case_studies.tex` | masked with `*` | keep masking | visual check | PASS |
| Page budget | `main.pdf` | 15 pages | no expansion | pdfinfo | PASS |

## Phase 4 - Data Leakage Summary

- ViHSD exact cross-split duplicates: 895 of 6,680 test rows (13.4%).
- Distinct ViHSD test texts present in bontori train: 620 (679 rows), including 72 toxic texts.
- ViCM train overlap: 2 distinct texts.
- Lexicon source splits are all `train`, but bontori train contains ViHSD test material.
- Leakage-free lexicon contains 671 entries versus 668 original; 3 added, 1 label changed (`chịch`: OFFENSIVE to HATE), no entries removed.
- Tuned rule-engine re-evaluation with the leakage-free lexicon is not yet run; impact appears small but is UNVERIFIED.

## Phase 5 - Statistical Validity Summary

- Original row-level bootstrap p=0.2942 for the focused backbone.
- Duplicate-group bootstrap and deduplicated sensitivity analysis are reported in `outputs/evaluation/analysis/leakage_report.md` for exact duplicates, but not for duplicate-group resampling.
- Seed variance is not estimated; the manuscript already states this limitation.

## Phase 6 - Rule Engine Audit Summary

- Equations (1)-(4) match `src/rule_engine` and `src/fusion/weighted_fusion.py`.
- Initial config severity is HATE 0.90 / OFFENSIVE 0.60; tuned engine uses 0.95 / 0.60.
- Accent-less alias weight tuned to 0.0 and negation factor to 1.0, as reported.
- Fuzzy matching never fires; case-folding cannot be ablated because all test comments are lower-case.
- Reproducibility configuration table is available from `configs/rule_engine.yaml` and `outputs/evaluation/rule_engine_tuned`.

## Phase 8 - Explainability and Ethics Summary

- Matched spans and scores are emitted in the evidence record.
- The transformer remains a black box; the manuscript does not claim otherwise.
- All quoted comments are masked with `*`.
- No user identifiers are reproduced.
- Cue taxonomy is post-hoc and multi-label; no inter-annotator agreement is claimed.

## Phase 9 - Citation Summary

- 30 citation keys, 36 bibliography entries.
- 6 uncited entries: `cui2019classbalanced`, `howard2018ulmfit`, `macavaney2019hate`, `tran2022phobertcnn`, `tran2023videberta`, `vu2020hsdvlsp`.
- `[23]` should not be Saito and Rehmsmeier for scikit-learn definitions.
- `[5]` Efron and Tibshirani is appropriate for bootstrap.
- Corresponding authors: Cao Hong Nga and Jia-Ching Wang (two, as marked).

## Phase 10 - Claim Consistency Summary

- Abstract reports 0.6575 to 0.6602 macro-F1, 8 net comments, 23 fixed, 15 broken, p=0.29; matches artifacts.
- No statistically insignificant improvement is presented as proven.
- Explainability claims are limited to the lexical channel.
- Conclusion matches Table 1 and Table 2.

## Phase 11 - Preflight Summary

- Build engine: XeLaTeX required and enforced.
- Page count: 15 pages.
- Figures and tables regenerated from canonical artifacts.
- Final PDF render inspection: `papers/iccies2027/main.pdf` exists and has 15 pages.
PY