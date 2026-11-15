# Manuscript Numeric Patch Guide

Status: factual patch guide only. AI did not generate or rewrite manuscript prose.

Author decisions recorded:
- Keep author metadata exactly as currently in source.
- Keep `sensitiveai-vi-custom` as the focused case study.
- Treat the 3-seed analysis as preliminary.

## 1. Data and lexicon: add leakage and leak-free facts

File: `FINALIZATION/source/iccies2027/sections/data.tex`

Insertion point: after line 43 (end of the lexicon-entry paragraph, before the `551 of 668` paragraph).

Facts to include:
- ViHSD exact cross-split duplicates: 895 of 6,680 test rows (13.4%).
- Distinct ViHSD test texts found in the bontori train material: 620 texts / 679 rows; 72 are toxic.
- ViCM train overlap: 2 distinct texts.
- Original lexicon: 668 entries.
- Leakage-free lexicon: 671 entries.
- Added: 3 entries (`bả do`, `má dào`, `quái thai`).
- Removed: 0.
- Label change: 1 (`ch?ch`, OFFENSIVE -> HATE).
- Standalone tuned rule engine, leakage-free lexicon: validation macro-F1 0.5398; test macro-F1 0.5328; test accuracy 0.8440.
- Original tuned rule engine: validation macro-F1 0.5428; test macro-F1 0.5258; test accuracy 0.8421.
- Leakage-free fusion: 5,863 correct; 23 corrections; 13 regressions; macro-F1 0.6612; accuracy 0.8777.
- Original hybrid: 5,861 correct; 23 corrections; 15 regressions; macro-F1 0.6602; accuracy 0.8774.
- Alphas and thresholds unchanged: HATE 0.2 / 0.14, OFFENSIVE 0.9 / 0.66, CLEAN 1.0 / 0.60.

Do not replace the canonical main-text numbers. Present the leak-free run as a sensitivity/mitigation check.

## 2. Results: add leak-free sensitivity paragraph

File: `FINALIZATION/source/iccies2027/sections/results.tex`

Insertion point: after line 24 (end of the paragraph that reports the standalone rule-engine result), before `\input{tables/fusion_variants}`.

Facts to include:
- Leak-free rule scores change 35 of 6,680 hybrid decisions relative to the original rule scores in the earlier sensitivity pass.
- With the fully retuned leak-free rule engine and frozen fusion protocol, hybrid correct = 5,863.
- Corrections = 23; regressions = 13; changed = 36.
- Leak-free hybrid macro-F1 = 0.6612; accuracy = 0.8777.
- AI-only reference remains 0.6575 macro-F1 and 0.8762 accuracy.
- Direction of conclusion is unchanged: the effect is small and not a reliable accuracy gain.

Optional table: copy `tables/leak_free_sensitivity_tabular.tex` into the source tree (already copied) and wrap it in a `table` float with an author-written caption.

## 3. Discussion/Limitations: update the seed statement

File: `FINALIZATION/source/iccies2027/sections/discussion.tex`

Current paragraph: lines 64--66 (`One seed per backbone.`).

Facts to include:
- The canonical paper still reports seed 42.
- Additional training runs: seed 7 and seed 123.
- Seed 42: AI macro-F1 0.6575; hybrid 0.6602; delta +0.0027.
- Seed 7: AI macro-F1 0.6717; hybrid 0.6676; delta -0.0041.
- Seed 123: AI macro-F1 0.6627; hybrid 0.6617; delta -0.0010.
- Mean delta across 3 seeds: -0.0008.
- Fusion effect is inconsistent across seeds.
- This is a preliminary 3-seed check, not a full variance estimate.

Optional table: copy `tables/multi_seed_preliminary_tabular.tex` into the source tree (already copied) and wrap it with an author-written caption.

## 4. Conclusion: optional one-fact update

File: `FINALIZATION/source/iccies2027/sections/conclusion.tex`

Insertion point: after line 8 (end of the first paragraph).

Fact to include if the author wants the new sensitivity result in the conclusion:
- Leakage-free fusion gives 5,863 correct, 23 corrections, 13 regressions, macro-F1 0.6612; the conclusion remains non-significant/small.
- Three-seed preliminary mean delta: -0.0008.

Do not claim statistical significance or a reliable improvement.

## 5. Suggested table files

Already copied into the source tree:
- `FINALIZATION/source/iccies2027/tables/leak_free_sensitivity_tabular.tex`
- `FINALIZATION/source/iccies2027/tables/multi_seed_preliminary_tabular.tex`

These files contain tabular bodies only. The author must add the `table` float, caption and label.

## 6. Validation after manual edits

Run:
- `python -m pytest FINALIZATION/tests -q`
- `python scripts/audit_iccies2027_assertions.py`
- Rebuild with XeLaTeX.
- Check 15-page limit, citations, and that canonical numbers were not accidentally replaced by sensitivity numbers.
