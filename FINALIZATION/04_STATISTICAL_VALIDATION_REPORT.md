# 04 Statistical Validation Report

## Verified tests
- Paired bootstrap over comments: 10000 resamples.
- Focused backbone: delta AI-hybrid = -0.0027; 95% CI [-0.0079, +0.0027]; p=0.2942.
- Exact McNemar: b=15, c=23, p=0.2559.

## Duplicate-aware analysis
- Test rows: 6680; distinct texts: 6571; duplicated rows: 109.
- One duplicated text appears twice among the 15 regressions, so regressions cover 14 distinct texts.
- Unique-text-group bootstrap and deduplicated sensitivity analysis are reported for exact duplicates in the leakage report but not fully rerun here.
- McNemar assumes paired observations; repeated comments may affect independence assumptions.

## Limitations
- Single seed (42) per backbone.
- Primary model selected by test-set correction/regression balance.
- No multi-seed variance estimate.
- Do not describe the non-significant improvement as statistically proven.

## Validation-first primary model selection (analysis)
- Criterion: highest standalone (AI-only) validation macro-F1.
- Standalone validation macro-F1:
  - sensitiveai-vi: 0.6961
  - sensitiveai-vi-custom: 0.6957
  - sensitiveai-vi-customdlr2stage: 0.7089
  - sensitiveai-vi-custom-curriculum: 0.6942
- Note: an earlier draft listed 0.7099 / 0.7059 / 0.7139 / 0.7051. Those values match no
  stored artifact and were corrected against `outputs/evaluation/fusion/*/metrics.json`
  (alpha = 1.0) and the per-model validation predictions.
- Under this criterion, `sensitiveai-vi-customdlr2stage` would be selected.
- Test macro-F1 for dlr2stage: 0.6638 standalone, 0.6637 hybrid (near-zero fusion effect,
  3 corrections and 2 regressions per manuscript).
- Therefore, switching the primary model would change the paper's analytical focus. This requires author decision; no manuscript prose was changed.

## Duplicate-aware analysis (completed)
- Test rows: 6680; distinct texts: 6571; duplicated rows: 109.
- Dedup sensitivity: AI macro-F1 0.6574; hybrid macro-F1 0.6606; AI accuracy 0.8758; hybrid accuracy 0.8772.
- Group bootstrap (300 resamples over distinct texts): delta mean -0.0030; 95% CI [-0.0082, 0.0018]; p=0.4933.
- Conclusion: non-significant improvement remains non-significant under duplicate-aware resampling.

## Multi-seed preliminary results (updated)
- Seed 42: AI macro-F1 0.6575; hybrid macro-F1 0.6602; delta +0.0027.
- Seed 7: AI macro-F1 0.6717; hybrid macro-F1 0.6676; delta -0.0041.
- Seed 123: AI macro-F1 0.6627; hybrid macro-F1 0.6617; delta -0.0010.
- Mean delta across 3 seeds: -0.0008.
- Fusion effect remains inconsistent and small; conclusion of no reliable hybrid improvement remains supported.
- Three seeds only; variance estimate still preliminary and not a full replication study.
