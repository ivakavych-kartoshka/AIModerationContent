# ICCIES 2027 - paper numbers (single source of truth)

Focus model: `sensitiveai-vi-custom`; rule evidence threshold `0.01`; rule scores from `outputs\evaluation\rule_engine_tuned\test\rule_scores.csv`.

## Test distribution

- CLEAN: 5548 (83.05%)
- HATE: 688 (10.30%)
- OFFENSIVE: 444 (6.65%)

## Components

| system | acc | macro-F1 | HATE F1 | OFF F1 | CLEAN F1 |
|---|---|---|---|---|---|
| AI-only (argmax) | 0.8762 | 0.6575 | 0.5830 | 0.4493 | 0.9403 |
| ai_only_argmax | 0.8762 | 0.6575 | 0.5830 | 0.4493 | 0.9403 |
| rule_only_argmax | 0.8494 | 0.4714 | 0.4867 | 0.0000 | 0.9274 |
| rule_only_frozen_thresholds | 0.8421 | 0.5258 | 0.4102 | 0.2389 | 0.9284 |
| fusion_argmax | 0.8769 | 0.6583 | 0.5851 | 0.4490 | 0.9408 |
| fusion_thresholds | 0.8769 | 0.6583 | 0.5851 | 0.4490 | 0.9408 |
| fusion_meta_classifier | 0.8626 | 0.6513 | 0.5818 | 0.4379 | 0.9341 |
| fusion_per_channel_argmax | 0.8774 | 0.6593 | 0.5882 | 0.4488 | 0.9408 |
| fusion_per_channel_thresholds | 0.8774 | 0.6602 | 0.5906 | 0.4494 | 0.9407 |

Rule engine standalone (tuned): acc 0.8421, macro-F1 0.5258.

## Rule channel

- coverage_count: 1052
- coverage_pct: 15.75
- precision_on_firing_pct: 58.75
- recall_nonclean_pct: 54.59
- false_alarm_clean_pct: 7.82

## Corrections / regressions

- ai_correct: 5853
- fusion_correct: 5861
- fixed: 23
- broken: 15
- both_wrong: 804

Fixed transitions: CLEAN<-HATE=6, CLEAN<-OFFENSIVE=1, HATE<-CLEAN=9, HATE<-OFFENSIVE=7

Broken transitions: CLEAN->HATE=6, HATE->CLEAN=5, OFFENSIVE->HATE=4

## Correction provenance

- fixed_hate: 16
- fixed_hate_with_rule_fire: 16
- fixed_hate_with_rule_direction_ok: 16
- fixed_hate_rule_hate_median: 0.7652
- fixed_hate_rule_hate_mean: 0.7507
- fixed_clean: 7
- fixed_clean_with_rule_silent: 7
- fixed_clean_with_rule_fire_wrong_dir: 0
- rule_consistent_total: 23

## Opportunities / regressions

- ai_wrong_total: 827
- ai_wrong_nonclean_correct_rule_evidence: 148
- ai_wrong_nonclean_correct_rule_evidence_fixed: 16
- ai_wrong_clean_rule_silent: 151
- ai_wrong_clean_rule_silent_fixed: 7
- missed_corrections: 132
- broken_total: 15
- broken_with_rule_fire: 10
- broken_with_rule_fire_wrong_direction: 10
- broken_to_HATE: 10

## Cue taxonomy (HATE corrections, multi-label)

- n: 16
- teencode: 10
- explicit: 5
- derogatory: 8
- no_cue: 1
- multi_label: True

## Backbones

| model | alpha | AI macro-F1 | fusion macro-F1 | delta |
|---|---|---|---|---|
| sensitiveai-vi | 0.9 | 0.6545 | 0.6494 | -0.0051 |
| sensitiveai-vi-custom | 0.9 | 0.6575 | 0.6602 | +0.0027 |
| sensitiveai-vi-customdlr2stage | 0.9 | 0.6638 | 0.6637 | -0.0000 |
| sensitiveai-vi-custom-curriculum | 0.9 | 0.6590 | 0.6618 | +0.0028 |

## Significance

| model | delta(AI-fusion) | 95% CI | p_boot | b | c | p_McNemar |
|---|---|---|---|---|---|---|
| sensitiveai-vi | +0.0051 | [-0.0020, +0.0120] | 0.1592 | 61 | 30 | 0.0015 |
| sensitiveai-vi-custom | -0.0027 | [-0.0079, +0.0027] | 0.2942 | 15 | 23 | 0.2559 |
| sensitiveai-vi-customdlr2stage | +0.0000 | [-0.0020, +0.0024] | 0.9782 | 2 | 3 | 1.0000 |
| sensitiveai-vi-custom-curriculum | -0.0028 | [-0.0058, +0.0004] | 0.0868 | 8 | 31 | 0.0003 |
