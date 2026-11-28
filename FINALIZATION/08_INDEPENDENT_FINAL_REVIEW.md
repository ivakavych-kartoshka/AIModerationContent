# 08 Independent Final Review

Reviewed against the current 15-page build (`FINALIZATION/source/iccies2027/main.pdf`),
the corrected artifacts, and the condensed source. Items from the earlier review are
updated below.

## Strengths
- Numerical claims reproduce exactly from frozen artifacts.
- Equations match implementation.
- Figures regenerated from canonical artifacts.
- Rule channel behaviour is quantified and auditable.
- Limitations and non-significance are reported honestly.
- Manuscript prose was condensed under explicit author authorization to meet the
  12--15 page limit; all numbers, claims, and author metadata were preserved.

## Resolved critical issues
- Citation misuse: RESOLVED (misuse removed; all remaining references are cited).
- Six uncited references: RESOLVED (reference list reduced to the cited entries).
- Primary model selected by test-set correction/regression balance: RESOLVED in
  framing---the paper states explicitly that the focused backbone is a post-hoc,
  exploratory case study rather than a validation-based selection, and reports
  `sensitiveai-vi-customdlr2stage` (0.7089 validation, 0.6638 test) as the strongest
  standalone backbone.
- Single seed per backbone: DOCUMENTED---the preliminary three-seed sensitivity
  (+0.0027, -0.0041, -0.0010; mean -0.0008) is reported as preliminary only and does
  not replace the canonical single-seed numbers.
- Cross-corpus leakage: DOCUMENTED AND RE-EVALUATED---895 overlapping test rows
  (620 distinct texts, 13.4%) are disclosed, and the 671-entry leak-free lexicon is
  reported as a separate sensitivity analysis without displacing the canonical results.

## Remaining major issues (disclosed in the manuscript)
- Duplicate-aware statistical analysis is incomplete: 109 duplicated test rows, no
  duplicate-group bootstrap.
- Cue taxonomy is post-hoc and unblinded.
- No human evaluation or inter-annotator agreement.
- Leak-free lexicon regeneration still lacks a committed filter script.

## Minor issues
- Two authors are marked with `(*)`; the note now reads "are the corresponding
  authors", consistent with both markers. The author must still confirm whether one or
  two corresponding authors are intended.
- Page budget is tight: exactly 15 pages, with no spare space.
- The political false-positive case from a lexicon entry is disclosed and discussed.

## Recommendation
Submission-ready from the manuscript and reproducibility standpoint, subject to the
author's confirmation of metadata (corresponding authors, ORCID, affiliations) and of
the ICCIES 2027 format/blinding requirements. The remaining items are disclosed
limitations, not undisclosed defects.
