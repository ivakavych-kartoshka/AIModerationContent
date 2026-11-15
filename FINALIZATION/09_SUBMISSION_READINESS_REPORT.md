# 09 Submission Readiness Report

## Checklist
- Final PDF exists: YES; source and submission copy are synchronized (see page count).
- Page count: 15 in the rebuilt source (`FINALIZATION/source/iccies2027/main.pdf`), within
  the 12--15 page budget of ICCIES 2027.
  The submission copy (`FINALIZATION/submission/iccies2027_final.pdf`) was refreshed from this build and contains the leak-free / 3-seed / model-framing text.
- Build engine: XeLaTeX.
- Tests passed: 9/9.
- Preliminary multi-seed analysis: 3 seeds; mean delta -0.0008; not integrated into canonical manuscript numbers.
- Multi-seed preliminary results: 3 seeds; mean delta -0.0008.
- Reproducibility assertions: PASS.
- Figures regenerated: YES.
- Tables generated: YES.
- Manuscript prose changed by AI: YES, condensed under explicit author authorization to meet the page limit; all numbers, claims, and author metadata preserved.
- Critical issues remaining: none unresolved; see status list below.

## Blocking items and status
1. Citation [23] misuse: RESOLVED in corrected source copy (removed misuse and six unused references).
2. Six uncited references: RESOLVED in corrected source copy.
3. Test-focused primary model selection: RESOLVED in framing---the manuscript now states the focused backbone is a post-hoc, exploratory case study and reports the strongest standalone backbone separately.
4. Single-seed limitation: DOCUMENTED, with preliminary 3-seed sensitivity reported; it is not integrated into the canonical manuscript numbers.
5. Leakage mitigation: DOCUMENTED---leak-free rule engine and leak-free fusion are reported as a separate sensitivity analysis; canonical numbers remain the headline. Leak-free lexicon regeneration still lacks a committed filter script (reproducibility artifact only).

## Final status for full paper
READY FOR AUTHOR SUBMISSION CHECK.
Manuscript, numbers, references, and page budget are final; the author must confirm
metadata (corresponding authors, ORCID, affiliations) and ICCIES 2027 format/blinding
requirements before submission.

## Abstract/title-only submission
- Title and abstract are available in `FINALIZATION/submission/abstract_title_submission.md`.
- Full paper is ready to submit after the author-side checks above.

## Numeric sensitivity artifacts
- FINALIZATION/tables/leak_free_sensitivity_tabular.tex
- FINALIZATION/tables/multi_seed_preliminary_tabular.tex
- Source prose remains unchanged; author must place/approve these tables.
