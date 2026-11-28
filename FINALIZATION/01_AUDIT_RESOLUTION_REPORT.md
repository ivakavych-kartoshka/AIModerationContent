# 01 Audit Resolution Report

## Resolved
- Figure 1 alpha mismatch: regenerated `architecture.pdf` from frozen metrics; canonical values are alpha_H=0.2, alpha_O=0.9, alpha_C=1.0.
- Figure 2(b) discrepancy: traced to archived rule-silence run with 29/20/49; regenerated from canonical transition counts 23/15/38.
- README provenance line corrected to 0.2/0.9/1.0 and 0.14/0.66/0.60.
- Reproducibility assertions created and passing.
- Leakage-free lexicon created and compared.
- Canonical numbers regenerated.

## Partially resolved
- Cross-corpus leakage quantified; leak-free rule engine tuned and leak-free fusion integrated in separate outputs (outputs/evaluation/fusion_leakfree).
- Citation [23] misuse and six uncited references identified but not changed in manuscript prose.
- Test-focused primary model selection identified and documented; not changed.

## Unresolved / human actions
- Author approval for any prose or reference changes.
- Decide corresponding-author count.
- Primary model framing decision: keep sensitiveai-vi-custom as case study; report sensitiveai-vi-customdlr2stage as strongest standalone backbone; frame focused model as exploratory/post-hoc. (Author decision recorded.)
- Decide whether additional seeds/resources are available.
