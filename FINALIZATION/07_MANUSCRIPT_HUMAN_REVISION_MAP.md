# 07 Manuscript Human Revision Map

## Citation and references (human author action)
- Replace the scikit-learn metric-definition citation currently using `saito2015precision` with a proper scikit-learn reference or reword the sentence to cite the library directly.
- Remove or cite the six unused references: `cui2019classbalanced`, `howard2018ulmfit`, `macavaney2019hate`, `tran2022phobertcnn`, `tran2023videberta`, `vu2020hsdvlsp`.
- Verify DOI and publication metadata for all model and dataset references.

## Scientific claims (human author action)
- Reclassify the primary model selection as exploratory or select the primary model using validation-only criteria.
- State the single-seed limitation prominently in abstract/conclusion if additional seeds are not available.
- Document cross-corpus leakage and the leak-free lexicon comparison in Data/Lexicon or Discussion.
- Clarify that cue taxonomy annotation is post-hoc and unblinded.
- Add duplicate-group bootstrap or deduplicated sensitivity analysis if available.

## Metadata (human author action)
- Verify author names, affiliations, emails.
- Decide whether two corresponding authors are intended; the source marks both Cao Hong Nga and Jia-Ching Wang with `(*)` and now uses the plural note "are the corresponding authors", consistent with both markers.

## Prohibited for AI
- Do not insert AI-generated replacement academic paragraphs.
- Do not paraphrase existing prose.
- Use this map only as factual instructions for human authors.

## Primary model framing decision
- Keep `sensitiveai-vi-custom` as the primary case study for fusion provenance.
- Report `sensitiveai-vi-customdlr2stage` as the strongest standalone backbone by validation macro-F1 (0.7089) and test macro-F1 (0.6638).
- Frame the focused model as an exploratory/post-hoc case study, not a confirmatory primary-model selection.
- Human author must revise the corresponding prose in `sections/setup.tex`; AI did not generate replacement text.
