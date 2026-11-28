# 06 Contribution and Novelty Report

## Defensible contribution
The strongest defensible contribution is methodological/analytical rather than aggregate accuracy:
- Frozen hybrid protocol with per-decision provenance.
- Rule-driven versus threshold-driven attribution.
- Identification of 132 missed corrections.
- Quantified rule channel behaviour and failure modes.

## Contribution verification matrix
| Claim | Support | Status |
|---|---|---|
| Hybrid improves macro-F1 | 0.6575 -> 0.6602, p=0.29 | Not statistically supported |
| Rule channel is precise and narrow | 15.75% coverage, 58.75% precision on firing | Supported |
| Corrections are rule-consistent | 16 HATE recoveries with rule firing in corrected direction; 7 CLEAN de-flagging with silent rule | Supported |
| Regressions localize failure mode | 10 rule-driven, 5 threshold-driven | Supported |
| Explainability by construction for lexical channel | Evidence record recomputes lexical contribution | Supported with scope limitation |
| Complete transformer explainability | Not claimed; transformer remains black box | Correctly scoped |

## Novelty caution
- Hybrid rule-transformer and explainable hate speech systems already exist.
- The novelty is the systematic provenance analysis, not a new architecture.
- Do not overstate the aggregate improvement.
