# Paragraphs to rewrite by hand (AI-condensed -> your own voice)

Only paragraphs that changed substantially (similarity < 0.90) are listed.
Paragraphs with similarity >= 0.90 are trivial edits (word swaps) and are NOT listed - low AI risk, can keep as-is.

## Priority tiers
- **P1 (sim < 0.70)**: heavy condensation - rewrite fully by hand first.
- **P2 (0.70-0.79)**: moderate condensation - rewrite or at least rephrase.
- **P3 (0.80-0.89)**: light edits - small rephrasing is enough.

## Time budget (2 days)
- Day 1: P1 (12 paragraphs) fully by hand.
- Day 2 morning: P2 (11) + P3 (13), then rebuild + verify page count.
- Day 2 afternoon: upload to CMT before 14:59.

## Index
- [0.26] discussion.tex | \paragraph{Cross-corpus overlap.}
- [0.32] discussion.tex | Four changes follow: (i) \textbf{evidence-gated fusion}---abstain or esc
- [0.53] conclusion.tex | The leak-free sensitivity run (671 entries) yields fusion macro-F1 0.661
- [0.54] discussion.tex | \paragraph{Preliminary three-seed check.}
- [0.55] method.tex | For every comment the system emits, with the label, an evidence record: 
- [0.62] results.tex | We retain \texttt{sensitiveai-vi-custom} as a focused, exploratory case 
- [0.63] setup.tex | The split is hashed and re-checked, the rule-engine evaluation is re-rea
- [0.64] related_work.tex | \paragraph{Explainability and hybrid systems.}
- [0.64] results.tex | The rule engine fires on 1{,}052 of 6{,}680 comments (15.75\%) at 58.75\
- [0.64] analysis.tex | The 15 regressions split into 10 rule-driven and 5 threshold-driven case
- [0.66] setup.tex | Table~\ref{tab:backbones} lists the four backbones, all using
- [0.68] discussion.tex | In a binary ``remove / keep'' framing the asymmetry of Sect.~\ref{sec:re
- [0.71] analysis.tex | Table~\ref{tab:cases} shows five decisions. Case~1 is the rule rescue: t
- [0.72] results.tex | With the 671-entry leak-free lexicon, the standalone rule engine reaches
- [0.73] method.tex | \begin{figure}[ht]
- [0.73] discussion.tex | \emph{Triage and accountability.} A queue can sort comments by whether t
- [0.74] data.tex | Each entry stores the surface term and canonical form, a label (\hate{} 
- [0.75] data.tex | An exact-text cross-corpus audit nevertheless found 895 of the 6{,}680 V
- [0.76] method.tex | \paragraph{Normalisation and views.}
- [0.77] discussion.tex | Comments are public ViHSD test comments, masked as described in Sect.~\r
- [0.77] data.tex | After cleaning, the corpus contains 33{,}398 comments. Every system---ru
- [0.78] data.tex | The lexicon is deliberately not a list of slurs. Most entries---551 of 6
- [0.78] analysis.tex | All 7 \clean{} de-flagging cases have a silent rule ($r_H=r_O=0$, so
- [0.82] results.tex | \begin{figure}[ht]
- [0.84] discussion.tex | First, \emph{coverage is low}: the rule fires on 15.75\% of comments, so
- [0.84] data.tex | The rule engine is driven by a Vietnamese hate-speech lexicon of \textbf
- [0.85] results.tex | Table~\ref{tab:fusion-variants} reports every variant and Fig.~\ref{fig:
- [0.86] results.tex | Per-channel weights add 0.0010 macro-F1 over a single global weight (0.6
- [0.87] discussion.tex | The result is better read as a successful audit: 38 of 6{,}680 decisions
- [0.87] analysis.tex | The 132 are not missed for weak evidence: their median rule \hate{} scor
- [0.87] introduction.tex | The contributions are: (i) a frozen, reproducible hybrid protocol in whi
- [0.88] introduction.tex | The release of UIT-ViHSD \cite{luu2021vihsd}, roughly 33k comments label
- [0.88] introduction.tex | We ask a narrow question: \emph{can a lexicon-based rule engine be attac
- [0.88] analysis.tex | The 5 threshold-driven regressions are the mirror image: the rule is sil
- [0.89] discussion.tex | \emph{Editing and diagnosis.} Adding a slur variant is a configuration c
- [0.90] analysis.tex | We annotated the 16 \hate{} recoveries with a multi-label cue taxonomy: 

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: discussion.tex | sim=0.26
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\paragraph{Cross-corpus overlap.}
The audit in Sect.~\ref{sec:lexicon} found 895 overlapping test rows and motivated the
671-entry leak-free lexicon re-evaluated in Sect.~\ref{sec:leak-free-sensitivity};
this bounds, but does not eliminate, cross-corpus dependence.
`
### ORIGINAL (pre-condensation, for reference)
`
\paragraph{Cross-corpus overlap.}
The exact-text audit found 895 of 6{,}680 ViHSD test rows (13.4\%) with matches
in the auxiliary corpus material examined for lexicon development; 620 distinct
test texts also occur in the Bontori/R-ViHSD training partition. This overlap
creates a potential source of lexical leakage for the canonical lexicon.
Rebuilding the lexicon after excluding the overlapping Bontori/R-ViHSD training
texts produced a 671-entry leak-free lexicon. The resulting standalone
rule-engine and fusion scores are reported separately in
Section~\ref{sec:leak-free-sensitivity}. This sensitivity analysis addresses
the identified overlap but does not establish the absence of all possible
cross-corpus dependencies.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: discussion.tex | sim=0.32
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
Four changes follow: (i) \textbf{evidence-gated fusion}---abstain or escalate when
the rule fires strongly against a confident \clean{}; (ii) \textbf{separate thresholds
per mechanism}, requiring $r_H\ge0.5$ before the \hate{} channel can override, making
the rule- and threshold-driven paths separately tunable; (iii) \textbf{per-class
calibration} of the \clean{} channel, which now blocks the missed corrections; (iv)
\textbf{human-in-the-loop} for the 804 both-wrong comments no fusion can fix.
`
### ORIGINAL (pre-condensation, for reference)
`
Four changes follow. (i) \textbf{Evidence-gated fusion}: when the rule fires
strongly against a high-confidence \clean{} prediction, abstain or escalate.
(ii) \textbf{Separate thresholds per mechanism}: requiring $r_H\ge0.5$ before
the \hate{} channel can override would make the rule-driven and
threshold-driven paths separately tunable. (iii) \textbf{Per-class calibration}
of the \clean{} channel, which currently blocks the missed corrections. (iv)
\textbf{Human-in-the-loop} for the 804 both-wrong comments, which no fusion of
these two channels can fix.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: conclusion.tex | sim=0.53
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The leak-free sensitivity run (671 entries) yields fusion macro-F1 0.6612 and
accuracy 0.8777 (23 corrections, 13 regressions); the standalone engine reaches
0.5328 and 0.8440. A preliminary three-seed check gives fusion-minus-AI
differences of $+0.0027$, $-0.0041$ and $-0.0010$ (mean $-0.0008$), so the
single-seed gain is not stable. The broader lesson is methodological: a hybrid
judged only by its aggregate metric looks null, whereas the same hybrid
decomposed by channel and direction yields an auditable procedure and a design
agenda. Future work: validate evidence-gated fusion, separate profanity from
political entries, restore negation and accent-less matching, and measure
whether the evidence record helps reviewers.
`
### ORIGINAL (pre-condensation, for reference)
`
The broader lesson is methodological: a hybrid evaluated only by its aggregate
metric looks like a null result, whereas the same hybrid decomposed by channel
and direction yields an auditable decision procedure and a design agenda.
Future work: validate evidence-gated fusion, separate profanity from political
entries, restore negation and accent-less matching, and measure whether the
evidence record helps reviewers.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: discussion.tex | sim=0.54
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\paragraph{Preliminary three-seed check.}
The main sweep used seed 42. A preliminary three-seed check gives fusion-minus-AI
macro-F1 differences of $+0.0027$, $-0.0041$ and $-0.0010$ (mean $-0.0008$): the sign
changes, so the single-seed gain is not stable and this does not replace a full
multi-seed evaluation.
`
### ORIGINAL (pre-condensation, for reference)
`
\paragraph{Preliminary three-seed check.}
The main backbone sweep was run with seed 42. A preliminary three-seed check of
the focused fusion comparison produced macro-F1 differences (fusion minus
AI-only) of $+0.0027$, $-0.0041$ and $-0.0010$, for a mean difference of
$-0.0008$. The sign changes across runs, and the mean is slightly negative; the
single-seed gain is therefore not stable across this preliminary check. These
runs do not replace a full multi-seed evaluation of every backbone and fusion
variant.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: method.tex | sim=0.55
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
For every comment the system emits, with the label, an evidence record: matched
raw-text spans, each span's term and label, matching view and match type, per-match
score, aggregated rule scores, per-channel weights and the three fused scores. Given
the record and the transformer probabilities, a reviewer can recompute the decision
exactly---our meaning of \emph{faithful by construction} for the lexical channel, while
the neural contribution remains a black box.
`
### ORIGINAL (pre-condensation, for reference)
`
For every comment the system emits, with the label, an evidence record: matched
spans in the raw text, dictionary term and label of each span, matching view and
match type, per-match score, aggregated rule scores, per-channel weights and the
three fused scores. Given the record and the transformer probabilities, a reviewer
can recompute the decision exactly, which is what we mean by \emph{faithful by
construction} for the lexical channel; the neural contribution remains a black box,
and we do not claim otherwise. Every tunable quantity is frozen before the single test
pass, and the error analysis is descriptive and performed after it.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: results.tex | sim=0.62
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
We retain \texttt{sensitiveai-vi-custom} as a focused, exploratory case study for
tracing decision provenance---a post-hoc analysis, not a confirmatory validation-based
selection. The strongest standalone backbone by validation macro-F1 is
\texttt{sensitiveai-vi-customdlr2stage} (0.7089), also strongest on test (0.6638), so
the fusion results are not evidence that the focused backbone is best standalone.
`
### ORIGINAL (pre-condensation, for reference)
`
The fusion analysis retains \texttt{sensitiveai-vi-custom} as a focused,
exploratory case study for tracing decision provenance. This is a post-hoc analysis,
not a confirmatory primary-model selection based on validation-only criteria. The
strongest standalone backbone by validation macro-F1 is
\texttt{sensitiveai-vi-customdlr2stage} (0.7089), which also obtains the strongest
test macro-F1 (0.6638). Accordingly, the fusion results below should not be read as
showing that the focused backbone is the best-performing standalone model.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: setup.tex | sim=0.63
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The split is hashed and re-checked, the rule-engine evaluation is re-read rather than
recomputed, and the analysis script asserts that the rule scores shipped with the fusion
predictions match the frozen tuned output---an earlier version read the \emph{untuned}
scores and over-reported rule coverage threefold.
`
### ORIGINAL (pre-condensation, for reference)
`
The split is hashed and re-checked, the rule-engine evaluation directory is recorded in
the fusion metrics and re-read rather than recomputed, and the analysis script asserts
that the rule scores shipped with the fusion predictions match the frozen tuned output.
This is not cosmetic: an earlier version silently read the \emph{untuned} scores and
over-reported rule coverage by a factor of three.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: related_work.tex | sim=0.64
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\paragraph{Explainability and hybrid systems.}
HateXplain \cite{mathew2021hatexplain} made human rationales and explanation quality
benchmark quantities. LIME \cite{ribeiro2016why} and SHAP \cite{lundberg2017shap} are
model-agnostic but do not guarantee that highlighted tokens are the ones the decision
used. Our rule channel is part of the decision function, so its matched spans are
\emph{faithful by construction} for the lexical component, though the neural component
remains opaque. We settle the accuracy question with a paired test, analyse provenance,
and report the rule channel's own error profile and the direction of every
rule-induced change; to our knowledge, this audit is new for Vietnamese hate speech.
`
### ORIGINAL (pre-condensation, for reference)
`
\paragraph{Explainability and hybrid systems.}
HateXplain \cite{mathew2021hatexplain} made human rationales and explanation quality
benchmark quantities. LIME \cite{ribeiro2016why} and SHAP
\cite{lundberg2017shap} are model-agnostic but do not guarantee that the highlighted
tokens are the ones the decision used. Our rule channel is part of the decision
function, so its matched spans are \emph{faithful by construction} for the lexical
component, while the neural component remains opaque. We differ in two ways: we fix
the accuracy question with a paired test and analyse provenance instead, and we
report the rule channel's own error profile and the direction of every rule-induced
change. To our knowledge, this per-decision provenance audit has not been reported
for Vietnamese hate speech.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: results.tex | sim=0.64
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The rule engine fires on 1{,}052 of 6{,}680 comments (15.75\%) at 58.75\% precision,
recovers 54.59\% of non-\clean{} comments and fires on 7.82\% of \clean{} ones
(Fig.~\ref{fig:results}c): useful as evidence, not as a decision, being silent on 45\%
of toxic comments and wrong about much of what it flags. Coverage is low by
construction---most entries are multi-word, context-required phrases and validation
zeroed the accent-less alias weight---so it matches near-canonical forms, not
respellings.
`
### ORIGINAL (pre-condensation, for reference)
`
The rule engine fires on 1{,}052 of 6{,}680 comments (15.75\%), not on the majority
of the test set. When it fires, 58.75\% of the comments are truly non-\clean{}, so
roughly two in five firings are false alarms; it recovers 54.59\% of non-\clean{}
comments and fires on 7.82\% of \clean{} comments (Fig.~\ref{fig:results}c). It is
useful as evidence, not as a decision: silent on 45\% of toxic comments and wrong
about a substantial share of those it flags. Coverage is low by construction---most
entries are multi-word, context-required phrases and validation zeroed the
accent-less alias weight---so the channel matches near-canonical forms rather than
respellings: narrow and inspectable rather than broad and opaque.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: analysis.tex | sim=0.64
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The 15 regressions split into 10 rule-driven and 5 threshold-driven cases, aligning
perfectly with direction (Fig.~\ref{fig:results}b). In all 10 rule-driven cases (four
\off{}$\to$\hate{}, six \clean{}$\to$\hate{}) the rule fires towards \hate{}, so fusion
trusts lexical evidence over the neural label and the lexicon is wrong about
intensity or target. The \off{} cases are the expected failure of a phrase lexicon:
profanity used as an intensifier, not an attack. The \clean{} cases are false
positives on non-abusive text; one is a political comment whose charged multi-word
entry fires at $r_H=0.62$ while the transformer prefers \clean{} ($p_C=0.515$ vs.\
$p_H=0.473$), but the \clean{} channel misses $\tau_C=0.60$ and the rule lifts $f_H$
to $0.588$, far above $\tau_H=0.14$, so the decision goes to \hate{}. A lexicon
entry's \emph{political} sensitivity can thus leak into moderation decisions
(Sect.~\ref{sec:discussion}).
`
### ORIGINAL (pre-condensation, for reference)
`
The 15 regressions split into 10 rule-driven and 5 threshold-driven cases, and the
split aligns perfectly with direction (Fig.~\ref{fig:results}b). In all 10 rule-driven
cases (four
\off{}$\to$\hate{}, six \clean{}$\to$\hate{}) the rule fires in the \hate{}
direction, so fusion trusts lexical \hate{} evidence over the neural label and the
lexicon is wrong about intensity or target. The \off{} cases are the expected failure
of a phrase lexicon: profanity used as an intensifier rather than an attack. The
\clean{} cases are false positives on non-abusive text. One is a political comment
whose lexicon entry is a charged multi-word expression: the rule fires at $r_H=0.62$,
the transformer prefers \clean{} ($p_C=0.515$ against $p_H=0.473$), but the \clean{}
channel misses $\tau_C=0.60$ while the rule lifts $f_H$ to $0.588$, far above
$\tau_H=0.14$, so the decision goes to \hate{}. A lexicon entry's \emph{political}
sensitivity can therefore leak into moderation decisions
(Sect.~\ref{sec:discussion}).
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: setup.tex | sim=0.66
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
Table~\ref{tab:backbones} lists the four backbones, all using
\texttt{microsoft/mdeberta-v3-base} with the same tokenizer, split and evaluation
code and differing only in head and optimisation recipe. The focused model,
\model{sensitiveai-vi-custom}, gives the clearest positive net effect (23 corrections,
15 regressions) and is thus the clearest subject for studying what fusion changes.
`
### ORIGINAL (pre-condensation, for reference)
`
Table~\ref{tab:backbones} lists the four backbones. All use
\texttt{microsoft/mdeberta-v3-base}, the same tokenizer, split and evaluation code,
and differ only in the head and optimisation recipe. The focused model,
\model{sensitiveai-vi-custom}, is chosen because it gives the clearest positive net
effect (23 corrections against 15 regressions), not because it has the highest
standalone score: the strongest backbone, \model{sensitiveai-vi-dlr2stage}, reaches
0.6638 macro-F1 but is almost indifferent to fusion (3 corrections, 2 regressions),
which makes it a poor subject for a study of what fusion changes.
`

==============================================================================
PRIORITY: P1 (heavy rewrite - highest AI risk) | FILE: discussion.tex | sim=0.68
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
In a binary ``remove / keep'' framing the asymmetry of Sect.~\ref{sec:regressions} is
a recall-oriented trade-off; in a three-class framing it is not, because an \off{}
comment mislabelled as \hate{} can trigger a stronger sanction than warranted, and a
political comment mislabelled as \hate{} is a different harm. That entry is a charged
multi-word expression labelled \hate{} from co-occurrence, so the lexicon is not
neutral---it encodes editorial judgments entering at weight 0.8. Deployments should
separate \emph{profanity} from \emph{political} entries and weight them accordingly.
`
### ORIGINAL (pre-condensation, for reference)
`
In a binary ``remove / keep'' framing the asymmetry of
Sect.~\ref{sec:regressions} is a recall-oriented trade-off; in a three-class
framing it is not, because an \off{} comment mislabelled as \hate{} can trigger
a stronger sanction than the content warrants, and a political comment
mislabelled as \hate{} is a different kind of harm.
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: analysis.tex | sim=0.71
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
Table~\ref{tab:cases} shows five decisions. Case~1 is the rule rescue: two exact
matches carry \hate{} from $p_H{=}0.19$ to $f_H{=}0.81$, past the 0.78 \clean{}
channel. Case~2 is the one correction plain argmax would not make: \hate{} is 0.01
\emph{below} \off{}, but \off{} misses $\tau_O{=}0.66$ while \hate{} clears
$\tau_H{=}0.14$---the 0.0009 per-channel thresholds buy. Case~3 is a de-flagging: a
zero-weight accent-less alias leaves the rule silent, the \hate{} channel collapses to
$0.2\,p_H$, and no channel clears its threshold, so the fallback returns \clean{}.
Cases~4--5 are the failure modes of Sect.~\ref{sec:regressions}: an over-firing rule
on profanity used as an interjection, and a \hate{} channel that misses $\tau_H$ by
0.001.
`
### ORIGINAL (pre-condensation, for reference)
`
Table~\ref{tab:cases} shows five decisions covering these mechanisms. Case~1 is the
canonical rule rescue: two exact matches carry \hate{} from $p_H{=}0.19$ to
$f_H{=}0.81$, past the 0.78 \clean{} channel the AI is confident in. Case~2 is the one
correction in the five that the plain argmax would not make---\hate{} is 0.01 \emph{below}
\off{}, but \off{} falls short of its own $\tau_O{=}0.66$ while \hate{} clears
$\tau_H{=}0.14$, which is the 0.0009 that per-channel thresholds buy
(Table~\ref{tab:fusion-variants}). Case~3 is a de-flagging: a zero-weight accent-less alias
leaves the rule silent, the \hate{} channel collapses to $0.2\,p_H$, no channel clears
its threshold and the fallback returns \clean{}. Cases~4 and 5 are the two failure
modes of Sect.~\ref{sec:regressions}: an over-firing rule on profanity used as an
interjection, and a \hate{} channel that misses $\tau_H$ by exactly 0.001 and de-flags
a comment the AI got right.
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: results.tex | sim=0.72
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
With the 671-entry leak-free lexicon, the standalone rule engine reaches 0.5398
validation and 0.5328 test macro-F1 (accuracy 0.8440), with thresholds \hate{} 0.76,
\off{} 0.00 and \clean{} 0.64. Leak-free fusion obtains 5{,}863 correct test
predictions (accuracy 0.8777), 23 corrections and 13 regressions (36 changes),
macro-F1 0.6612; the canonical 668-entry setup gives 5{,}861 correct, 23 corrections
and 15 regressions (38 changes), macro-F1 0.6602. This is a separate sensitivity
result, not the headline; no paired-bootstrap test is available.
`
### ORIGINAL (pre-condensation, for reference)
`
With the 671-entry leak-free lexicon, the standalone rule engine reaches 0.5398
validation macro-F1 and 0.5328 test macro-F1, with test accuracy 0.8440. Its selected
thresholds are \hate{} 0.76, \off{} 0.00 and \clean{} 0.64. The corresponding
leak-free fusion obtains 5{,}863 correct test predictions (accuracy 0.8777),
correcting 23 cases and regressing on 13 (36 changed decisions in total), with
macro-F1 0.6612. For comparison, the canonical 668-entry setup obtains 5{,}861
correct predictions, 23 corrections and 15 regressions (38 changed decisions),
with macro-F1 0.6602. These sensitivity results are reported separately from the
canonical headline results; no separate paired-bootstrap test is available for the
leak-free run.
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: method.tex | sim=0.73
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\begin{figure}[ht]
\centering
\includegraphics[width=0.80\textwidth]{figures/architecture.pdf}
\caption{The hybrid. A comment enters two branches: the neural branch produces class
probabilities, the rule branch per-label lexical scores and the spans that produced
them. Fusion mixes each channel with its own weight $\alpha_c$, and frozen per-channel
thresholds turn the three scores into a label; the evidence record is emitted with the
decision. Weights and thresholds shown were selected on validation for the focused
backbone.}
\label{fig:architecture}
\end{figure}
`
### ORIGINAL (pre-condensation, for reference)
`
\begin{figure}[ht]
\centering
\includegraphics[width=0.92\textwidth]{figures/architecture.pdf}
\caption{The hybrid. A comment enters two independent branches. The neural branch
produces class probabilities; the rule branch produces per-label lexical scores and
the spans that produced them. Fusion mixes each channel with its own weight
$\alpha_c$, and frozen per-channel thresholds turn the three scores into a label.
The evidence record is emitted with the decision and makes the lexical part of the
decision inspectable. Weights and thresholds shown are the values selected on
validation for the focused backbone.}
\label{fig:architecture}
\end{figure}
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: discussion.tex | sim=0.73
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\emph{Triage and accountability.} A queue can sort comments by whether the decision
was rule-supported: a \hate{} decision with a matched span and $r_H=0.8$ is easier to
review and defend.
`
### ORIGINAL (pre-condensation, for reference)
`
\emph{Triage and accountability.} A queue can sort comments by whether the decision
was rule-supported: a \hate{} decision with a matched span and $r_H=0.8$ is easier
to review than one with no lexical evidence, and the span, the entry and the
arithmetic are what a platform can defend in policy terms.
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: data.tex | sim=0.74
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
Each entry stores the surface term and canonical form, a label (\hate{} or \off{}) with
per-label severity, a training-estimated confidence, a context-required flag, and a
review status (\emph{verified} or \emph{auto}), retained in the evidence record.
`
### ORIGINAL (pre-condensation, for reference)
`
Each entry stores the surface term and a canonical form, a lexical label
(\hate{} or \off{}) with a per-label severity, an entry confidence estimated
from the training data, a context-required flag, and a review status
(\emph{verified} or \emph{auto}); both statuses may match, but the flag is
retained in the evidence record.
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: data.tex | sim=0.75
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
An exact-text cross-corpus audit nevertheless found 895 of the 6{,}680 ViHSD test rows
(13.4\%) with a matching text in the auxiliary corpus material examined for lexicon
development, corresponding to 620 distinct test texts also present in the Bontori/R-ViHSD
training partition. This is a potential source of lexicon-construction leakage even
though ViHSD test comments were not deliberately sampled. To assess its effect, we
rebuilt the lexicon after excluding the overlapping Bontori/R-ViHSD training texts; the
leak-free lexicon has 671 entries versus 668 canonical. Unless stated otherwise, main
results use the canonical 668-entry lexicon; leak-free results are in
Sect.~\ref{sec:leak-free-sensitivity}.
`
### ORIGINAL (pre-condensation, for reference)
`
An exact-text cross-corpus audit nevertheless found that 895 of the 6{,}680
ViHSD test rows (13.4\%) had a matching text in the auxiliary corpus material
examined for lexicon development. These matches correspond to 620 distinct test
texts that also occur in the Bontori/R-ViHSD training partition. This is a
potential source of lexicon-construction leakage even though ViHSD test
comments were not deliberately sampled as lexicon-training examples. To assess
its effect, we rebuilt the lexicon after excluding the Bontori/R-ViHSD training
texts that overlap with the ViHSD test set. The resulting leak-free lexicon
contains 671 entries, compared with 668 in the canonical lexicon. Unless stated
otherwise, the main results refer to the canonical 668-entry lexicon; the
leak-free sensitivity results are reported in
Section~\ref{sec:leak-free-sensitivity}.
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: method.tex | sim=0.76
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\paragraph{Normalisation and views.}
The comment is Unicode-normalised to NFC, invisible characters are removed,
whitespace is collapsed, character runs are shortened to at most three, and the text
is lower-cased while keeping Vietnamese diacritics. An \emph{offset map} back to the
raw comment lets a cleaned-view match be reported as an original span. Auxiliary views
handle the writing tricks that matter: accent-less, leet, repeat-collapsed, and a
compact view with separators removed (\texttt{n g u}). Each hit records the term,
label, view, raw span and a \emph{match type} (e.g.\ \emph{exact},
\emph{accentless\_alias}, \emph{leet}, \emph{separator\_inserted}, \emph{fuzzy}).
`
### ORIGINAL (pre-condensation, for reference)
`
\paragraph{Normalisation and views.}
The comment is Unicode-normalised to NFC, invisible characters are removed,
whitespace is collapsed, character runs are shortened to at most three, and the text
is lower-cased while keeping Vietnamese diacritics. Normalisation returns an
\emph{offset map} to the raw comment, so a match on a cleaned view is always reported
as a span of the original text. Auxiliary views handle the writing tricks that matter:
accent-less, leet, repeat-collapsed, and a compact view with separators removed
(\texttt{n g u}). The matcher records, for every hit, the dictionary term, label, view,
raw span and a \emph{match type} (\emph{exact}, \emph{accentless\_alias}, \emph{leet},
\emph{repeat\_collapsed}, \emph{separator\_inserted}, \emph{obfuscated\_combined} or
\emph{fuzzy}).
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: discussion.tex | sim=0.77
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
Comments are public ViHSD test comments, masked as described in Sect.~\ref{sec:data}.
The system is decision support, not automatic sanctioning: the directional asymmetry
and the political false positive show that automatic \hate{} labels from this pipeline
should be reviewed before affecting a user, and the lexicon should be a governed
artefact with a change log since its entries shape moderation outcomes.
`
### ORIGINAL (pre-condensation, for reference)
`
The comments are public ViHSD test comments and are masked as described in
Sect.~\ref{sec:data}. The system is decision support, not automatic
sanctioning: the directional asymmetry and the political false positive show
that automatic \hate{} labels from this pipeline should be reviewed before they
affect a user. The lexicon should be treated as a governed artefact with a
change log, because its entries shape moderation outcomes.
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: data.tex | sim=0.77
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
After cleaning, the corpus contains 33{,}398 comments. Every system---rule engine,
backbone, fusion variant, analysis script---reads the same fixed split, and each run
stores and re-checks the split's SHA-256 digest, ruling out a silently re-shuffled
split. Training has 24{,}046 comments (2{,}556 \hate{}, 1{,}605 \off{}, 19{,}885
\clean{}), validation 2{,}672 (270/212/2{,}190) and test 6{,}680 (688/444/5{,}548).
`
### ORIGINAL (pre-condensation, for reference)
`
After cleaning, the corpus contains 33{,}398 comments. We use the fixed split
shipped with the project, and every system---rule engine, backbone, fusion
variant and analysis script---reads the same files. Each run directory stores
the SHA-256 digest of the split it used and re-checks it before training,
ruling out a silently re-shuffled split. Training contains 24{,}046 comments
    (2{,}556 \hate{}, 1{,}605 \off{}, 19{,}885 \clean{}), validation 2{,}672
(270/212/2{,}190) and test 6{,}680 (688/444/5{,}548).
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: data.tex | sim=0.78
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The lexicon is deliberately not a list of slurs. Most entries---551 of 668 (82.5\%)---are
multi-word constructions such as \emph{mả cha}, \emph{con mẹ} or \emph{ngu hơn}, and 524
(78.4\%) are context-required, because Vietnamese insults often work by comparison and
implication rather than a single taboo word. Only 74 entries carry an \off{} label against
594 for \hate{}, so the engine is weak on \off{} and fusion keeps that channel mostly
neural. The lexicon is \emph{editable} (a new variant needs no retraining),
\emph{deterministic} (same comment, same spans) and \emph{inspectable} (each score
decomposes into a term, match type and context factor).
`
### ORIGINAL (pre-condensation, for reference)
`
The lexicon is deliberately not a list of slurs. Most entries---551 of 668
(82.5\%)---are multi-word constructions such as \emph{mả cha}, \emph{con mẹ} or
\emph{ngu hơn}, and 524 (78.4\%) are context-required, because Vietnamese
insults frequently work by comparison and implication rather than by a single
taboo word. Only 74 entries carry an \off{} lexical label against 594 for
\hate{}, which is why the rule engine is weak on \off{} and fusion keeps that
channel mostly neural. Three properties make the lexicon worth keeping: it is
\emph{editable} (a new variant needs no retraining), \emph{deterministic} (the
same comment yields the same spans) and \emph{inspectable} (each score
decomposes into a term, a match type and a context factor).
`

==============================================================================
PRIORITY: P2 (moderate rewrite) | FILE: analysis.tex | sim=0.78
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
All 7 \clean{} de-flagging cases have a silent rule ($r_H=r_O=0$, so
$\tilde{\mathbf r}=(0,0,1)$). With $\alpha_H=0.2$ the rule term vanishes from the
\hate{} channel and $f_H=0.2\,p_H$, so $p_H\ge0.70$ is needed merely to reach
$\tau_H=0.14$; a flagged \hate{} at $p_H\approx0.5$ ends at $f_H\approx0.10$ and falls
back to \clean{}, whose channel keeps the full neural weight ($\alpha_C=1$). Silence
therefore acts only as a multiplicative prior on \hate{}. The alternative
convention---a silent rule contributes nothing and neural scores pass through---changes
30 decisions (10 for it, 14 against) and raises macro-F1 from 0.6602 to 0.6614
(95\% CI $[-0.0015,+0.0042]$, $p=0.44$; McNemar 14--10, $p=0.54$); it is undecided,
so we keep the shipped convention for its net corrections ($+8$ vs.\ $+4$).
`
### ORIGINAL (pre-condensation, for reference)
`
All 7 \clean{} de-flagging cases have a silent rule ($r_H=r_O=0$, so
$\tilde{\mathbf r}=(0,0,1)$). With $\alpha_H=0.2$ the rule term disappears from the
\hate{} channel and $f_H=0.2\,p_H$, so a comment needs $p_H\ge0.70$ merely to reach
$\tau_H=0.14$; one flagged \hate{} at $p_H\approx0.5$ ends at $f_H\approx0.10$ and the
decision falls back to \clean{}, whose channel keeps the full neural weight
($\alpha_C=1$). The rule never touches the \clean{} channel, so silence acts only as a
multiplicative prior on \hate{}. We tested the alternative convention, where a silent
rule contributes nothing and the neural scores pass through unchanged: it changes 30
decisions (10 for it, 14 against) and raises macro-F1 from 0.6602 to 0.6614
(95\% CI $[-0.0015,+0.0042]$, $p=0.44$; McNemar 14--10, $p=0.54$). It is undecided,
so we keep the shipped convention for its net corrections ($+8$ against $+4$).
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: results.tex | sim=0.82
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\begin{figure}[ht]
\centering
\includegraphics[width=0.85\textwidth]{figures/results.pdf}
\caption{Quantitative summary for the focused backbone. (a)\ Macro-F1 of every
decision variant; the dotted line is AI-only, the green bar the per-channel hybrid.
(b)\ Provenance of the 38 changed decisions, by correction and regression type.
(c)\ Behaviour of the rule channel on test: coverage, precision when firing, recall on
non-\clean{} comments, and false alarms on \clean{}; values are percentages, raw
counts are in the text.}
\label{fig:results}
\end{figure}
`
### ORIGINAL (pre-condensation, for reference)
`
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{figures/results.pdf}
\caption{Quantitative summary for the focused backbone. (a)\ Macro-F1 of every
decision variant; the dotted line is AI-only and the green bar is the per-channel
fusion variant. (b)\ Provenance of the 38 changed decisions in the canonical setup:
the four correction types and the three regression types. (c)\ Behaviour of the
rule channel on the test split: coverage, precision when firing, recall on non-\clean{}
comments, and false alarms on \clean{}. Values are percentages; raw counts are in
the text and in panel~(b).}
\label{fig:results}
\end{figure}
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: discussion.tex | sim=0.84
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
First, \emph{coverage is low}: the rule fires on 15.75\% of comments, so a perfect
rule channel could only change decisions in that subset. Second, \emph{the majority
class dominates}: with 83\% \clean{}, only the 1{,}132 toxic comments can improve, and
38 changes is 3.4\% of that support---enough to matter operationally, too little to
move a macro-average beyond its interval. Third, \emph{tuning is safety-oriented}:
protecting \clean{} and gating \hate{} on lexical support produces both the
de-flagging and the missed corrections. Neutrality is a \emph{choice}, not an absence
of signal; the leak-free run shows the same pattern.
`
### ORIGINAL (pre-condensation, for reference)
`
First, \emph{coverage is low}: the rule fires on 15.75\% of comments, so a
perfect rule channel could still only change decisions in that subset. Second,
\emph{the majority class dominates}: with 83\% \clean{}, only the 1{,}132 toxic
comments can improve, and 38 changes is 3.4\% of that support---enough to
matter operationally, too little to move a macro-average beyond its confidence
interval. Third, \emph{tuning is safety-oriented}: protecting \clean{} and
gating \hate{} on lexical support produces both the de-flagging and the missed
corrections. Neutrality is a \emph{choice}, not an absence of signal. The
leak-free sensitivity run shows a similar decision pattern, with 36 changes (23
corrections and 13 regressions) and macro-F1 0.6612; it is reported as a
separate sensitivity result rather than merged with the canonical analysis.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: data.tex | sim=0.84
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The rule engine is driven by a Vietnamese hate-speech lexicon of \textbf{668 entries},
mined and labelled from the \emph{training} portions of ViHSD; R-ViHSD, a cleaned
release adding TDTU YouTube and voz-forum comments \cite{bontoriTdtu}; and the code-mixed
ViCM corpus \cite{nguyen2026vicm}. The three training splits yield 1{,}062 candidate
terms, first attested 257/665/140 in ViHSD/ViCM/R-ViHSD; labelling reduces them to the
entries kept below. No evaluation split is intentionally sampled, and each entry records
its supporting corpora (414 of 668 occur in all three).
`
### ORIGINAL (pre-condensation, for reference)
`
The rule engine is driven by a Vietnamese hate-speech lexicon of \textbf{668
    entries}, mined and labelled from the \emph{training} portions of three
corpora: ViHSD; R-ViHSD, a cleaned release that adds TDTU YouTube and voz-forum
comments to ViHSD \cite{bontoriTdtu}; and the code-mixed ViCM corpus
\cite{nguyen2026vicm}. The three training splits yield 1{,}062 candidate terms,
of which 257, 665 and 140 are first attested in ViHSD, ViCM and R-ViHSD;
labelling reduces them to the entries kept below. No evaluation split is
intentionally sampled for dictionary construction, and each entry records which
corpora supported it (414 of the 668 occur in all three).
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: results.tex | sim=0.85
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
Table~\ref{tab:fusion-variants} reports every variant and Fig.~\ref{fig:results}(a)
plots the ordering. AI alone reaches 0.6575 macro-F1 and 0.8762 accuracy; the
canonical per-channel hybrid reaches 0.6602 and 0.8774, a gain of 0.0027 and eight
more correct comments (5{,}861 versus 5{,}853), concentrated in \hate{}: F1 rises
from 0.5830 to 0.5906 through recall (0.5334 to 0.5494) at a small precision cost
(0.6427 to 0.6385), while \off{} and \clean{} barely move (0.4493 to 0.4494, 0.9403
to 0.9407)---added \hate{} recall, not a reallocation between classes.
`
### ORIGINAL (pre-condensation, for reference)
`
Table~\ref{tab:fusion-variants} reports every decision variant for the focused
backbone and Fig.~\ref{fig:results}(a) plots the same ordering. The AI alone reaches
0.6575 macro-F1 and 0.8762 accuracy; the canonical per-channel hybrid reaches 0.6602
and 0.8774, a gain of 0.0027 macro-F1 and eight additional correct comments (5{,}861
versus 5{,}853). The gain is concentrated in \hate{}: F1 rises from 0.5830 to 0.5906,
driven by recall (0.5334 to 0.5494) at a small cost in precision (0.6427 to 0.6385).
\off{} and \clean{} do not pay for it (0.4493 to 0.4494, 0.9403 to 0.9407): the
change is an extension of \hate{} recall, not a reallocation between classes.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: results.tex | sim=0.86
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
Per-channel weights add 0.0010 macro-F1 over a single global weight (0.6583 to
0.6593), thresholds a further 0.0009. The logistic meta-classifier is \emph{worse}
than AI alone (0.6513): six features on 2{,}672 validation comments overfit and shift
towards \hate{} recall. Thresholds make the rule engine usable alone---0.5258
macro-F1, identical standalone and in the pipeline, against 0.4714 with argmax, which
collapses on \off{}---but far below the transformer (0.8421).
`
### ORIGINAL (pre-condensation, for reference)
`
The other variants locate the effect: per-channel weights add 0.0010 macro-F1 over
a single global weight (0.6583 to 0.6593) and thresholds a further 0.0009. The
logistic meta-classifier is \emph{worse} than the AI alone (0.6513): fitting six
features on 2{,}672 validation comments overfits the rule scores and shifts the
operating point towards \hate{} recall. Thresholds do make the rule engine usable
on its own---0.5258 macro-F1, identical standalone and inside the pipeline, against
0.4714 with argmax, which collapses on \off{}---but it stays far below the
transformer (0.8421 accuracy).
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: discussion.tex | sim=0.87
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The result is better read as a successful audit: 38 of 6{,}680 decisions change,
each with a channel, a direction and a mechanism.
`
### ORIGINAL (pre-condensation, for reference)
`
The result is better read as a successful audit: in the canonical setup, 38 of
    6{,}680 decisions change, each described by a channel, a direction and a
mechanism.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: analysis.tex | sim=0.87
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The 132 are not missed for weak evidence: their median rule \hate{} score is 0.7609
(quartiles 0.67--0.85), indistinguishable from the 0.7652 median of corrected cases.
They are missed because the \clean{} channel keeps the full AI weight ($\alpha_C=1.0$),
so a confidently wrong \clean{} cannot be outvoted: the median fused \clean{} score is
0.97 for missed versus 0.46 for corrected cases. Evidence-gated fusion
(Sect.~\ref{sec:implications}) would abstain, but we did not tune such a rule on test,
which would invalidate the analysis.
`
### ORIGINAL (pre-condensation, for reference)
`
The 132 are not missed because the lexical evidence is weak: their median rule
\hate{} score is 0.7609 (quartiles 0.67--0.85), indistinguishable from the 0.7652
median of the corrected cases. They are missed because the \clean{} channel keeps the
full AI weight ($\alpha_C=1.0$), so a confidently wrong \clean{} cannot be outvoted:
the median fused \clean{} score is 0.97 for missed cases versus 0.46 for corrected
ones.
Evidence-gated fusion (Sect.~\ref{sec:implications}) would abstain here; we did not tune
such a rule on test, which would invalidate the analysis.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: introduction.tex | sim=0.87
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The contributions are: (i) a frozen, reproducible hybrid protocol in which the rule
engine is an independent, inspectable channel; (ii) a per-decision provenance analysis
assigning every change to a rule- or threshold-driven mechanism; (iii) a behavioural
audit of the rule channel with a cue taxonomy of recovered \hate{} cases; (iv) 132
missed corrections, motivating evidence-gated rather than score-averaged fusion; and
(v) five case studies with matched spans and per-channel scores.
`
### ORIGINAL (pre-condensation, for reference)
`
The contributions are: (i) a frozen, reproducible hybrid protocol in which the rule
engine is an independent, inspectable channel rather than a competing classifier;
(ii) a per-decision provenance analysis assigning every changed decision to a
rule-driven or threshold-driven mechanism; (iii) a behavioural audit of the rule
channel with a multi-label cue taxonomy of the recovered \hate{} cases; (iv) the
identification of 132 missed corrections, which motivates evidence-gated rather than
score-averaged fusion; and (v) five case studies with matched spans and per-channel
scores.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: introduction.tex | sim=0.88
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The release of UIT-ViHSD \cite{luu2021vihsd}, roughly 33k comments labelled
\clean{}, \off{} or \hate{}, made supervised transformer fine-tuning the default.
PhoBERT \cite{nguyen2020phobert}, ViSoBERT \cite{nguyen2023visobert}, multilingual
DeBERTa-v3 variants \cite{he2021deberta,he2023debertav3,liu2024mdeberta} and
ViHateT5 \cite{nguyen2024vihatet5} all perform well, yet on ViHSD strong encoders
plateau: about 83\% of test comments are \clean{}, \hate{} and \off{} overlap in
annotation, and their boundary is often intensity rather than lexical content.
`
### ORIGINAL (pre-condensation, for reference)
`
The release of UIT-ViHSD \cite{luu2021vihsd}, roughly 33k comments labelled
\clean{}, \off{} or \hate{}, made supervised transformer fine-tuning the default
approach. Monolingual encoders such as PhoBERT \cite{nguyen2020phobert} and
ViSoBERT \cite{nguyen2023visobert}, multilingual DeBERTa-v3 variants
\cite{he2021deberta,he2023debertav3,liu2024mdeberta} and task-specific systems
\cite{nguyen2024vihatet5} all perform well, yet on ViHSD strong encoders plateau:
about 83\% of test comments are \clean{}, \hate{} and \off{} overlap in annotation,
and their boundary is often a matter of intensity rather than lexical content.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: introduction.tex | sim=0.88
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
We ask a narrow question: \emph{can a lexicon-based rule engine be attached to a
transformer as an evidence channel, so every decision carries inspectable lexical
support, without degrading accuracy?} We do not claim higher accuracy: the focused
backbone improves by 0.0027 macro-F1 ($p=0.29$), a net of eight comments out of
6{,}680. We claim something more useful for deployment: the small change is
\emph{attributable}. Each of the 23 corrections and 15 regressions traces to a specific
mechanism, and the trace exposes a systematic failure that aggregate metrics hide.
`
### ORIGINAL (pre-condensation, for reference)
`
We ask a narrow question: \emph{can a lexicon-based rule engine be attached to a
transformer as an evidence channel, so that every decision carries inspectable
lexical support, without degrading accuracy?} We do not claim that the hybrid is
more accurate: on the ViHSD test split the focused backbone improves by 0.0027
macro-F1, a paired-bootstrap $p$ of 0.29, a net of eight comments out of 6{,}680. We
claim something different and, for deployment, more useful: the small change is
\emph{attributable}. Every one of the 23 corrections and 15 regressions traces to a
specific mechanism, and the trace exposes a systematic failure that aggregate
metrics hide.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: analysis.tex | sim=0.88
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
The 5 threshold-driven regressions are the mirror image: the rule is silent, the
fused \hate{} score is $0.2\,p_H$, and below $p_H\approx0.7$ the \hate{} channel no
longer wins; two miss by exactly 0.001 ($f_H=0.1390$ vs.\ $\tau_H=0.14$). One
mechanism ($\alpha_H=0.2$, $\tau_H=0.14$) thus produces 7 correct and 5 incorrect
de-flagging decisions and cannot be removed without also removing the corrections.
The decomposition is exhaustive---26 of 38 changes rule-driven, 12
threshold-driven---which is what makes the hybrid explainable: not a heat-map but a
statement of which channel moved, in which direction, with which spans, and by how
much.
`
### ORIGINAL (pre-condensation, for reference)
`
The 5 threshold-driven regressions are the mirror image: the rule is silent, the
fused \hate{} score is $0.2\,p_H$, and when $p_H$ falls below roughly 0.7 the \hate{}
channel no longer wins; two miss by exactly 0.001 ($f_H=0.1390$ against
$\tau_H=0.14$). The symmetry is the point: one mechanism ($\alpha_H=0.2$,
$\tau_H=0.14$) produces 7 correct de-flagging decisions and 5 incorrect ones, and
cannot be removed without also removing the corrections. The decomposition is
exhaustive---26 of the 38 changes rule-driven, 12 threshold-driven---and that is what
makes the hybrid explainable: not a heat-map over the transformer but a statement of
which channel moved, in which direction, with which spans, and by how much.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: discussion.tex | sim=0.89
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
\emph{Editing and diagnosis.} Adding a slur variant is a configuration change, not a
retraining run, and it affects the \hate{} channel (weight 0.8) most. The decomposition
exists only because the channels are separate: a monolithic classifier improving by
0.0027 could not show whether the change was lexical or neural.
`
### ORIGINAL (pre-condensation, for reference)
`
\emph{Editing and diagnosis.} Adding a slur variant is a configuration change, not
a retraining run, and an edit affects the \hate{} channel (weight 0.8) far more than
the others. The decomposition exists only because the channels are separate: a
monolithic classifier improving by 0.0027 macro-F1 would give no handle on whether
the change was lexical or neural.
`

==============================================================================
PRIORITY: P3 (light touch) | FILE: analysis.tex | sim=0.90
==============================================================================
### CURRENT (in manuscript now) - REWRITE THIS BY HAND
`
We annotated the 16 \hate{} recoveries with a multi-label cue taxonomy: ten contain
teencode or abbreviated profanity (\texttt{đ*o}, \texttt{dm}, \texttt{cmnr}), five
standard-spelling profanity or an animal slur, and eight derogatory wording without
profanity (\emph{ngu}, \emph{thua}, \emph{nhục}); one has no surface cue, and
categories overlap, so counts exceed 16. The channel is not merely a profanity
detector: much of its value comes from derogatory wording a slur list would miss.
`
### ORIGINAL (pre-condensation, for reference)
`
We annotated the 16 \hate{} recoveries with a multi-label cue taxonomy: ten
contain teencode or abbreviated profanity (\texttt{đ*o}, \texttt{dm},
\texttt{cmnr}), five
standard-spelling profanity or an animal slur, and eight derogatory wording without
profanity (\emph{ngu}, \emph{thua}, \emph{nhục}); one has no surface cue. The
categories overlap, so the counts sum to more than 16. The channel is therefore not
merely a profanity detector: much of its value comes from derogatory wording a slur
list would miss, and teencode is its largest single category.
`

