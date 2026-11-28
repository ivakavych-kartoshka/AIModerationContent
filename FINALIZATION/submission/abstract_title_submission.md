# Abstract and Title Submission Package

## Title
An Explainable Hybrid of Lexicon Rules and Multilingual        Transformers for Vietnamese Hate Speech Detection

## Abstract
Fine-tuned multilingual transformers dominate Vietnamese hate speech detection,
yet they are opaque and brittle on the surface forms moderation sees: teencode,
dropped diacritics, inserted separators and abbreviated profanity.  We ask whether
a lexicon rule engine can be attached to such a transformer as an \emph{evidence
channel} rather than a competing classifier.  On the ViHSD test set (6{,}680
comments, 83\% \clean) we fuse a 668-term Vietnamese hate-speech lexicon, matched
through accent-less, leet, repeat-collapsed and separator-stripped views, with
four backbones that differ only in their optimisation recipe.  The headline effect
is small and not distinguishable from zero: the focused backbone improves from
0.6575 to 0.6602 macro-F1, a net of 8 comments (23 fixed, 15 broken),
with a paired-bootstrap $p=0.29$.  The value we report is not the delta but its
\emph{provenance}.  The rule channel fires on 15.75\% of test comments with
58.75\% precision, and all 23 corrections are rule-consistent: 16 \hate{}
recoveries where the rule fires in the corrected direction, and 7 \clean{}
de-flagging cases where it is correctly silent.  All 10 rule-driven regressions
push towards \hate{}, localising the failure mode to an aggressive channel rather
than to lexical coverage.  We further identify 132 missed corrections where the
rule already carries directional evidence fusion does not act on.  These
diagnostics, not the fourth decimal of macro-F1, are the contribution.
\keywords{Vietnamese hate speech \and ViHSD \and Explainable AI \and
          Hybrid fusion \and Lexicon rules \and Error analysis.}
