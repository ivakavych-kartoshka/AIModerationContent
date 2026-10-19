"""TF-IDF noise filter for the lexical dictionary (research plan, section 36).

The corpus extractor (``scripts/15_extract_corpus_words.py``) proposes a term
as soon as it appears in enough toxic rows and few clean rows, and the labeler
(``src/data/lexical_labeler.py``) turns each proposal into a ``HATE`` /
``OFFENSIVE`` entry. Both steps only look at raw counts, so a term that is
merely *common* in the training vocabulary - a generic sentence fragment, a
platform name, a number - can slip into the dictionary as an ``auto`` entry.
That is exactly the noise audited in ``data/processed/lexicon/verified_lexicon.csv``.

This module adds the classic **TF-IDF** view on top of the counts:

* ``tf_*``      - raw occurrence count of the term in each class corpus;
* ``idf``       - ``log((N + s) / (df + s)) + 1`` over *all* training rows,
  where ``df`` is the number of rows containing the term. A term that shows up
  everywhere has ``idf -> 1``; a term that is rare in the corpus has a large
  ``idf``. Putting ``idf`` into the weight is what stops ubiquitous words from
  looking important;
* ``tfidf_*``   - class-normalised TF (``tf / rows of that class``) times the
  ``idf`` - directly comparable across terms and classes;
* ``tfidf_ratio`` - ``tfidf_toxic / (tfidf_toxic + tfidf_clean)`` in ``[0, 1]``;
  near ``1`` means the weight lives almost entirely in the toxic classes, near
  ``0`` means the term is a CLEAN term;
* ``tfidf_score`` - ``idf * (ntf_toxic - ntf_clean)``: the signed, IDF-weighted
  separation between the toxic and CLEAN rates. Because ``idf`` multiplies the
  difference, a *common* term (small ``idf``) must separate the classes much
  more strongly than a rare one to reach the same score.

A term is kept only when ``tfidf_ratio >= min_ratio`` and
``tfidf_score >= min_score``; otherwise it is dropped from the dictionary so it
never reaches the rule engine (section 36.2: false positives must not enter the
dictionary). The thresholds are initial hypotheses and belong to the same
validation-time tuning as the rest of :class:`~src.data.lexical_labeler.LabelingConfig`.

The functions here are pure (no I/O) so they can be unit-tested on hand-built
counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class TfidfConfig:
    """Thresholds of the TF-IDF gate (section 36, tune on validation)."""

    enabled: bool = True
    #: keep only when the toxic share of the TF-IDF weight reaches this
    min_ratio: float = 0.60
    #: keep only when the IDF-weighted toxic-vs-clean separation reaches this
    min_score: float = 0.004
    #: add-k smoothing inside the IDF
    smoothing: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "min_ratio": self.min_ratio,
            "min_score": self.min_score,
            "smoothing": self.smoothing,
        }


@dataclass
class TermTfidf:
    """Every TF-IDF number computed for one candidate term."""

    tf_hate: int
    tf_offensive: int
    tf_clean: int
    df_total: int
    idf: float
    tfidf_hate: float
    tfidf_offensive: float
    tfidf_clean: float
    tfidf_toxic: float
    ratio: float
    hate_share: float
    offensive_share: float
    score: float

    def to_row(self) -> Dict[str, float]:
        return {
            "tf_hate": self.tf_hate,
            "tf_offensive": self.tf_offensive,
            "tf_clean": self.tf_clean,
            "idf": round(self.idf, 4),
            "tfidf_hate": round(self.tfidf_hate, 4),
            "tfidf_offensive": round(self.tfidf_offensive, 4),
            "tfidf_clean": round(self.tfidf_clean, 4),
            "tfidf_toxic": round(self.tfidf_toxic, 4),
            "tfidf_ratio": round(self.ratio, 4),
            "tfidf_score": round(self.score, 4),
        }


def compute_idf(df_total: int, n_docs: int, smoothing: float = 1.0) -> float:
    """Smoothed inverse document frequency; ``>= 1`` for any term seen at least once."""
    numerator = n_docs + smoothing
    denominator = df_total + smoothing
    if numerator <= 0 or denominator <= 0:
        return 1.0
    return math.log(numerator / denominator) + 1.0


def term_tfidf(
    tf_hate: int,
    tf_offensive: int,
    tf_clean: int,
    df_total: int,
    n_hate: int,
    n_offensive: int,
    n_clean: int,
    cfg: TfidfConfig | None = None,
) -> TermTfidf:
    """Compute every TF-IDF number for one term from its per-class counts.

    ``tf_*`` are occurrence counts, ``n_*`` the number of training rows of each
    class, and ``df_total`` the number of rows (any class) that contain the term.
    """
    cfg = cfg or TfidfConfig()
    n_docs = max(int(n_hate) + int(n_offensive) + int(n_clean), 1)
    df_total = max(int(df_total), 0)

    idf_value = compute_idf(df_total, n_docs, cfg.smoothing)

    ntf_hate = tf_hate / max(n_hate, 1)
    ntf_off = tf_offensive / max(n_offensive, 1)
    ntf_clean = tf_clean / max(n_clean, 1)

    tfidf_hate = ntf_hate * idf_value
    tfidf_off = ntf_off * idf_value
    tfidf_clean = ntf_clean * idf_value

    tfidf_toxic = tfidf_hate + tfidf_off
    denom = tfidf_toxic + tfidf_clean
    ratio = tfidf_toxic / denom if denom > 0 else 0.0

    toxic_denom = tfidf_toxic if tfidf_toxic > 0 else 1.0

    return TermTfidf(
        tf_hate=int(tf_hate),
        tf_offensive=int(tf_offensive),
        tf_clean=int(tf_clean),
        df_total=df_total,
        idf=idf_value,
        tfidf_hate=tfidf_hate,
        tfidf_offensive=tfidf_off,
        tfidf_clean=tfidf_clean,
        tfidf_toxic=tfidf_toxic,
        ratio=ratio,
        hate_share=tfidf_hate / toxic_denom,
        offensive_share=tfidf_off / toxic_denom,
        score=idf_value * ((ntf_hate + ntf_off) - ntf_clean),
    )


def evaluate(tfidf_toxic: float, ratio: float, score: float, cfg: TfidfConfig | None = None) -> Tuple[bool, str]:
    """Shared gate: ``(keep, reason)`` from the three TF-IDF decision numbers."""
    cfg = cfg or TfidfConfig()
    if not cfg.enabled:
        return True, ""
    if tfidf_toxic <= 0.0:
        return False, "tfidf_no_toxic"
    if ratio < cfg.min_ratio:
        return False, "tfidf_clean_dominant"
    if score < cfg.min_score:
        return False, "tfidf_low_score"
    return True, ""


def passes(stats: TermTfidf, cfg: TfidfConfig | None = None) -> Tuple[bool, str]:
    """Return ``(keep, reason)`` for a labelled term; ``reason`` empty when kept."""
    return evaluate(stats.tfidf_toxic, stats.ratio, stats.score, cfg)


@dataclass
class CombinedTfidf:
    """TF-IDF numbers when only ``toxic`` (HATE + OFFENSIVE) vs ``CLEAN`` is known.

    Used by the corpus extractor, which counts a single toxic counter instead of
    separate HATE / OFFENSIVE counters.
    """

    tf_toxic: int
    tf_clean: int
    df_total: int
    idf: float
    tfidf_toxic: float
    tfidf_clean: float
    ratio: float
    score: float

    def to_row(self) -> Dict[str, float]:
        return {
            "tf_toxic": self.tf_toxic,
            "tf_clean": self.tf_clean,
            "idf": round(self.idf, 4),
            "tfidf_toxic": round(self.tfidf_toxic, 4),
            "tfidf_clean": round(self.tfidf_clean, 4),
            "tfidf_ratio": round(self.ratio, 4),
            "tfidf_score": round(self.score, 4),
        }


def combined_tfidf(
    tf_toxic: int,
    tf_clean: int,
    n_toxic: int,
    n_clean: int,
    cfg: TfidfConfig | None = None,
) -> CombinedTfidf:
    """TF-IDF numbers from combined toxic/CLEAN document frequencies."""
    cfg = cfg or TfidfConfig()
    n_docs = max(int(n_toxic) + int(n_clean), 1)
    df_total = int(tf_toxic) + int(tf_clean)
    idf_value = compute_idf(df_total, n_docs, cfg.smoothing)
    ntf_toxic = tf_toxic / max(int(n_toxic), 1)
    ntf_clean = tf_clean / max(int(n_clean), 1)
    tfidf_toxic = ntf_toxic * idf_value
    tfidf_clean = ntf_clean * idf_value
    denom = tfidf_toxic + tfidf_clean
    return CombinedTfidf(
        tf_toxic=int(tf_toxic),
        tf_clean=int(tf_clean),
        df_total=df_total,
        idf=idf_value,
        tfidf_toxic=tfidf_toxic,
        tfidf_clean=tfidf_clean,
        ratio=tfidf_toxic / denom if denom > 0 else 0.0,
        score=idf_value * (ntf_toxic - ntf_clean),
    )
