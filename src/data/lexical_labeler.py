"""Lexical labelling of corpus-extracted candidates (research plan, section 36).

Every term produced by ``scripts/15_extract_corpus_words.py`` is currently a
bare string. This module turns each candidate into a lexicon entry carrying
the vocabulary labels ``HATE`` / ``OFFENSIVE`` (section 36.2) together with the
evidence behind the decision.

Evidence (section 36.1)
-----------------------
Only **training splits** are scanned - ViHSD ``train.csv``, ViCM ``train.xlsx``
and bontori ``tdtu_train.parquet`` - so validation/test text never influences
the dictionary. For every candidate we count, per canonical label, the number
of training rows whose token n-gram set contains the term:

* ``n_hate`` / ``n_offensive`` / ``n_clean`` - document frequency per label,
* ``r_*`` - the same value divided by the number of rows of that label
  (class-size normalisation, otherwise the biggest class dominates),
* ``share_hate = r_hate / (r_hate + r_offensive)`` - which toxic class the
  term is associated with,
* ``lift_* = r_* / r_clean`` - how much more often the term shows up in that
  class than in CLEAN rows (kills corpus artefacts such as "thớt" or
  "content" that only *correlate* with toxicity).

Labelling rules (initial hypotheses, tuned later on the validation split)
-------------------------------------------------------------------------
* ``HATE``        when ``share_hate >= share_threshold`` and ``lift_hate >= clean_lift``
* ``OFFENSIVE``   when ``share_hate <= 1 - share_threshold`` and ``lift_offensive >= clean_lift``
* both labels     when the share sits in the middle and both lifts pass
                  -> ``context_required = true``
* otherwise       no lexical label; the term goes to the review queue.

A vote is additionally vetoed when the term is common in CLEAN rows
(``n_clean >= clean_count_cap``) *unless* its lift is overwhelming
(``lift >= strong_lift``) - this removes corpus artefacts such as "content",
"trump" or "bọn này" that merely correlate with toxicity (section 36.2:
false positives must not enter the dictionary).

``severity`` is the fixed table from section 39 (HATE 0.90, OFFENSIVE 0.60);
``confidence`` mixes evidence size, dominance margin and clean lift. Terms are
marked ``verified`` / ``auto`` / ``needs_review`` / ``rejected`` so that
ambiguous entries can be checked by a human before the dictionary is frozen.

Human review (section 36.2)
---------------------------
``manual_review.csv`` (optional, next to the outputs) can override the
automatic decision per term: ``term,review_status[,lexical_labels[,notes]]``.
Re-running the labeler merges those rows last, so an ambiguous entry can be
promoted to ``verified`` - or a bad ``auto`` label demoted - before the
dictionary is frozen.

Outputs (written to ``data/processed/lexicon/``, section 42)
------------------------------------------------------------
``candidates_train.csv``    full evidence table for every candidate
``verified_lexicon.csv``    entries with at least one lexical label (schema 36.2)
``review_queue.csv``        weak / ambiguous / rejected terms + reason
``label_mapping.yaml``      dataset label -> canonical label mapping
``provenance.json``         SHA-256 + row counts + parameters + deviations

Usage
-----
    python -m src.data.lexical_labeler
    python -m src.data.lexical_labeler --min-evidence 5 --show-terms 20
    python -m src.data.lexical_labeler --validate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

from ..common.constants import (
    LABELS,
    LEXICON_PROCESSED_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from ..common.logging_utils import get_logger
from .corpus_lexicon import _map_bontori_label, _map_vicm_label, tokenize_row
from .lexicon import OUTPUT_REPORT, _sha256, normalize_term

LOGGER = get_logger("sensitiveai.lexical_labeler")

# --------------------------------------------------------------------------- #
# Output files (section 42 naming, placed in the existing processed lexicon dir)
# --------------------------------------------------------------------------- #
CANDIDATES_INPUT = "sensitive_words_from_datasets.txt"
OUTPUT_CANDIDATES = "candidates_train.csv"
OUTPUT_VERIFIED = "verified_lexicon.csv"
OUTPUT_REVIEW_QUEUE = "review_queue.csv"
OUTPUT_LABEL_MAPPING = "label_mapping.yaml"
OUTPUT_PROVENANCE = "provenance.json"
MANUAL_REVIEW_FILE = "manual_review.csv"

DICTIONARY_VERSION_DEFAULT = "v1"
SOURCE_SPLIT = "train"


# --------------------------------------------------------------------------- #
# Configuration (section 39 - initial hypotheses, tune on validation)
# --------------------------------------------------------------------------- #
@dataclass
class LabelingConfig:
    """Thresholds of the lexical labelling rules (all overridable via CLI)."""

    min_evidence: int = 3
    share_threshold: float = 0.6
    clean_lift: float = 3.0
    #: veto a vote when the term is common in CLEAN rows, unless lift is huge
    clean_count_cap: int = 20
    strong_lift: float = 15.0
    verified_confidence: float = 0.7
    verified_evidence: int = 10
    auto_confidence: float = 0.4
    #: severity table, section 39 of the plan
    severity: Dict[str, float] = field(
        default_factory=lambda: {"HATE": 0.90, "OFFENSIVE": 0.60}
    )
    #: confidence weights: evidence size / dominance margin / clean lift
    confidence_weights: Dict[str, float] = field(
        default_factory=lambda: {"evidence": 0.4, "margin": 0.3, "lift": 0.3}
    )
    confidence_full_evidence: float = 10.0
    confidence_lift_cap: float = 100.0
    dictionary_version: str = DICTIONARY_VERSION_DEFAULT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_evidence": self.min_evidence,
            "share_threshold": self.share_threshold,
            "clean_lift": self.clean_lift,
            "clean_count_cap": self.clean_count_cap,
            "strong_lift": self.strong_lift,
            "verified_confidence": self.verified_confidence,
            "verified_evidence": self.verified_evidence,
            "auto_confidence": self.auto_confidence,
            "severity": dict(self.severity),
            "confidence_weights": dict(self.confidence_weights),
            "confidence_full_evidence": self.confidence_full_evidence,
            "confidence_lift_cap": self.confidence_lift_cap,
            "dictionary_version": self.dictionary_version,
        }


# --------------------------------------------------------------------------- #
# Evidence collection (training splits only)
# --------------------------------------------------------------------------- #
def _load_vihsd_train() -> List[Tuple[str, str, str]]:
    import pandas as pd

    path = PROCESSED_DIR / "vihsd" / "train.csv"
    if not path.is_file():
        LOGGER.warning("[labeler] missing %s", path)
        return []
    df = pd.read_csv(path, dtype={"text": "string"})
    rows = [
        ("vihsd", str(lab), str(txt) if pd.notna(txt) else "")
        for txt, lab in zip(df["text"], df["label_name"])
        if str(lab) in LABELS
    ]
    LOGGER.info("[labeler] ViHSD train: %d usable rows", len(rows))
    return rows


def _load_vicm_train() -> List[Tuple[str, str, str]]:
    from . import additional

    path = RAW_DIR / "additional" / "vicm-code-mixed" / "train.xlsx"
    if not path.is_file():
        LOGGER.warning("[labeler] missing %s", path)
        return []
    rows: List[Tuple[str, str, str]] = []
    for record in additional._xlsx_rows(path):
        text = record.get("text") or record.get("Text") or record.get("B") or ""
        label = _map_vicm_label(record.get("Label") or record.get("label") or record.get("C"))
        if label in LABELS:
            rows.append(("vicm", label, str(text)))
    LOGGER.info("[labeler] ViCM train: %d usable rows", len(rows))
    return rows


def _load_bontori_train() -> List[Tuple[str, str, str]]:
    import pandas as pd

    path = RAW_DIR / "additional" / "bontori-vietnamese-hsd-cleaned" / "tdtu_train.parquet"
    if not path.is_file():
        LOGGER.warning("[labeler] missing %s", path)
        return []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - missing engine
        LOGGER.warning("[labeler] cannot read %s: %s", path, exc)
        return []
    rows: List[Tuple[str, str, str]] = []
    for text, label_raw in zip(df.get("text", []), df.get("label", [])):
        label = _map_bontori_label(label_raw)
        if label in LABELS:
            rows.append(("bontori", label, str(text) if pd.notna(text) else ""))
    LOGGER.info("[labeler] bontori train: %d usable rows", len(rows))
    return rows


def load_training_rows() -> List[Tuple[str, str, str]]:
    """All ``(dataset, canonical_label, text)`` rows of the three train splits."""
    rows: List[Tuple[str, str, str]] = []
    for loader in (_load_vihsd_train, _load_vicm_train, _load_bontori_train):
        rows.extend(loader())
    LOGGER.info("[labeler] training rows in total: %d", len(rows))
    return rows


def load_candidates(path: str | Path) -> List[str]:
    """One normalised term per line; ``#`` lines are comments."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate list not found: {path}")
    seen: Set[str] = set()
    ordered: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = normalize_term(line)
        if term and term not in seen:
            seen.add(term)
            ordered.append(term)
    LOGGER.info("[labeler] candidates loaded: %d (%s)", len(ordered), path)
    return ordered


def collect_evidence(
    candidates: Sequence[str],
    rows: Iterable[Tuple[str, str, str]],
) -> Tuple[Dict[str, Counter], Dict[str, Set[str]], Counter, Dict[str, Counter]]:
    """Count, per candidate, the training rows that contain it.

    Returns ``(per-term label counts, per-term dataset set, rows per label,
    rows per dataset and label)``.
    """
    candidate_set = set(candidates)
    evidence: Dict[str, Counter] = defaultdict(Counter)
    term_datasets: Dict[str, Set[str]] = defaultdict(set)
    rows_by_label: Counter = Counter()
    rows_by_dataset_label: Dict[str, Counter] = defaultdict(Counter)

    for dataset, label, text in rows:
        rows_by_label[label] += 1
        rows_by_dataset_label[dataset][label] += 1
        tokens = tokenize_row(text)
        if not tokens:
            continue
        hits = set(tokens)
        hits.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if len(a) + len(b) >= 3)
        for term in hits & candidate_set:
            evidence[term][label] += 1
            term_datasets[term].add(dataset)

    LOGGER.info(
        "[labeler] evidence: %d/%d candidates seen in train, rows by label %s",
        len(evidence),
        len(candidates),
        dict(rows_by_label),
    )
    return evidence, term_datasets, rows_by_label, rows_by_dataset_label


# --------------------------------------------------------------------------- #
# Labelling rules (section 36.2)
# --------------------------------------------------------------------------- #
def _confidence(total_toxic: int, share_hate: float, lift: float, cfg: LabelingConfig) -> float:
    weights = cfg.confidence_weights
    ev = min(1.0, total_toxic / max(cfg.confidence_full_evidence, 1e-9))
    margin = abs(share_hate - 0.5) * 2.0
    lift_capped = min(max(lift, 0.0), cfg.confidence_lift_cap)
    lift_score = math.log1p(lift_capped) / math.log1p(cfg.confidence_lift_cap) if lift_capped > 0 else 0.0
    value = (
        weights.get("evidence", 0.4) * ev
        + weights.get("margin", 0.3) * margin
        + weights.get("lift", 0.3) * lift_score
    )
    return round(min(max(value, 0.05), 0.99), 4)


def label_term(
    term: str,
    counts: Counter,
    datasets: Set[str],
    rows_by_label: Counter,
    cfg: LabelingConfig,
) -> Dict[str, Any]:
    """Apply the labelling rules to one candidate and return its record."""
    n_hate = int(counts.get("HATE", 0))
    n_off = int(counts.get("OFFENSIVE", 0))
    n_clean = int(counts.get("CLEAN", 0))
    total_toxic = n_hate + n_off

    n_hate_rows = max(int(rows_by_label.get("HATE", 0)), 1)
    n_off_rows = max(int(rows_by_label.get("OFFENSIVE", 0)), 1)
    n_clean_rows = max(int(rows_by_label.get("CLEAN", 0)), 1)

    r_hate = n_hate / n_hate_rows
    r_off = n_off / n_off_rows
    r_clean = n_clean / n_clean_rows
    toxic_rate = total_toxic / (n_hate_rows + n_off_rows)

    denom = r_hate + r_off
    share_hate = r_hate / denom if denom > 0 else 0.5
    share_off = 1.0 - share_hate

    def _lift(rate: float) -> float:
        if n_clean == 0:
            return float("inf") if rate > 0 else 0.0
        return rate / max(r_clean, 1e-12)

    lift_hate = _lift(r_hate)
    lift_off = _lift(r_off)
    lift_toxic = _lift(toxic_rate)

    record: Dict[str, Any] = {
        "term": term,
        "n_hate": n_hate,
        "n_offensive": n_off,
        "n_clean": n_clean,
        "total_toxic": total_toxic,
        "r_hate": round(r_hate, 6),
        "r_offensive": round(r_off, 6),
        "r_clean": round(r_clean, 6),
        "share_hate": round(share_hate, 4),
        "lift_hate": round(lift_hate, 3) if math.isfinite(lift_hate) else 999.0,
        "lift_offensive": round(lift_off, 3) if math.isfinite(lift_off) else 999.0,
        "lift_toxic": round(lift_toxic, 3) if math.isfinite(lift_toxic) else 999.0,
        "source_datasets": ";".join(sorted(datasets)),
        "lexical_labels": [],
        "severity": 0.0,
        "confidence": 0.0,
        "context_required": False,
        "review_status": "rejected",
        "reason": "",
    }

    # --- weak / absent evidence ------------------------------------------ #
    if total_toxic == 0:
        record["reason"] = "no_toxic_evidence"
        return record
    if total_toxic < cfg.min_evidence:
        record["review_status"] = "needs_review"
        record["reason"] = "weak_evidence"
        return record

    # --- votes ------------------------------------------------------------- #
    # A vote needs the class share, enough lift over CLEAN rows, and a clean
    # count below the cap unless the lift is overwhelming (anti-artefact rule).
    def _vote(lift: float) -> bool:
        if lift < cfg.clean_lift:
            return False
        return n_clean < cfg.clean_count_cap or lift >= cfg.strong_lift

    hate_pass = share_hate >= cfg.share_threshold and _vote(lift_hate)
    off_pass = share_off >= cfg.share_threshold and _vote(lift_off)
    middle = cfg.share_threshold > share_hate > (1.0 - cfg.share_threshold)
    middle_pass = middle and _vote(lift_hate) and _vote(lift_off)

    votes: List[str] = []
    both_by_default = False
    if hate_pass:
        votes.append("HATE")
    if off_pass:
        votes.append("OFFENSIVE")
    if not votes and middle_pass:
        votes = ["HATE", "OFFENSIVE"]
        both_by_default = True
    if not votes:
        record["review_status"] = "needs_review"
        base_ok = (
            (share_hate >= cfg.share_threshold and lift_hate >= cfg.clean_lift)
            or (share_off >= cfg.share_threshold and lift_off >= cfg.clean_lift)
            or (lift_hate >= cfg.clean_lift and lift_off >= cfg.clean_lift)
        )
        record["reason"] = "clean_context_leak" if base_ok else "ambiguous_or_clean_leak"
        return record

    lift_for_conf = max(lift_hate if "HATE" in votes else 0.0, lift_off if "OFFENSIVE" in votes else 0.0)
    confidence = _confidence(total_toxic, share_hate, lift_for_conf, cfg)
    severity = max(cfg.severity.get(label, 0.0) for label in votes)

    record["lexical_labels"] = votes
    record["severity"] = round(severity, 4)
    record["confidence"] = confidence
    record["context_required"] = bool(both_by_default or n_clean > 0)

    if confidence >= cfg.verified_confidence and total_toxic >= cfg.verified_evidence:
        record["review_status"] = "verified"
        record["reason"] = "strong_evidence"
    elif confidence >= cfg.auto_confidence:
        record["review_status"] = "auto"
        record["reason"] = "auto_labelled"
    else:
        record["review_status"] = "needs_review"
        record["reason"] = "low_confidence"
    return record


def label_all(
    candidates: Sequence[str],
    evidence: Dict[str, Counter],
    term_datasets: Dict[str, Set[str]],
    rows_by_label: Counter,
    cfg: LabelingConfig,
) -> List[Dict[str, Any]]:
    records = [
        label_term(term, evidence.get(term, Counter()), term_datasets.get(term, set()), rows_by_label, cfg)
        for term in candidates
    ]
    records.sort(key=lambda r: r["term"])
    return records


# --------------------------------------------------------------------------- #
# Manual review overrides (section 36.2)
# --------------------------------------------------------------------------- #
VALID_STATUSES = ("verified", "auto", "needs_review", "rejected")


def load_overrides(path: str | Path) -> Dict[str, Dict[str, str]]:
    """Read ``manual_review.csv`` into ``{term: {review_status, lexical_labels, notes}}``."""
    path = Path(path)
    if not path.is_file():
        return {}
    overrides: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            term = normalize_term(row.get("term", ""))
            status = (row.get("review_status") or "").strip().lower()
            if not term or status not in VALID_STATUSES:
                LOGGER.warning("[labeler] ignoring override for %r (status %r)", term or "?", status)
                continue
            overrides[term] = {
                "review_status": status,
                "lexical_labels": (row.get("lexical_labels") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
    LOGGER.info("[labeler] manual overrides loaded: %d (%s)", len(overrides), path.name)
    return overrides


def apply_overrides(
    records: List[Dict[str, Any]],
    overrides: Dict[str, Dict[str, str]],
    cfg: LabelingConfig,
) -> int:
    """Apply human decisions in place; returns how many records were touched."""
    if not overrides:
        return 0
    applied = 0
    for record in records:
        override = overrides.get(record["term"])
        if not override:
            continue
        status = override["review_status"]
        labels = [x for x in override["lexical_labels"].replace(",", ";").split(";") if x]
        if labels:
            invalid = [x for x in labels if x not in ("HATE", "OFFENSIVE")]
            if invalid:
                LOGGER.warning("[labeler] %r: invalid labels %s - keeping automatic", record["term"], invalid)
                labels = []
        if status == "rejected":
            labels = []
        record["review_status"] = status
        if labels:
            record["lexical_labels"] = labels
            record["severity"] = round(max(cfg.severity.get(label, 0.0) for label in labels), 4)
            record["context_required"] = bool(record["context_required"] or len(labels) > 1)
        elif status == "rejected":
            record["lexical_labels"] = []
            record["severity"] = 0.0
        record["reason"] = "manual_override"
        applied += 1
    LOGGER.info("[labeler] manual overrides applied: %d", applied)
    return applied


# --------------------------------------------------------------------------- #
# Validation of the produced lexicon (section 42 - validate_lexicon)
# --------------------------------------------------------------------------- #
def validate_lexicon(output_dir: Path) -> Dict[str, Any]:
    """Schema checks on ``verified_lexicon.csv``; raises on hard violations."""
    problems: List[str] = []
    path = output_dir / OUTPUT_VERIFIED
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run the label stage first")

    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    required = [
        "entry_id",
        "term",
        "canonical_form",
        "lexical_labels",
        "severity",
        "confidence",
        "context_required",
        "source_datasets",
        "source_split",
        "review_status",
        "dictionary_version",
    ]
    ids: Set[str] = set()
    counts: Counter = Counter()
    for row in rows:
        for key in required:
            if key not in row or row[key] in (None, ""):
                problems.append(f"{row.get('entry_id', '?')}: missing field {key}")
        if row["entry_id"] in ids:
            problems.append(f"duplicate entry_id {row['entry_id']}")
        ids.add(row["entry_id"])
        labels = [x for x in row["lexical_labels"].split(";") if x]
        if not labels or any(x not in ("HATE", "OFFENSIVE") for x in labels):
            problems.append(f"{row['entry_id']}: invalid lexical_labels {row['lexical_labels']!r}")
        severity = float(row["severity"])
        confidence = float(row["confidence"])
        if not (0.0 < severity <= 1.0):
            problems.append(f"{row['entry_id']}: severity {severity} outside (0,1]")
        if not (0.0 < confidence <= 1.0):
            problems.append(f"{row['entry_id']}: confidence {confidence} outside (0,1]")
        if row["source_split"] != SOURCE_SPLIT:
            problems.append(f"{row['entry_id']}: source_split must be 'train'")
        counts[row["review_status"]] += 1
        for label in labels:
            counts[f"label_{label}"] += 1

    summary = {"entries": len(rows), "by_status": dict(counts), "problems": problems[:50]}
    if problems:
        LOGGER.error("[labeler] validation found %d problem(s), first: %s", len(problems), problems[0])
    else:
        LOGGER.info("[labeler] validation OK: %d entries, %s", len(rows), dict(counts))
    return summary


# --------------------------------------------------------------------------- #
# Outputs (section 42)
# --------------------------------------------------------------------------- #
def _write_candidates_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "term",
        "n_hate",
        "n_offensive",
        "n_clean",
        "total_toxic",
        "r_hate",
        "r_offensive",
        "r_clean",
        "share_hate",
        "lift_hate",
        "lift_offensive",
        "lift_toxic",
        "source_datasets",
        "review_status",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_verified_csv(path: Path, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = [
        "entry_id",
        "term",
        "canonical_form",
        "lexical_labels",
        "severity",
        "confidence",
        "context_required",
        "source_datasets",
        "source_split",
        "review_status",
        "dictionary_version",
    ]
    labelled = [r for r in records if r["lexical_labels"]]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(labelled, start=1):
            row = {
                "entry_id": f"lex_{index:04d}",
                "term": record["term"],
                "canonical_form": record["term"],
                "lexical_labels": ";".join(record["lexical_labels"]),
                "severity": record["severity"],
                "confidence": record["confidence"],
                "context_required": str(bool(record["context_required"])).lower(),
                "source_datasets": record["source_datasets"],
                "source_split": SOURCE_SPLIT,
                "review_status": record["review_status"],
                "dictionary_version": record.get(
                    "dictionary_version", DICTIONARY_VERSION_DEFAULT
                ),
            }
            writer.writerow(row)
    LOGGER.info("[labeler] wrote %s (%d labelled entries)", path.name, len(labelled))
    return labelled


def _write_review_queue(path: Path, records: Sequence[Dict[str, Any]]) -> int:
    fields = ["term", "review_status", "reason", "n_hate", "n_offensive", "n_clean", "share_hate", "source_datasets"]
    queue = [r for r in records if r["review_status"] in ("needs_review", "rejected")]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(queue, key=lambda r: (-r["total_toxic"], r["term"])):
            writer.writerow(record)
    LOGGER.info("[labeler] wrote %s (%d terms need review)", path.name, len(queue))
    return len(queue)


def build_label_mapping() -> Dict[str, Any]:
    """Manual-review mapping of each dataset's native labels to canonical ones."""
    return {
        "dictionary_version": DICTIONARY_VERSION_DEFAULT,
        "canonical_labels": list(LABELS),
        "review_status": "confirmed",
        "mappings": {
            "vihsd": {"0": "CLEAN", "1": "OFFENSIVE", "2": "HATE"},
            "vicm": {"0": "CLEAN", "1": "HATE", "2": "OFFENSIVE"},
            "bontori": {"CLEAN": "CLEAN", "OFFENSIVE": "OFFENSIVE", "HATE": "HATE"},
        },
        "notes": [
            "ViHSD native ids from uitnlp/vihsd (constants.VIHSD_ID2LABEL).",
            "ViCM as coded in src/data/corpus_lexicon.py::_map_vicm_label - verify against the ViCM README before freezing.",
            "bontori already publishes canonical label names.",
            "Sentence labels are NOT word labels: a lexical label is a vocabulary signal only (plan section 35).",
        ],
    }


def _write_label_mapping(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(build_label_mapping(), fh, sort_keys=False, allow_unicode=True, default_flow_style=False)
    LOGGER.info("[labeler] wrote %s", path.name)


def _input_fingerprints() -> Dict[str, Any]:
    from .corpus_lexicon import DEFAULT_FILES_BONTORI, DEFAULT_FILES_VICM

    files: Dict[str, Any] = {}
    candidates_path = LEXICON_PROCESSED_DIR / CANDIDATES_INPUT
    if candidates_path.is_file():
        files[str(candidates_path.relative_to(PROCESSED_DIR.parent.parent))] = _sha256(candidates_path)
    vihsd = PROCESSED_DIR / "vihsd" / "train.csv"
    if vihsd.is_file():
        files["data/processed/vihsd/train.csv"] = _sha256(vihsd)
    for name in DEFAULT_FILES_VICM:
        p = RAW_DIR / "additional" / "vicm-code-mixed" / name
        if p.is_file():
            files[f"data/raw/additional/vicm-code-mixed/{name}"] = _sha256(p)
    for name in DEFAULT_FILES_BONTORI:
        p = RAW_DIR / "additional" / "bontori-vietnamese-hsd-cleaned" / name
        if p.is_file():
            files[f"data/raw/additional/bontori-vietnamese-hsd-cleaned/{name}"] = _sha256(p)
    return files


def _write_provenance(
    path: Path,
    cfg: LabelingConfig,
    records: Sequence[Dict[str, Any]],
    rows_by_label: Counter,
    rows_by_dataset_label: Dict[str, Counter],
    output_dir: Path,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    outputs: Dict[str, Any] = {}
    for name in (OUTPUT_CANDIDATES, OUTPUT_VERIFIED, OUTPUT_REVIEW_QUEUE, OUTPUT_LABEL_MAPPING):
        target = output_dir / name
        if target.is_file():
            outputs[name] = {"sha256": _sha256(target), "bytes": target.stat().st_size}

    labelled = [r for r in records if r["lexical_labels"]]
    status_counts = Counter(r["review_status"] for r in records)
    label_sets = Counter(";".join(r["lexical_labels"]) for r in labelled)

    provenance = {
        "generated_by": "python scripts/16_label_lexicon.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dictionary_version": cfg.dictionary_version,
        "source_split": SOURCE_SPLIT,
        "parameters": cfg.to_dict(),
        "inputs": _input_fingerprints(),
        "rows_by_label": dict(sorted(rows_by_label.items())),
        "rows_by_dataset_and_label": {
            ds: dict(sorted(counter.items())) for ds, counter in sorted(rows_by_dataset_label.items())
        },
        "candidates_total": len(records),
        "entries_with_label": len(labelled),
        "status_counts": dict(sorted(status_counts.items())),
        "label_sets": dict(sorted(label_sets.items())),
        "manual_overrides": {
            "file": MANUAL_REVIEW_FILE if overrides else None,
            "applied": len(overrides) if overrides else 0,
            "terms": sorted(overrides.keys()) if overrides else [],
        },
        "outputs": outputs,
        "deviations": [
            "Candidate discovery (scripts/15_extract_corpus_words.py) scanned train+dev+test of every "
            "dataset; plan section 36.1 asks for train-only extraction. Labels below are computed on "
            "train rows only - re-extraction from train is a follow-up before the dictionary is frozen.",
        ],
    }
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(provenance, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    LOGGER.info("[labeler] wrote %s", path.name)


def run_labeling(
    candidates_path: str | Path = LEXICON_PROCESSED_DIR / CANDIDATES_INPUT,
    output_dir: str | Path = LEXICON_PROCESSED_DIR,
    cfg: Optional[LabelingConfig] = None,
    overrides_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Full pipeline: load -> evidence -> label -> overrides -> write -> validate."""
    cfg = cfg or LabelingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(candidates_path)
    rows = load_training_rows()
    evidence, term_datasets, rows_by_label, rows_by_dataset_label = collect_evidence(candidates, rows)
    records = label_all(candidates, evidence, term_datasets, rows_by_label, cfg)
    for record in records:
        record["dictionary_version"] = cfg.dictionary_version

    overrides_file = Path(overrides_path) if overrides_path else output_dir / MANUAL_REVIEW_FILE
    overrides = load_overrides(overrides_file)
    apply_overrides(records, overrides, cfg)

    _write_candidates_csv(output_dir / OUTPUT_CANDIDATES, records)
    _write_verified_csv(output_dir / OUTPUT_VERIFIED, records)
    queue_size = _write_review_queue(output_dir / OUTPUT_REVIEW_QUEUE, records)
    _write_label_mapping(output_dir / OUTPUT_LABEL_MAPPING)
    _write_provenance(
        output_dir / OUTPUT_PROVENANCE, cfg, records, rows_by_label, rows_by_dataset_label, output_dir, overrides
    )

    validation = validate_lexicon(output_dir)

    labelled = [r for r in records if r["lexical_labels"]]
    summary = {
        "candidates": len(records),
        "labelled": len(labelled),
        "review_queue": queue_size,
        "status_counts": dict(Counter(r["review_status"] for r in records)),
        "label_sets": dict(Counter(";".join(r["lexical_labels"]) for r in labelled)),
        "validation": validation,
    }
    LOGGER.info("[labeler] summary: %s", {k: v for k, v in summary.items() if k != "validation"})
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.lexical_labeler",
        description="Label the extracted candidates with lexical HATE / OFFENSIVE labels (plan section 36).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", default=str(LEXICON_PROCESSED_DIR / CANDIDATES_INPUT), help="Input candidate list.")
    p.add_argument("--output-dir", default=str(LEXICON_PROCESSED_DIR), help="Where the lexicon files are written.")
    p.add_argument("--min-evidence", type=int, default=3, help="Minimum toxic rows before a label is assigned.")
    p.add_argument("--share-threshold", type=float, default=0.6, help="Dominant-class share needed for a single label.")
    p.add_argument("--clean-lift", type=float, default=3.0, help="Minimum toxic-vs-clean rate ratio.")
    p.add_argument("--clean-count-cap", type=int, default=20, help="Veto a vote when the term appears in >= N clean rows.")
    p.add_argument("--strong-lift", type=float, default=15.0, help="Lift that overrides the clean-count cap.")
    p.add_argument("--overrides", default=None, help="CSV with manual decisions (default: <output-dir>/manual_review.csv).")
    p.add_argument("--verified-confidence", type=float, default=0.7, help="Confidence needed for review_status=verified.")
    p.add_argument("--verified-evidence", type=int, default=10, help="Evidence needed for review_status=verified.")
    p.add_argument("--auto-confidence", type=float, default=0.4, help="Confidence floor for auto labelling.")
    p.add_argument("--dictionary-version", default=DICTIONARY_VERSION_DEFAULT, help="Dictionary version stamp.")
    p.add_argument("--validate-only", action="store_true", help="Only validate the existing outputs.")
    p.add_argument("--show-terms", type=int, default=15, help="Print N sample labelled entries (0 = none).")
    return p


def _ensure_utf8_console() -> None:
    """Make Vietnamese output survive a cp1252 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - exotic streams
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_console()
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)

    if args.validate_only:
        summary = validate_lexicon(output_dir)
        return 1 if summary["problems"] else 0

    cfg = LabelingConfig(
        min_evidence=args.min_evidence,
        share_threshold=args.share_threshold,
        clean_lift=args.clean_lift,
        clean_count_cap=args.clean_count_cap,
        strong_lift=args.strong_lift,
        verified_confidence=args.verified_confidence,
        verified_evidence=args.verified_evidence,
        auto_confidence=args.auto_confidence,
        dictionary_version=args.dictionary_version,
    )
    summary = run_labeling(Path(args.candidates), output_dir, cfg, overrides_path=args.overrides)

    LOGGER.info(
        "[labeler] %d candidates -> %d labelled, %d in review queue",
        summary["candidates"],
        summary["labelled"],
        summary["review_queue"],
    )
    if args.show_terms > 0:
        path = output_dir / OUTPUT_VERIFIED
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows[: args.show_terms]:
            LOGGER.info(
                "[labeler]   %s | %-16s | %-16s | sev %s | conf %s | %s",
                row["entry_id"],
                row["term"],
                row["lexical_labels"],
                row["severity"],
                row["confidence"],
                row["review_status"],
            )
    if summary["validation"]["problems"]:
        return 1
    LOGGER.info("[labeler] done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
