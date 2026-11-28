"""Mine a rule-engine lexicon from the English **train split only**.

The rule engine already defines the schema (``verified_lexicon.csv``); this
module produces the same columns from data statistics so the English pipeline
can be built exactly like the Vietnamese one:

    df(term, label)  ->  precision / lift vs. the class prior
    label kept when  precision >= --min-precision, lift >= --min-lift, df >= --min-count

Only ``data/splits_en/train.csv`` is read.  Validation and test are never used,
matching the project's leakage protocol.

Output (default):
    data/processed/lexicon/verified_lexicon_en_extracted.csv
    data/processed/lexicon/english/lexicon_report.json
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from ..common.constants import (
    HATE_SPEECH_OFFENSIVE_SPLITS_DIR,
    ID2LABEL,
    LABEL_COLUMN,
    LEXICON_PROCESSED_DIR,
    TEXT_COLUMN,
)
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

WORD_RE = re.compile(r"[a-z][a-z'\u2019\-]{2,}")
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here
    i me my we our you your he him his she her it its they them their
    is are was were be been being am do does did have has had
    to of in on at for with from by about as into through during before after
    above below between out up down off over under again further once
    not no nor so too very can will just should now what which who whom whose
    when where why how all any both each few more most other some such only own
    same s t don didn doesn isn aren wasn weren won wouldn couldn shouldn
    im ive id ill youre youve youll hes shes its weve theyre theyve
    rt amp via http https www
    """.split()
)


def _wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound: stable precision estimate for small counts."""
    if n <= 0:
        return 0.0
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denominator)


def _terms(text: str, include_bigrams: bool = False) -> Set[str]:
    words = [w for w in WORD_RE.findall(str(text).lower()) if w not in STOPWORDS]
    terms: Set[str] = set(words)
    if include_bigrams:
        for left, right in zip(words, words[1:]):
            terms.add(f"{left} {right}")
    return terms


def extract_lexicon(
    train_csv: str | Path = HATE_SPEECH_OFFENSIVE_SPLITS_DIR / "train.csv",
    output_csv: str | Path = LEXICON_PROCESSED_DIR / "verified_lexicon_en_extracted.csv",
    report_path: str | Path = LEXICON_PROCESSED_DIR / "english" / "lexicon_report.json",
    min_count: int = 15,
    min_lift: float = 2.0,
    min_hate_precision: float = 0.40,
    min_offensive_precision: float = 0.85,
    min_clean_precision: float = 0.60,
    min_wilson: float = 0.35,
    min_prior_margin: float = 0.08,
    max_terms: int = 1500,
    include_bigrams: bool = False,
    include_clean: bool = False,
    merged_version: str = "v1-en-combined",
    merge_with: str | Path | None = None,
) -> Dict[str, object]:
    train_csv = Path(train_csv)
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Missing {train_csv}. Run 'python scripts/52_prepare_english_data.py' first."
        )
    frame = pd.read_csv(train_csv, dtype={TEXT_COLUMN: "string"})
    labels = frame[LABEL_COLUMN].astype(int).tolist()
    n_docs = len(frame)
    prior = {label_id: labels.count(label_id) / max(n_docs, 1) for label_id in range(len(ID2LABEL))}

    df_count: Counter = Counter()
    label_count: Dict[str, Counter] = {str(label_id): Counter() for label_id in range(len(ID2LABEL))}
    for text, label_id in zip(frame[TEXT_COLUMN].tolist(), labels):
        seen = _terms(text, include_bigrams=include_bigrams)
        for term in seen:
            df_count[term] += 1
            label_count[str(label_id)][term] += 1

    rows: List[Dict[str, object]] = []
    for term, df in df_count.items():
        if df < min_count:
            continue
        if include_bigrams and " " in term and df < min_count * 2:
            # phrases need stronger evidence than single tokens
            continue
        stats = {}
        keep: List[int] = []
        for label_id in range(len(ID2LABEL)):
            count = label_count[str(label_id)][term]
            precision = count / df
            lift = precision / max(prior[label_id], 1e-9)
            stats[label_id] = (count, precision, lift, _wilson_lower(count, df))

        def _keep(label_id: int, floor: float) -> bool:
            count, precision, lift, wilson = stats[label_id]
            if precision < floor or wilson < min_wilson:
                return False
            # The Wilson lower bound (not the raw share) must clear the class
            # prior by a margin.  This is what removes high-frequency generic
            # words: with n=87 and p=0.86 the lower bound barely reaches the
            # 0.77 OFFENSIVE prior, so the term carries no usable signal.
            if wilson < prior[label_id] + min_prior_margin:
                return False
            # Lift alone is unusable for the dominant class (prior ~= 0.77);
            # require it only where the class is not already the majority.
            return True if label_id == 1 else lift >= min_lift

        for label_id, floor in (
            (0, min_hate_precision),
            (1, min_offensive_precision),
            (2, min_clean_precision),
        ):
            if _keep(label_id, floor):
                keep.append(label_id)
        if not include_clean:
            # CLEAN entries would only lower ``rule_clean_score`` for benign
            # words, so the rule engine consumes toxic entries only by default.
            keep = [label_id for label_id in keep if label_id != 2]
        if not keep:
            continue

        # Severity = precision (the share of term occurrences that really carry
        # this label).  Confidence grows with document frequency.
        label_sev: Dict[str, float] = {}
        label_conf: Dict[str, float] = {}
        reliability = min(0.99, 0.50 + 0.50 * (1.0 - math.exp(-df / 20.0)))
        for label_id in range(len(ID2LABEL)):
            label_name = ID2LABEL[label_id]
            if label_id in keep:
                label_sev[label_name] = round(min(0.99, max(0.05, stats[label_id][1])), 4)
                label_conf[label_name] = round(reliability, 4)
            elif label_id == 2:
                toxic = max(stats[0][1], stats[1][1])
                label_sev["CLEAN"] = round(min(0.99, max(0.01, 1.0 - toxic)), 4)
                label_conf["CLEAN"] = round(reliability, 4)
            else:
                label_sev[label_name] = 0.02
                label_conf[label_name] = round(min(0.7, reliability * 0.5), 4)

        primary = max(keep, key=lambda lid: (stats[lid][2], stats[lid][0]))
        rows.append(
            {
                "entry_id": "",
                "term": term,
                "canonical_form": term,
                "lexical_labels": ";".join(ID2LABEL[lid] for lid in sorted(keep)),
                "severity": round(stats[primary][1], 4),
                "confidence": round(reliability, 4),
                "context_required": "true" if any(stats[lid][1] < 0.80 for lid in keep) else "false",
                "source_datasets": "davidson",
                "source_split": "train",
                "review_status": "auto",
                "dictionary_version": "v1-en-extracted",
                "label_severities": ";".join(f"{k}:{v}" for k, v in label_sev.items()),
                "label_confidences": ";".join(f"{k}:{v}" for k, v in label_conf.items()),
                "document_frequency": int(df),
                "HATE_precision": round(stats[0][1], 4),
                "OFFENSIVE_precision": round(stats[1][1], 4),
                "CLEAN_precision": round(stats[2][1], 4),
                "HATE_lift": round(stats[0][2], 3),
                "OFFENSIVE_lift": round(stats[1][2], 3),
                "CLEAN_lift": round(stats[2][2], 3),
            }
        )

    rows.sort(key=lambda row: (max(float(row["HATE_lift"]), float(row["OFFENSIVE_lift"])), row["document_frequency"]), reverse=True)
    rows = rows[: max_terms]
    for index, row in enumerate(rows, start=1):
        row["entry_id"] = f"enx_{index:04d}"

    existing_terms: Set[str] = set()
    merged_rows: List[Dict[str, object]] = []
    if merge_with and Path(merge_with).is_file():
        with Path(merge_with).open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing_terms.add(str(row.get("term", "")))
                merged = dict(row)
                merged["dictionary_version"] = merged_version
                merged_rows.append(merged)
    for row in rows:
        if row["term"] in existing_terms:
            continue
        row["dictionary_version"] = merged_version
        merged_rows.append(row)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "entry_id", "term", "canonical_form", "lexical_labels", "severity", "confidence",
        "context_required", "source_datasets", "source_split", "review_status",
        "dictionary_version", "label_severities", "label_confidences",
        "document_frequency", "HATE_precision", "OFFENSIVE_precision", "CLEAN_precision",
        "HATE_lift", "OFFENSIVE_lift", "CLEAN_lift",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_split": "train only",
        "n_documents": n_docs,
        "class_prior": {ID2LABEL[k]: round(v, 6) for k, v in prior.items()},
        "parameters": {
            "min_count": min_count,
            "min_lift": min_lift,
            "min_hate_precision": min_hate_precision,
            "min_offensive_precision": min_offensive_precision,
            "min_clean_precision": min_clean_precision,
            "min_wilson": min_wilson,
            "min_prior_margin": min_prior_margin,
            "max_terms": max_terms,
            "include_bigrams": include_bigrams,
            "include_clean": include_clean,
        },
        "n_extracted": len(rows),
        "n_written": len(merged_rows),
        "merged_seed_terms": len(existing_terms),
        "output_csv": str(output_csv),
        "top_hate": [r["term"] for r in rows if "HATE" in str(r["lexical_labels"])][:20],
        "top_offensive": [r["term"] for r in rows if "OFFENSIVE" in str(r["lexical_labels"])][:20],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(
        "[lexicon-en] extracted=%d written=%d (seed=%d) -> %s",
        len(rows), len(merged_rows), len(existing_terms), output_csv,
    )
    return report


__all__ = ["extract_lexicon"]
