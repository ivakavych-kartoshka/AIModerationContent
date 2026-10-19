"""Re-label the pruned lexicon with the lexical labelling algorithm (plan 36).

``scripts/20_prune_lexicon.py`` removes the noise the TF-IDF gate rejects, but it
only *drops* rows: every surviving entry keeps the labels it carried when the
dictionary was frozen. This module closes the loop: it takes the **already
pruned** dictionary, runs the full lexical labelling algorithm
(:func:`src.data.lexical_labeler.label_term`) on exactly those terms, and writes
a freshly labelled copy.

Evidence is pooled over **all** training datasets (the same corpus step 16 uses,
section 36.1: only train splits). Because the candidate set is the pruned one,
terms the algorithm can no longer label (weak evidence, clean-dominated, below
the share/lift thresholds) are reported in the audit instead of silently kept.

Outputs (written to ``data/processed/lexicon/relabeling/``; the input is read
from ``data/processed/lexicon/pruning/`` and ``lexicon_report.json`` is updated
at the lexicon root)
------------------------------------------------
``verified_lexicon_relabeled.csv``            the re-labelled dictionary (same
                                              schema as ``verified_lexicon.csv``)
``verified_lexicon_relabeled_<dataset>.csv``  per-dataset copies, split on
                                              ``source_datasets``; their union is
                                              the combined file
``lexicon_relabel_changes.csv``               one row per input term comparing
                                              the old vs new labels / status
``lexicon_report.json``                       gains a ``relabeling`` section

Usage
-----
    python -m src.data.lexicon_relabeler
    python -m src.data.lexicon_relabeler --share-threshold 0.7
    python -m src.data.lexicon_relabeler --no-tfidf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..common.constants import LEXICON_PROCESSED_DIR, LEXICON_SUBDIRS
from ..common.logging_utils import get_logger
from .lexical_labeler import LabelingConfig, collect_evidence, label_term, load_training_rows
from .lexicon import OUTPUT_REPORT, _sha256, normalize_term
from .lexicon_pruner import (
    LEXICON_FIELDS,
    OUTPUT_PRUNED,
    _dataset_slug,
    _datasets_of,
    _write_rows,
    load_lexicon_rows,
    pruning_dir,
)

LOGGER = get_logger("sensitiveai.lexicon_relabeler")

INPUT_PRUNED = OUTPUT_PRUNED
OUTPUT_RELABELED = "verified_lexicon_relabeled.csv"
OUTPUT_RELABEL_AUDIT = "lexicon_relabel_changes.csv"
#: per-dataset copies of the re-labelled dictionary: <prefix><dataset>.csv
PER_DATASET_PREFIX = "verified_lexicon_relabeled_"
UNKNOWN_DATASET = "unknown"
SOURCE_SPLIT = "train"

#: Every re-label output lives in ``<lexicon dir>/relabeling``; the report and
#: the active dictionary stay at the lexicon root.
RELABELING_SUBDIR = LEXICON_SUBDIRS["relabeling"]


def relabeling_dir(output_dir: str | Path) -> Path:
    """``<lexicon dir>/relabeling`` - home of the re-label outputs."""
    return Path(output_dir) / RELABELING_SUBDIR

AUDIT_FIELDS = (
    "entry_id",
    "term",
    "old_lexical_labels",
    "new_lexical_labels",
    "old_review_status",
    "new_review_status",
    "old_confidence",
    "new_confidence",
    "old_severity",
    "new_severity",
    "source_datasets",
    "changed",
    "reason",
)


def _unique_terms(rows: Sequence[Dict[str, str]]) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for row in rows:
        term = normalize_term(row.get("canonical_form") or row.get("term") or "")
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _relabelled_row(record: Dict[str, Any], entry_id: str) -> Dict[str, Any]:
    return {
        "entry_id": entry_id,
        "term": record["term"],
        "canonical_form": record["term"],
        "lexical_labels": ";".join(record["lexical_labels"]),
        "severity": record["severity"],
        "confidence": record["confidence"],
        "context_required": str(bool(record["context_required"])).lower(),
        "source_datasets": record["source_datasets"],
        "source_split": SOURCE_SPLIT,
        "review_status": record["review_status"],
        "dictionary_version": record.get("dictionary_version", "v1"),
    }


def relabel_lexicon(
    lexicon_path: str | Path = None,
    output_dir: str | Path = LEXICON_PROCESSED_DIR,
    cfg: Optional[LabelingConfig] = None,
    split_by_dataset: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Re-run the labelling algorithm over the pruned dictionary.

    The combined dictionary is written to ``verified_lexicon_relabeled.csv``;
    when ``split_by_dataset`` is set each source dataset gets its own copy. The
    audit records every input term, including those the algorithm would no longer
    label.
    """
    cfg = cfg or LabelingConfig()
    lexicon_path = (
        Path(lexicon_path) if lexicon_path else pruning_dir(LEXICON_PROCESSED_DIR) / INPUT_PRUNED
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_lexicon_rows(lexicon_path)
    old_by_term = {
        normalize_term(row.get("canonical_form") or row.get("term") or ""): row
        for row in rows
    }
    terms = _unique_terms(rows)
    LOGGER.info(
        "[relabel] loaded %d entries (%d unique terms) from %s",
        len(rows),
        len(terms),
        lexicon_path.name,
    )

    train_rows = load_training_rows()
    evidence, term_datasets, rows_by_label, _by_ds = collect_evidence(terms, train_rows)
    records = [
        label_term(
            term,
            evidence.get(term, Counter()),
            term_datasets.get(term, set()),
            rows_by_label,
            cfg,
        )
        for term in terms
    ]

    labelled = sorted((r for r in records if r["lexical_labels"]), key=lambda r: r["term"])
    relabelled_rows = [
        _relabelled_row(record, f"lex_{index:04d}") for index, record in enumerate(labelled, start=1)
    ]
    row_by_term = {row["term"]: row for row in relabelled_rows}

    audit_rows: List[Dict[str, Any]] = []
    changed = 0
    by_status: Counter = Counter()
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    dropped_by_reason: Counter = Counter()
    for record in sorted(records, key=lambda r: r["term"]):
        old = old_by_term.get(record["term"], {})
        old_labels = old.get("lexical_labels", "")
        new_labels = ";".join(record["lexical_labels"])
        old_status = old.get("review_status", "")
        new_status = record["review_status"]
        is_changed = (old_labels != new_labels) or (old_status != new_status)
        if is_changed:
            changed += 1
        audit_rows.append(
            {
                "entry_id": old.get("entry_id", ""),
                "term": record["term"],
                "old_lexical_labels": old_labels,
                "new_lexical_labels": new_labels,
                "old_review_status": old_status,
                "new_review_status": new_status,
                "old_confidence": old.get("confidence", ""),
                "new_confidence": record["confidence"],
                "old_severity": old.get("severity", ""),
                "new_severity": record["severity"],
                "source_datasets": record["source_datasets"],
                "changed": str(is_changed).lower(),
                "reason": record["reason"],
            }
        )
        if record["lexical_labels"]:
            by_status[new_status] += 1
            for dataset in _datasets_of(record):
                by_dataset.setdefault(dataset, []).append(row_by_term[record["term"]])
        else:
            dropped_by_reason[record["reason"]] += 1

    summary: Dict[str, Any] = {
        "generated_by": "python scripts/22_relabel_lexicon.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(lexicon_path),
        "input_terms": len(terms),
        "relabelled": len(relabelled_rows),
        "dropped_by_algorithm": len(terms) - len(relabelled_rows),
        "changed": changed,
        "unchanged": len(terms) - changed,
        "relabelled_by_review_status": dict(sorted(by_status.items())),
        "dropped_by_reason": dict(sorted(dropped_by_reason.items())),
        "relabelled_by_dataset": {name: len(items) for name, items in sorted(by_dataset.items())},
        "split_by_dataset": split_by_dataset,
        "parameters": cfg.to_dict(),
    }

    if dry_run:
        LOGGER.info(
            "[relabel] dry-run: %d/%d terms re-labelled, %d changed, dropped %s",
            summary["relabelled"],
            summary["input_terms"],
            changed,
            summary["dropped_by_reason"],
        )
        return summary

    out_dir = relabeling_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    relabelled_path = out_dir / OUTPUT_RELABELED
    _write_rows(relabelled_path, LEXICON_FIELDS, relabelled_rows)
    audit_path = out_dir / OUTPUT_RELABEL_AUDIT
    _write_rows(audit_path, AUDIT_FIELDS, audit_rows)
    summary["outputs"] = {
        OUTPUT_RELABELED: {"terms": len(relabelled_rows), "sha256": _sha256(relabelled_path)},
        OUTPUT_RELABEL_AUDIT: {"rows": len(audit_rows), "sha256": _sha256(audit_path)},
    }
    LOGGER.info(
        "[relabel] wrote %s (%d/%d re-labelled, %d changed) and %s",
        relabelled_path.name,
        len(relabelled_rows),
        len(terms),
        changed,
        audit_path.name,
    )

    if split_by_dataset and by_dataset:
        summary["per_dataset_outputs"] = {}
        for dataset, items in sorted(by_dataset.items()):
            name = f"{PER_DATASET_PREFIX}{_dataset_slug(dataset)}.csv"
            path = out_dir / name
            _write_rows(path, LEXICON_FIELDS, items)
            summary["outputs"][name] = {"terms": len(items), "sha256": _sha256(path)}
            summary["per_dataset_outputs"][dataset] = {"file": name, "terms": len(items)}
            LOGGER.info("[relabel]   %s: %d terms -> %s", dataset, len(items), name)

    _update_report(output_dir, summary)
    return summary


def _update_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    report_path = output_dir / OUTPUT_REPORT
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"generated_by": "python scripts/12_build_lexicon.py", "outputs": {}, "counts": {}}
    report["relabeling"] = {
        "parameters": summary["parameters"],
        "input": summary["input"],
        "input_terms": summary["input_terms"],
        "relabelled": summary["relabelled"],
        "dropped_by_algorithm": summary["dropped_by_algorithm"],
        "changed": summary["changed"],
        "unchanged": summary["unchanged"],
        "relabelled_by_review_status": summary["relabelled_by_review_status"],
        "relabelled_by_dataset": summary["relabelled_by_dataset"],
        "dropped_by_reason": summary["dropped_by_reason"],
        "per_dataset_outputs": summary.get("per_dataset_outputs", {}),
    }
    outputs = report.setdefault("outputs", {})
    for name, info in summary.get("outputs", {}).items():
        outputs[name] = {
            "terms": info.get("terms", info.get("rows")),
            "sha256": info.get("sha256"),
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("[relabel] updated %s", report_path.name)


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - exotic streams
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.lexicon_relabeler",
        description="Re-label the pruned lexicon with the lexical labelling algorithm (plan section 36).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--lexicon",
        default=str(pruning_dir(LEXICON_PROCESSED_DIR) / INPUT_PRUNED),
        help="Input (pruned) lexicon CSV.",
    )
    p.add_argument("--output-dir", default=str(LEXICON_PROCESSED_DIR), help="Where the re-labelled lexicon is written.")
    p.add_argument("--no-tfidf", action="store_true", help="Disable the TF-IDF gate while re-labelling.")
    p.add_argument("--tfidf-min-ratio", type=float, default=0.6, help="Minimum toxic share of TF-IDF weight.")
    p.add_argument("--tfidf-min-score", type=float, default=0.004, help="Minimum IDF-weighted toxic-vs-clean separation.")
    p.add_argument("--tfidf-smoothing", type=float, default=1.0, help="Add-k smoothing inside the IDF.")
    p.add_argument("--share-threshold", type=float, default=0.6, help="Minimum class share of a vote.")
    p.add_argument("--clean-lift", type=float, default=3.0, help="Minimum lift over CLEAN rows.")
    p.add_argument("--min-evidence", type=int, default=3, help="Minimum toxic rows to consider a term.")
    p.add_argument("--no-per-dataset", action="store_true", help="Only write the combined file, skip per-dataset copies.")
    p.add_argument("--dry-run", action="store_true", help="Report the decisions without writing files.")
    return p


def _build_cfg(args: argparse.Namespace) -> LabelingConfig:
    from .tfidf_filter import TfidfConfig

    return LabelingConfig(
        min_evidence=args.min_evidence,
        share_threshold=args.share_threshold,
        clean_lift=args.clean_lift,
        tfidf=TfidfConfig(
            enabled=not args.no_tfidf,
            min_ratio=args.tfidf_min_ratio,
            min_score=args.tfidf_min_score,
            smoothing=args.tfidf_smoothing,
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_console()
    args = build_parser().parse_args(argv)
    summary = relabel_lexicon(
        lexicon_path=Path(args.lexicon),
        output_dir=Path(args.output_dir),
        cfg=_build_cfg(args),
        split_by_dataset=not args.no_per_dataset,
        dry_run=args.dry_run,
    )
    LOGGER.info(
        "[relabel] %d/%d terms re-labelled by dataset %s; %d changed; dropped: %s",
        summary["relabelled"],
        summary["input_terms"],
        summary["relabelled_by_dataset"],
        summary["changed"],
        summary["dropped_by_reason"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
