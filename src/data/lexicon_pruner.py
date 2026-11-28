"""TF-IDF pruning of the *frozen* lexicon (research plan, section 36).

``scripts/16_label_lexicon.py`` already runs the TF-IDF gate while labelling
candidates, so every ``auto`` / ``verified`` row of ``verified_lexicon.csv``
passed the gate at build time. The frozen dictionary can still carry noise:

* entries promoted by a human through ``manual_review.csv`` bypass the gate;
* terms merged in from an external lexicon are trusted blindly;
* editing the TF-IDF thresholds later does not retro-actively clean a
  dictionary that was frozen with older values.

This module re-scores **the whole final lexicon** against the training corpus,
applies the same :mod:`src.data.tfidf_filter` gate, and writes a pruned copy
plus a full audit of every decision. It reads **all** rows regardless of
``review_status`` so nothing escapes the audit, and only *training* splits are
scanned (section 36.1: validation/test rows must never touch the dictionary).

Outputs (written to ``data/processed/lexicon/pruning/``; ``lexicon_report.json``
is updated at the lexicon root)
------------------------------------------------
``verified_lexicon_pruned.csv``   same schema as ``verified_lexicon.csv`` minus
                                  the terms the gate rejects
``lexicon_prune_audit.csv``       one row per lexicon entry with its counts,
                                  IDF, TF-IDF numbers and the keep/drop reason
``verified_lexicon_preprune.csv`` backup written by ``--in-place``
``lexicon_report.json``           gains a ``tfidf_pruning`` section (root)

Usage
-----
    python -m src.data.lexicon_pruner
    python -m src.data.lexicon_pruner --tfidf-min-score 0.005
    python -m src.data.lexicon_pruner --in-place        # overwrite the dictionary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..common.constants import LEXICON_PROCESSED_DIR, LEXICON_SUBDIRS
from ..common.logging_utils import get_logger
from .lexical_labeler import collect_evidence, load_training_rows
from .lexicon import OUTPUT_REPORT, _sha256, normalize_term
from .tfidf_filter import TfidfConfig, TermTfidf, term_tfidf
from .tfidf_filter import passes as tfidf_passes

LOGGER = get_logger("sensitiveai.lexicon_pruner")

INPUT_VERIFIED = "verified_lexicon.csv"
OUTPUT_PRUNED = "verified_lexicon_pruned.csv"
OUTPUT_AUDIT = "lexicon_prune_audit.csv"
BACKUP_UNPRUNED = "verified_lexicon_preprune.csv"
#: per-dataset copies of the pruned dictionary: <prefix><dataset>.csv
PER_DATASET_PREFIX = "verified_lexicon_pruned_"
UNKNOWN_DATASET = "unknown"

LEXICON_FIELDS = (
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
)

#: Every prune output (pruned dictionary, per-dataset copies, audit, backup)
#: lives in ``<lexicon dir>/pruning``; the report and the active dictionary stay
#: at the lexicon root.
PRUNING_SUBDIR = LEXICON_SUBDIRS["pruning"]


def pruning_dir(output_dir: str | Path) -> Path:
    """``<lexicon dir>/pruning`` - home of the TF-IDF prune outputs."""
    return Path(output_dir) / PRUNING_SUBDIR

AUDIT_FIELDS = (
    "entry_id",
    "term",
    "lexical_labels",
    "review_status",
    "n_hate",
    "n_offensive",
    "n_clean",
    "df_total",
    "idf",
    "tfidf_hate",
    "tfidf_offensive",
    "tfidf_clean",
    "tfidf_toxic",
    "tfidf_ratio",
    "tfidf_score",
    "decision",
    "reason",
)


def load_lexicon_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read every row of a lexicon CSV (any ``review_status``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"lexicon not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [column for column in LEXICON_FIELDS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return [dict(row) for row in reader]


def collect_term_stats(
    terms: Sequence[str],
    rows: Sequence[Tuple[str, str, str]],
    cfg: TfidfConfig,
) -> Tuple[Dict[str, TermTfidf], Counter]:
    """Score every term against the training rows with the shared TF-IDF maths.

    Returns ``(term -> TermTfidf, rows_by_label)``; terms absent from the corpus
    keep an all-zero :class:`~src.data.tfidf_filter.TermTfidf`.
    """
    evidence, _datasets, rows_by_label, _by_ds = collect_evidence(list(terms), rows)
    n_hate = max(int(rows_by_label.get("HATE", 0)), 1)
    n_offensive = max(int(rows_by_label.get("OFFENSIVE", 0)), 1)
    n_clean = max(int(rows_by_label.get("CLEAN", 0)), 1)

    stats: Dict[str, TermTfidf] = {}
    for term in terms:
        counts = evidence.get(term, Counter())
        tf_hate = int(counts.get("HATE", 0))
        tf_off = int(counts.get("OFFENSIVE", 0))
        tf_clean = int(counts.get("CLEAN", 0))
        stats[term] = term_tfidf(
            tf_hate=tf_hate,
            tf_offensive=tf_off,
            tf_clean=tf_clean,
            df_total=tf_hate + tf_off + tf_clean,
            n_hate=n_hate,
            n_offensive=n_offensive,
            n_clean=n_clean,
            cfg=cfg,
        )
    return stats, rows_by_label


def decide(
    stats: TermTfidf,
    cfg: TfidfConfig,
    keep_unseen: bool = False,
) -> Tuple[bool, str]:
    """Keep/drop decision for one frozen entry, mirroring the labelling gate."""
    if stats.df_total == 0:
        if keep_unseen:
            return True, "unseen_in_train_kept"
        return False, "unseen_in_train"
    return tfidf_passes(stats, cfg)


def _write_rows(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _audit_row(row: Dict[str, str], stats: TermTfidf, keep: bool, reason: str) -> Dict[str, Any]:
    audit = {
        "entry_id": row.get("entry_id", ""),
        "term": row.get("term", ""),
        "lexical_labels": row.get("lexical_labels", ""),
        "review_status": row.get("review_status", ""),
        "n_hate": stats.tf_hate,
        "n_offensive": stats.tf_offensive,
        "n_clean": stats.tf_clean,
        "df_total": stats.df_total,
        "decision": "kept" if keep else "dropped",
        "reason": reason,
    }
    audit.update(stats.to_row())
    return audit


def _datasets_of(row: Dict[str, str]) -> List[str]:
    """Dataset names recorded in a row's ``source_datasets`` (``;``-separated)."""
    raw = (row.get("source_datasets") or "").replace(",", ";")
    datasets = [part.strip() for part in raw.split(";") if part.strip()]
    return datasets or [UNKNOWN_DATASET]


def _dataset_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or UNKNOWN_DATASET


def prune_lexicon(
    lexicon_path: str | Path = LEXICON_PROCESSED_DIR / INPUT_VERIFIED,
    output_dir: str | Path = LEXICON_PROCESSED_DIR,
    cfg: Optional[TfidfConfig] = None,
    keep_unseen: bool = False,
    split_by_dataset: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Re-score the frozen lexicon and write the pruned copy + audit.

    The combined dictionary is always written to ``verified_lexicon_pruned.csv``.
    When ``split_by_dataset`` is set, a copy restricted to each source dataset is
    also written as ``verified_lexicon_pruned_<dataset>.csv``; a term that comes
    from several datasets appears in each of their files, so the union of the
    per-dataset files equals the combined file.
    """
    cfg = cfg or TfidfConfig()
    lexicon_path = Path(lexicon_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_lexicon_rows(lexicon_path)
    terms = [normalize_term(row["canonical_form"] or row["term"]) for row in rows]
    LOGGER.info("[pruner] loaded %d lexicon rows from %s", len(rows), lexicon_path.name)

    train_rows = load_training_rows()
    stats, rows_by_label = collect_term_stats(terms, train_rows, cfg)

    kept_rows: List[Dict[str, str]] = []
    audit_rows: List[Dict[str, Any]] = []
    dropped_by_status: Counter = Counter()
    kept_by_status: Counter = Counter()
    dropped_reasons: Counter = Counter()
    by_dataset: Dict[str, List[Dict[str, str]]] = {}

    for row, term in zip(rows, terms):
        stat = stats.get(term, term_tfidf(0, 0, 0, 0, 1, 1, 1, cfg))
        keep, reason = decide(stat, cfg, keep_unseen=keep_unseen)
        audit_rows.append(_audit_row(row, stat, keep, reason))
        if keep:
            kept_rows.append(row)
            kept_by_status[row.get("review_status", "?")] += 1
            for dataset in _datasets_of(row):
                by_dataset.setdefault(dataset, []).append(row)
        else:
            dropped_by_status[row.get("review_status", "?")] += 1
            dropped_reasons[reason] += 1

    summary: Dict[str, Any] = {
        "generated_by": "python scripts/20_prune_lexicon.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(lexicon_path),
        "input_terms": len(rows),
        "kept": len(kept_rows),
        "dropped": len(rows) - len(kept_rows),
        "dropped_by_review_status": dict(sorted(dropped_by_status.items())),
        "kept_by_review_status": dict(sorted(kept_by_status.items())),
        "dropped_by_reason": dict(sorted(dropped_reasons.items())),
        "kept_by_dataset": {name: len(items) for name, items in sorted(by_dataset.items())},
        "keep_unseen": keep_unseen,
        "split_by_dataset": split_by_dataset,
        "parameters": cfg.to_dict(),
        "rows_by_label": dict(sorted(rows_by_label.items())),
    }

    if dry_run:
        LOGGER.info(
            "[pruner] dry-run: kept %d / %d entries by dataset %s",
            summary["kept"],
            summary["input_terms"],
            summary["kept_by_dataset"],
        )
        return summary

    out_dir = pruning_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pruned_path = out_dir / OUTPUT_PRUNED
    _write_rows(pruned_path, LEXICON_FIELDS, kept_rows)
    audit_path = out_dir / OUTPUT_AUDIT
    _write_rows(audit_path, AUDIT_FIELDS, audit_rows)
    summary["outputs"] = {
        OUTPUT_PRUNED: {"terms": len(kept_rows), "sha256": _sha256(pruned_path)},
        OUTPUT_AUDIT: {"rows": len(audit_rows), "sha256": _sha256(audit_path)},
    }
    LOGGER.info(
        "[pruner] wrote %s (%d/%d kept) and %s",
        pruned_path.name,
        len(kept_rows),
        len(rows),
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
            LOGGER.info("[pruner]   %s: %d terms -> %s", dataset, len(items), name)

    _update_report(output_dir, summary)
    return summary


def _update_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    report_path = output_dir / OUTPUT_REPORT
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"generated_by": "python scripts/12_build_lexicon.py", "outputs": {}, "counts": {}}
    report["tfidf_pruning"] = {
        "parameters": summary["parameters"],
        "input_terms": summary["input_terms"],
        "kept": summary["kept"],
        "dropped": summary["dropped"],
        "keep_unseen": summary["keep_unseen"],
        "split_by_dataset": summary["split_by_dataset"],
        "kept_by_dataset": summary["kept_by_dataset"],
        "per_dataset_outputs": summary.get("per_dataset_outputs", {}),
        "dropped_by_review_status": summary["dropped_by_review_status"],
        "kept_by_review_status": summary["kept_by_review_status"],
        "dropped_by_reason": summary["dropped_by_reason"],
    }
    outputs = report.setdefault("outputs", {})
    for name, info in summary.get("outputs", {}).items():
        outputs[name] = {
            "terms": info.get("terms", info.get("rows")),
            "sha256": info.get("sha256"),
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("[pruner] updated %s", report_path.name)


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - exotic streams
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.lexicon_pruner",
        description="Re-score the frozen lexicon with the TF-IDF gate and write a pruned copy (plan section 36).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lexicon", default=str(LEXICON_PROCESSED_DIR / INPUT_VERIFIED), help="Input lexicon CSV.")
    p.add_argument("--output-dir", default=str(LEXICON_PROCESSED_DIR), help="Where the pruned lexicon is written.")
    p.add_argument("--no-tfidf", action="store_true", help="Disable the gate (audit only, drops nothing).")
    p.add_argument("--tfidf-min-ratio", type=float, default=0.6, help="Minimum toxic share of TF-IDF weight.")
    p.add_argument("--tfidf-min-score", type=float, default=0.004, help="Minimum IDF-weighted toxic-vs-clean separation.")
    p.add_argument("--tfidf-smoothing", type=float, default=1.0, help="Add-k smoothing inside the IDF.")
    p.add_argument("--keep-unseen", action="store_true", help="Keep terms that never appear in the training corpus.")
    p.add_argument("--no-per-dataset", action="store_true", help="Only write the combined file, skip per-dataset copies.")
    p.add_argument("--in-place", action="store_true", help="Overwrite verified_lexicon.csv (a backup is written).")
    p.add_argument("--dry-run", action="store_true", help="Report the decisions without writing files.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_console()
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    cfg = TfidfConfig(
        enabled=not args.no_tfidf,
        min_ratio=args.tfidf_min_ratio,
        min_score=args.tfidf_min_score,
        smoothing=args.tfidf_smoothing,
    )
    summary = prune_lexicon(
        lexicon_path=Path(args.lexicon),
        output_dir=output_dir,
        cfg=cfg,
        keep_unseen=args.keep_unseen,
        split_by_dataset=not args.no_per_dataset,
        dry_run=args.dry_run,
    )
    LOGGER.info(
        "[pruner] %d/%d entries kept by dataset %s; dropped: %s",
        summary["kept"],
        summary["input_terms"],
        summary["kept_by_dataset"],
        summary["dropped_by_reason"],
    )
    if args.in_place and not args.dry_run:
        src = output_dir / INPUT_VERIFIED
        backup = pruning_dir(output_dir) / BACKUP_UNPRUNED
        if src.is_file() and not backup.is_file():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(src.read_bytes())
            LOGGER.info("[pruner] backup -> %s", backup)
        pruned = pruning_dir(output_dir) / OUTPUT_PRUNED
        if pruned.is_file():
            src.write_bytes(pruned.read_bytes())
            LOGGER.info("[pruner] %s overwritten with the pruned dictionary", src.name)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
