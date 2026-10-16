"""Load the labelled lexicon produced by ``scripts/16_label_lexicon.py``.

``verified_lexicon.csv`` follows the schema of section 36.2. Only entries whose
``review_status`` is allowed by the rule-engine config take part in matching, so
ambiguous rows (``needs_review``) never block a comment unless a human promotes
them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..common.logging_utils import get_logger

LOGGER = get_logger("sensitiveai.rule_engine.lexicon")

VALID_LABELS = ("HATE", "OFFENSIVE")
STATUS_ORDER = ("verified", "auto", "needs_review", "rejected")
REQUIRED_COLUMNS = (
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


@dataclass(frozen=True)
class LexiconEntry:
    entry_id: str
    term: str
    canonical_form: str
    labels: Tuple[str, ...]
    severity: float
    confidence: float
    context_required: bool
    source_datasets: Tuple[str, ...]
    review_status: str
    dictionary_version: str


class Lexicon:
    """In-memory lexicon with fast term lookup."""

    def __init__(self, entries: Sequence[LexiconEntry], path: Path | None = None) -> None:
        self.entries: List[LexiconEntry] = list(entries)
        self.path = path
        self.by_term: Dict[str, LexiconEntry] = {entry.term: entry for entry in self.entries}
        self.terms: List[str] = sorted(self.by_term, key=len, reverse=True)
        versions = {entry.dictionary_version for entry in self.entries}
        self.dictionary_version: str = next(iter(versions)) if len(versions) == 1 else "mixed"

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, term: str) -> LexiconEntry | None:
        return self.by_term.get(term)

    def by_labels(self, label: str) -> List[LexiconEntry]:
        return [entry for entry in self.entries if label in entry.labels]

    @classmethod
    def load(
        cls,
        path: str | Path,
        allowed_statuses: Iterable[str] = ("verified", "auto"),
    ) -> "Lexicon":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Lexicon not found: {path}")
        allowed = set(allowed_statuses)
        entries: List[LexiconEntry] = []
        skipped = 0
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            for row in reader:
                status = (row["review_status"] or "").strip()
                if status not in allowed:
                    skipped += 1
                    continue
                labels = tuple(
                    label for label in (row["lexical_labels"] or "").replace(",", ";").split(";") if label
                )
                if not labels or any(label not in VALID_LABELS for label in labels):
                    skipped += 1
                    continue
                string_labels = (row["lexical_labels"] or "").replace(",", ";")
                entries.append(
                    LexiconEntry(
                        entry_id=row["entry_id"],
                        term=row["term"],
                        canonical_form=row["canonical_form"] or row["term"],
                        labels=labels,
                        severity=float(row["severity"]),
                        confidence=float(row["confidence"]),
                        context_required=str(row["context_required"]).strip().lower() in {"true", "1", "yes"},
                        source_datasets=tuple(
                            part for part in (row["source_datasets"] or "").split(";") if part
                        ),
                        review_status=status,
                        dictionary_version=str(row["dictionary_version"]).strip(),
                    )
                )
                if string_labels and row.get("source_split") != "train":
                    LOGGER.debug("[lexicon] %s does not come from the train split", row["entry_id"])

        LOGGER.info(
            "[lexicon] loaded %d entries from %s (allowed statuses=%s, skipped=%d)",
            len(entries),
            path.name,
            sorted(allowed),
            skipped,
        )
        return cls(entries, path)
