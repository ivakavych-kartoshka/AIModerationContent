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

VALID_LABELS = ("HATE", "OFFENSIVE", "CLEAN")
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
    # Per-label certainty (must cover all 3 labels for fusion)
    label_severity: Dict[str, float] = None
    label_confidence: Dict[str, float] = None


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
                base_sev = float(row.get("severity") or 0.0)
                base_conf = float(row.get("confidence") or 0.0)
                # Parse per-label severity/confidence if present (comma/semicolon separated)
                label_sev: Dict[str, float] = {}
                label_conf: Dict[str, float] = {}
                raw_sev_map = row.get("label_severities") or row.get("severity_by_label") or ""
                raw_conf_map = row.get("label_confidences") or row.get("confidence_by_label") or ""
                def _parse_map(raw: str) -> Dict[str, float]:
                    out: Dict[str, float] = {}
                    if not raw:
                        return out
                    for part in raw.replace(";", ",").split(","):
                        if not part.strip():
                            continue
                        if ":" in part:
                            k, v = part.split(":", 1)
                        elif "=" in part:
                            k, v = part.split("=", 1)
                        else:
                            continue
                        k = k.strip().upper()
                        try:
                            out[k] = float(v.strip())
                        except Exception:
                            continue
                    return out

                label_sev = _parse_map(raw_sev_map)
                label_conf = _parse_map(raw_conf_map)
                # Ensure all three labels present with derived values (no hardcoded 0.9/0.6)
                for lab in ("HATE", "OFFENSIVE", "CLEAN"):
                    if lab not in label_sev:
                        if lab in labels and base_sev > 0:
                            label_sev[lab] = float(base_sev)
                        elif lab == "CLEAN":
                            # CLEAN derives from absence of toxic certainty; compute from base
                            label_sev[lab] = max(0.0, min(1.0, 1.0 - max(label_sev.get("HATE", 0.0), label_sev.get("OFFENSIVE", 0.0))))
                        else:
                            label_sev[lab] = float(base_sev) if base_sev > 0 else float(base_conf)
                    if lab not in label_conf:
                        if lab in labels and base_conf > 0:
                            label_conf[lab] = float(base_conf)
                        elif lab == "CLEAN":
                            label_conf[lab] = max(0.0, min(1.0, 1.0 - max(label_conf.get("HATE", 0.0), label_conf.get("OFFENSIVE", 0.0))))
                        else:
                            label_conf[lab] = float(base_conf) if base_conf > 0 else float(label_sev.get(lab, 0.0))

                entries.append(
                    LexiconEntry(
                        entry_id=row["entry_id"],
                        term=row["term"],
                        canonical_form=row["canonical_form"] or row["term"],
                        labels=labels,
                        severity=float(base_sev),
                        confidence=float(base_conf),
                        context_required=str(row["context_required"]).strip().lower() in {"true", "1", "yes"},
                        source_datasets=tuple(
                            part for part in (row["source_datasets"] or "").split(";") if part
                        ),
                        review_status=status,
                        dictionary_version=str(row["dictionary_version"]).strip(),
                        label_severity=label_sev,
                        label_confidence=label_conf,
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
