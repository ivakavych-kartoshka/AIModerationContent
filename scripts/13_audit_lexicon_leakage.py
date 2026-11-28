"""Step 13 - lexicon-level leakage audit (test-split leakage).

Checks whether lexicon entries were mined from (or contain verbatim copies of)
the held-out test sets, per language.

    python scripts/13_audit_lexicon_leakage.py

Checks
-------
1. Provenance: ``source_split`` / ``source_datasets`` values (must not include
   ``test``/``validation``/``dev`` for mined entries).
2. Exact full-comment overlap: a lexicon term that equals (after normalisation)
   a held-out test comment.
3. Training-subset coverage: lexicon terms that never fire on the *training*
   text (a signal of test-mining).

Writes::
    outputs/evaluation/analysis/lexicon_leakage_report.json
    outputs/evaluation/analysis/lexicon_leakage_report.md
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.paths import project_relative  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
LEXICON_DIR = PROJECT_ROOT / "data" / "processed" / "lexicon"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "analysis"

BAD_SPLITS = {"test", "validation", "dev", "holdout"}


def _norm(text: str) -> str:
    """Lower-case, strip diacritics, collapse whitespace."""
    t = unicodedata.normalize("NFD", str(text or ""))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(_norm(text)) if len(t) > 1}


def audit(language: str, lexicon_path: Path, test_path: Path, train_path: Path) -> Dict[str, Any]:
    lexicon = pd.read_csv(lexicon_path, encoding="utf-8")
    test = pd.read_csv(test_path, encoding="utf-8")
    train = pd.read_csv(train_path, encoding="utf-8")

    split_col = "source_split" if "source_split" in lexicon.columns else None

    bad_split_rows = int(
        lexicon[lexicon[split_col].astype(str).isin(list(BAD_SPLITS))].shape[0]
    ) if split_col else 0

    train_text = {_norm(t) for t in train["text"].tolist()}
    test_text = {_norm(t) for t in test["text"].tolist()}
    train_tokens: Set[str] = set()
    for t in train["text"].tolist():
        train_tokens |= _tokens(t)
    test_tokens: Set[str] = set()
    for t in test["text"].tolist():
        test_tokens |= _tokens(t)

    terms = lexicon["term"].astype(str).str.strip()
    seen: Set[str] = set()
    exact_matches: List[str] = []
    only_in_test: List[str] = []
    only_in_train: List[str] = []
    never_fired_train: List[str] = []
    in_test_not_train: List[str] = []

    for term in terms:
        n = _norm(term)
        if not n or n in seen:
            continue
        seen.add(n)
        toks = _tokens(n)
        if n in test_text:
            exact_matches.append(term)
        fired_train = bool(toks & train_tokens) if toks else n in train_text
        fired_test = bool(toks & test_tokens) if toks else n in test_text
        if fired_test and not fired_train:
            in_test_not_train.append(term)
        if not fired_train:
            never_fired_train.append(term)

    flat_labels = [label for sublist in lexicon["lexical_labels"].fillna("").str.split(";") for label in sublist if label.strip()]
    label_counter = Counter(flat_labels)

    return {
        "language": language,
        "lexicon_file": project_relative(lexicon_path),
        "lexicon_terms": int(len(lexicon)),
        "bad_source_split_rows": int(bad_split_rows),
        "source_split_counts": (
            lexicon[split_col].value_counts(dropna=False).to_dict() if split_col else {}
        ),
        "source_datasets_counts": lexicon["source_datasets"].value_counts(dropna=False).to_dict(),
        "lexical_labels": dict(label_counter),
        "exact_full_comment_matches": exact_matches,
        "terms_fired_in_test_not_train": in_test_not_train,
        "terms_not_fired_in_train": sorted(set(never_fired_train)),
    }


def main() -> int:
    results = [
        audit(
            "vi",
            LEXICON_DIR / "verified_lexicon.csv",
            DATA_DIR / "test.csv",
            DATA_DIR / "train.csv",
        ),
        audit(
            "en",
            LEXICON_DIR / "verified_lexicon_en_combined.csv",
            DATA_DIR.parent / "splits_en" / "test.csv",
            DATA_DIR.parent / "splits_en" / "train.csv",
        ),
    ]
    payload = {"systems": {}, "results": results} if False else {"results": results}
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "lexicon_leakage_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# Lexicon-level leakage audit (held-out test sets)", ""]
    for r in results:
        lines += [
            f"## {r['language'].upper()} lexicon",
            "",
            f"- lexicon file: {r['lexicon_file']}",
            f"- lexicon terms: {r['lexicon_terms']}",
            f"- rows with bad source split (test/validation/dev): {r['bad_source_split_rows']}",
            f"- source split counts: {r['source_split_counts']}",
            f"- source dataset counts: {r['source_datasets_counts']}",
            f"- labels: {r['lexical_labels']}",
            f"- exact full-comment matches with held-out test: {len(r['exact_full_comment_matches'])}",
            f"- terms firing in test but not train: {len(r['terms_fired_in_test_not_train'])}",
            "",
        ]
    (ANALYSIS_DIR / "lexicon_leakage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in results:
        print(f"{r['language']}: terms={r['lexicon_terms']} bad_split_rows={r['bad_source_split_rows']} "
              f"exact_matches={len(r['exact_full_comment_matches'])} fired_test_not_train={len(r['terms_fired_in_test_not_train'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
