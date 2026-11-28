"""Step 14 - phrase-level provenance check for lexicon entries (leakage audit).

    python scripts/14_lexicon_phrase_provenance.py

For every multi-token lexicon entry, checks whether the *exact phrase* (after
normalisation) occurs as a substring of train / validation / test text.  An
entry whose phrase occurs only in the held-out test would be a leakage signal;
entries mined from the training split must occur in train (or in the mined
source corpora).

Writes::
    outputs/evaluation/analysis/lexicon_phrase_provenance.json
    outputs/evaluation/analysis/lexicon_phrase_provenance.md
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.paths import project_relative  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
LEXICON_DIR = PROJECT_ROOT / "data" / "processed" / "lexicon"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "analysis"


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFD", str(text or ""))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"[^0-9a-z\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def check(language: str, lexicon_path: Path, split_dir: Path) -> Dict[str, Any]:
    lexicon = pd.read_csv(lexicon_path, encoding="utf-8")
    corpus: Dict[str, List[str]] = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(split_dir / f"{split}.csv", encoding="utf-8")
        corpus[split] = [_norm(t) for t in frame["text"].tolist()]

    multiword = [t for t in lexicon["term"].astype(str).tolist() if len(t.split()) >= 2]
    only_test: List[str] = []
    missing_from_train: List[str] = []

    for term in multiword:
        phrase = _norm(term)
        if not phrase:
            continue
        hits = {s: sum(1 for text in corpus[s] if phrase in text) for s in corpus}
        if hits["test"] > 0 and hits["train"] == 0:
            only_test.append({"term": term, "test": hits["test"], "validation": hits["validation"]})
        if hits["train"] == 0:
            missing_from_train.append({"term": term, "hits": hits})

    return {
        "language": language,
        "lexicon_file": project_relative(lexicon_path),
        "multiword_entries": len(multiword),
        "phrase_found_only_in_test": only_test,
        "phrase_never_found_in_train": missing_from_train,
        "phrase_found_only_in_test_count": len(only_test),
        "phrase_never_found_in_train_count": len(missing_from_train),
    }


def main() -> int:
    results = [
        check("vi", LEXICON_DIR / "verified_lexicon.csv", DATA_DIR),
        check("en", LEXICON_DIR / "verified_lexicon_en_combined.csv", DATA_DIR.parent / "splits_en"),
    ]
    payload = {"results": results}
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "lexicon_phrase_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# Lexicon phrase provenance (train/val/test containment)", ""]
    for r in results:
        lines += [
            f"## {r['language'].upper()}",
            "",
            f"- lexicon file: {r['lexicon_file']}",
            f"- multi-token entries: {r['multiword_entries']}",
            f"- phrase found only in test: {r['phrase_found_only_in_test_count']}",
            f"- phrase never found in train: {r['phrase_never_found_in_train_count']}",
            "",
        ]
    (ANALYSIS_DIR / "lexicon_phrase_provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for r in results:
        print(f"{r['language']}: multiword={r['multiword_entries']} "
              f"only_test={r['phrase_found_only_in_test_count']} no_train={r['phrase_never_found_in_train_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
