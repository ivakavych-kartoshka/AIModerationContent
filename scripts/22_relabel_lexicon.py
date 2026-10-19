"""Step 22 - re-label the pruned lexicon with the labelling algorithm (plan 36).

    python scripts/22_relabel_lexicon.py
    python scripts/22_relabel_lexicon.py --share-threshold 0.7 --dry-run
    python scripts/22_relabel_lexicon.py --no-tfidf

Reads data/processed/lexicon/pruning/verified_lexicon_pruned.csv, runs the lexical
labelling algorithm (share / lift / confidence rules, evidence pooled over the
train splits) on exactly those terms, and writes verified_lexicon_relabeled.csv
plus per-dataset copies (verified_lexicon_relabeled_<dataset>.csv), a
before/after audit (lexicon_relabel_changes.csv) and an updated
lexicon_report.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.lexicon_relabeler import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
