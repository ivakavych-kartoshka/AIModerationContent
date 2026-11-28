"""Step 20 - prune the frozen lexicon with the TF-IDF gate (plan section 36).

    python scripts/20_prune_lexicon.py
    python scripts/20_prune_lexicon.py --tfidf-min-score 0.005 --dry-run
    python scripts/20_prune_lexicon.py --in-place

Re-scores every entry of data/processed/lexicon/verified_lexicon.csv against the
training corpus, drops the terms dominated by CLEAN usage / too weak to separate
toxic from clean, and writes verified_lexicon_pruned.csv plus a per-entry audit
(lexicon_prune_audit.csv) and an updated lexicon_report.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.lexicon_pruner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
