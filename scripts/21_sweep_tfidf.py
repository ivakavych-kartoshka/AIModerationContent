"""Step 21 - sweep the TF-IDF gate thresholds on the validation split (plan 36, 44).

    python scripts/21_sweep_tfidf.py
    python scripts/21_sweep_tfidf.py --ratios 0.5 0.6 0.7 --scores 0.003 0.004 0.005
    python scripts/21_sweep_tfidf.py --limit 500        # quick smoke run

Builds the pruned lexicon for every (min_ratio, min_score) pair, evaluates the
rule engine on the validation split and reports the macro-F1 of each point in
outputs/evaluation/rule_engine_tfidf_sweep/. The test split is never used here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.tfidf_sweep import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
