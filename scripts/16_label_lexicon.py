"""Step 16 - label the extracted candidates with HATE / OFFENSIVE (plan section 36).

    python scripts/16_label_lexicon.py
    python scripts/16_label_lexicon.py --min-evidence 5 --show-terms 20
    python scripts/16_label_lexicon.py --validate-only

Scans the three TRAIN splits (ViHSD / ViCM / bontori), counts per-label
evidence for every candidate of data/processed/lexicon/sources/sensitive_words_from_datasets.txt
and writes candidates_train.csv, verified_lexicon.csv, review_queue.csv,
label_mapping.yaml and provenance.json into data/processed/lexicon/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.lexical_labeler import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
