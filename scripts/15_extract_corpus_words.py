"""Step 15 - extract sensitive words from ViHSD, ViCM and bontori.

    python scripts/15_extract_corpus_words.py
    python scripts/15_extract_corpus_words.py --min-toxic-df 5 --max-clean-df 1
    python scripts/15_extract_corpus_words.py --show-terms 10

Scans ViHSD (cleaned), ViCM (xlsx+csv from additional), and bontori (parquet).
Produces per-source extracted files under data/processed/lexicon/ and merges
into sensitive_words_combined.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus_lexicon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
