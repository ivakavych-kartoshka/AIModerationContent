"""Step 12 - build the sensitive-word lexicon from two public datasets.

    python scripts/12_build_lexicon.py
    python scripts/12_build_lexicon.py --stage download
    python scripts/12_build_lexicon.py --stage extract summary --show-terms 10

Downloads the raw lexicon files into ``data/raw/lexicon/`` (never edited)
and writes normalized word lists into ``data/processed/lexicon/``:
one file per source plus a merged ``sensitive_words_combined.txt`` and a
``lexicon_report.json`` with counts, overlap and SHA-256 fingerprints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.lexicon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
