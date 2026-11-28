"""Step 13 - extract sensitive words from ViHSD into the lexicon.

    python scripts/13_extract_vihsd_words.py
    python scripts/13_extract_vihsd_words.py --min-toxic-df 10 --show-terms 20
    python scripts/13_extract_vihsd_words.py --splits train dev

Scans the cleaned ViHSD splits, keeps syllables / syllable-pairs that appear
repeatedly in HATE+OFFENSIVE rows and (almost) never in CLEAN rows, and
merges the new terms into ``data/processed/lexicon/``. Run
``scripts/12_build_lexicon.py`` first so the base lexicon exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.vihsd_lexicon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
