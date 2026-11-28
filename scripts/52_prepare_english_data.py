"""Step 52 - prepare the English dataset once.

    python scripts/52_prepare_english_data.py
    python scripts/52_prepare_english_data.py --stage verify summary
    python scripts/52_prepare_english_data.py --overwrite

Downloads the Davidson hate-speech/offensive dataset, cleans it and writes the
ONE fixed split in ``data/splits_en/``.  Run this before training or extracting
the English lexicon.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prepare_data_en import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
