"""Step 0 - prepare the data once.

    python scripts/00_prepare_data.py
    python scripts/00_prepare_data.py --strategy stratified --overwrite

Downloads ViHSD, cleans it and writes the ONE fixed split in ``data/splits/``.
Run this before any training script.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prepare_data import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
