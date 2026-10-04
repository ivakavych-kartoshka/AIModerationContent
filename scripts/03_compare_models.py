"""Step 3 - compare the four models and generate the paper-ready output.

    python scripts/03_compare_models.py

Writes ``evaluation/comparison/`` (metrics table, PR/ROC/val-curve comparisons),
copies the figures into ``paper/figures/`` and generates the LaTeX tables in
``paper/tables/`` directly from the evaluation results - no number is ever typed
by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.comparison.compare_models import main as compare_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(compare_main(sys.argv[1:]))
