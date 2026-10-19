"""Step 23 - hybrid fusion of the rule engine and the AI classifier (plan 40-44).

    python scripts/23_evaluate_fusion.py
    python scripts/23_evaluate_fusion.py --model sensitiveai-vi-customdlr2stage
    python scripts/23_evaluate_fusion.py --force-recompute

Tunes the mix weight alpha (and label-specific thresholds, and a logistic-
regression meta-classifier) on the VALIDATION split, freezes them, then scores
the TEST split exactly once with the rule-engine scores and the AI
probabilities.  See ``Rule_Engine_Plan.md`` sections 40-44.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion.run_fusion import main as _main  # noqa: E402


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-tty
            pass


if __name__ == "__main__":
    _ensure_utf8_console()
    sys.exit(_main(sys.argv[1:]))
