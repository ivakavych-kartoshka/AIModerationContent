"""Step 2 - evaluate every trained model on the test set.

    python scripts/02_evaluate_all.py
    python scripts/02_evaluate_all.py --models sensitiveai-vi --no-benchmark

Each model is evaluated exactly once with its best *validation* checkpoint.
Produces confusion matrices, PR/ROC curves, validation curves, the classification
report, ``model_summary.txt``, ``metrics.json`` and the row in
``evaluation/logs/evaluation_log.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.constants import MODEL_NAMES  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.run_evaluation import build_parser as eval_parser  # noqa: E402
from src.evaluation.run_evaluation import run as eval_run  # noqa: E402

LOGGER = get_logger("sensitiveai.eval_all")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    models: List[str] = list(MODEL_NAMES)
    passthrough: List[str] = []

    i = 0
    while i < len(argv):
        if argv[i] == "--models":
            i += 1
            models = []
            while i < len(argv) and not argv[i].startswith("--"):
                models.append(argv[i])
                i += 1
            continue
        passthrough.append(argv[i])
        i += 1

    parser = eval_parser()
    # the shared parser requires --model; this script sets it per model below
    if "--model" not in passthrough:
        passthrough = ["--model", models[0]] + passthrough
    args = parser.parse_args(passthrough)

    LOGGER.info("Evaluating %d model(s): %s", len(models), ", ".join(models))
    ok, failed = 0, 0

    for index, model in enumerate(models, start=1):
        LOGGER.info("")
        LOGGER.info("#" * 78)
        LOGGER.info("# [%d/%d] %s", index, len(models), model)
        LOGGER.info("#" * 78)
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model
        started = time.perf_counter()
        try:
            eval_run(model_args)
            ok += 1
            LOGGER.info("[eval_all] %s done in %.1fs", model, time.perf_counter() - started)
        except Exception as exc:
            failed += 1
            LOGGER.error("[eval_all] %s FAILED: %s", model, exc, exc_info=True)

    LOGGER.info("")
    LOGGER.info("[eval_all] %d succeeded, %d failed", ok, failed)
    if ok:
        LOGGER.info("Next: python scripts/03_compare_models.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
