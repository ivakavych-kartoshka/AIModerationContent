"""Step 1 - train the four models sequentially.

    python scripts/01_train_all.py                     # all four, in order
    python scripts/01_train_all.py --models sensitiveai-vi
    python scripts/01_train_all.py --dry-run           # only show the resolved plan

Order matters for readability only (each model is independent and uses the same
fixed split):

    1. sensitiveai-vi                    baseline, standard head
    2. sensitiveai-vi-custom             custom head
    3. sensitiveai-vi-customdlr2stage    custom head + DLR + 2 stages
    4. sensitiveai-vi-custom-curriculum  custom head + 3-stage curriculum

Model 2 may optionally be initialised from a checkpoint of model 1 - in that case
train model 1 first and pass ``--init-encoder-from-model1``.
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
from src.common.paths import find_best_report_dir  # noqa: E402
from src.training.run_experiment import build_parser as train_parser  # noqa: E402
from src.training.run_experiment import run as train_run  # noqa: E402

LOGGER = get_logger("sensitiveai.train_all")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    models: List[str] = list(MODEL_NAMES)
    passthrough: List[str] = []
    init_from_model1 = False

    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--models":
            i += 1
            models = []
            while i < len(argv) and not argv[i].startswith("--"):
                models.append(argv[i])
                i += 1
            continue
        if token == "--init-encoder-from-model1":
            init_from_model1 = True
            i += 1
            continue
        passthrough.append(token)
        i += 1

    parser = train_parser()
    # the shared parser requires --config; train_all sets it per model below
    if "--config" not in passthrough:
        passthrough = ["--config", f"configs/{models[0]}.yaml"] + passthrough
    args = parser.parse_args(passthrough)

    LOGGER.info("=" * 78)
    LOGGER.info("Training %d model(s): %s", len(models), ", ".join(models))
    LOGGER.info("=" * 78)

    results = []
    for index, model in enumerate(models, start=1):
        LOGGER.info("")
        LOGGER.info("#" * 78)
        LOGGER.info("# [%d/%d] %s", index, len(models), model)
        LOGGER.info("#" * 78)

        overrides = list(args.set)
        if init_from_model1 and model == "sensitiveai-vi-custom":
            try:
                ckpt = find_best_report_dir("sensitiveai-vi") / "best_model"
                overrides.append(f"encoder_checkpoint={ckpt.as_posix()}")
                LOGGER.info("[train_all] model 2 encoder initialised from %s", ckpt)
            except FileNotFoundError as exc:
                LOGGER.warning("[train_all] %s - model 2 will start from the pre-trained encoder", exc)

        model_args = argparse.Namespace(**vars(args))
        model_args.config = f"configs/{model}.yaml"
        model_args.set = overrides

        started = time.perf_counter()
        try:
            summary = train_run(model_args)
        except Exception as exc:  # keep going with the remaining models
            LOGGER.error("[train_all] %s FAILED: %s", model, exc, exc_info=True)
            results.append({"model": model, "status": "failed", "error": str(exc)})
            continue
        results.append(
            {
                "model": model,
                "status": "ok",
                "run_id": summary.get("run_id"),
                "seconds": round(time.perf_counter() - started, 1),
                "best_macro_f1": summary.get("training", {}).get("best_value"),
            }
        )

        if args.dry_run:
            LOGGER.info("[train_all] dry run - skipping the remaining models")
            break

    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info("Training summary")
    for row in results:
        LOGGER.info("  %-38s %-6s run=%s  %.1fs", row["model"], row["status"],
                    row.get("run_id", "-"), row.get("seconds", 0.0))
    LOGGER.info("=" * 78)
    if all(r["status"] == "ok" for r in results):
        LOGGER.info("Next: python scripts/02_evaluate_all.py")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
