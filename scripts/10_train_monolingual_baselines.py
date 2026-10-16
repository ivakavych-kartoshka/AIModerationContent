"""Step 10 - monolingual baselines (PhoBERT / ViSoBERT) on the fixed split.

    python scripts/10_train_monolingual_baselines.py
    python scripts/10_train_monolingual_baselines.py --models phobert-base
    python scripts/10_train_monolingual_baselines.py --dry-run

Trains the two Vietnamese monolingual encoders (``vinai/phobert-base-v2`` and
``uitnlp/visobert``) with the *exact* baseline recipe of ``sensitiveai-vi``
(standard head, unweighted cross-entropy, single LR, plain argmax) on the same
fixed split.  The published PhoBERT/ViSoBERT numbers in the paper's external
comparison come from other papers with different splits and preprocessing; these
runs give a same-split, same-protocol point of reference instead.

Outputs
-------
* runs in the normal ``outputs/reports/<model>/run_XXX`` tree
* evaluation mirrors/logs under ``outputs/baselines/evaluation``
* ``outputs/baselines/monolingual_summary.json`` + ``.md``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import dump_json  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.common.paths import project_relative  # noqa: E402
from src.evaluation.run_evaluation import build_parser as eval_parser  # noqa: E402
from src.evaluation.run_evaluation import run as eval_run  # noqa: E402
from src.training.run_experiment import build_parser as train_parser  # noqa: E402
from src.training.run_experiment import run as train_run  # noqa: E402

LOGGER = get_logger("sensitiveai.monolingual")

BASELINE_MODELS: tuple[str, ...] = ("phobert-base", "visobert-base")

METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "HATE_f1",
    "OFFENSIVE_f1",
    "CLEAN_f1",
    "macro_roc_auc",
)


def read_metrics(report_dir: Path) -> Dict[str, Any]:
    metrics_path = report_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {k: payload.get(k) for k in METRIC_KEYS if payload.get(k) is not None}


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# Monolingual baselines (same split / same recipe as sensitiveai-vi)",
        "",
        "| Model | Acc. | Macro-F1 | HATE-F1 | OFFENSIVE-F1 | CLEAN-F1 | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            return f"{value:.4f}" if isinstance(value, (int, float)) else "-"

        lines.append(
            f"| {row['model']} | {fmt('accuracy')} | {fmt('macro_f1')} | {fmt('HATE_f1')} "
            f"| {fmt('OFFENSIVE_f1')} | {fmt('CLEAN_f1')} | {fmt('macro_roc_auc')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/10_train_monolingual_baselines.py",
        description="Train + evaluate the PhoBERT / ViSoBERT monolingual baselines.",
    )
    parser.add_argument("--models", nargs="*", default=list(BASELINE_MODELS), choices=list(BASELINE_MODELS))
    parser.add_argument("--out-dir", default="outputs/baselines", help="Where to write evaluation mirrors/summary.")
    parser.add_argument("--dry-run", action="store_true", help="Show the resolved plan and stop.")
    parser.add_argument("--skip-eval", action="store_true", help="Train only; do not score the test split.")
    args = parser.parse_args(argv)

    root = Path(args.out_dir).resolve()
    eval_dir = root / "evaluation"

    LOGGER.info("Monolingual baselines: %s", ", ".join(args.models))
    rows: List[Dict[str, Any]] = []

    for index, model in enumerate(args.models, start=1):
        LOGGER.info("")
        LOGGER.info("#" * 78)
        LOGGER.info("# [%d/%d] %s", index, len(args.models), model)
        LOGGER.info("#" * 78)

        train_argv = ["--config", f"configs/{model}.yaml"]
        if args.dry_run:
            train_argv.append("--dry-run")
        summary = train_run(train_parser().parse_args(train_argv))
        if args.dry_run:
            continue

        report_dir = Path(summary["report_dir"])
        run_id = summary["run_id"]
        row: Dict[str, Any] = {
            "model": model,
            "run_id": run_id,
            "report_dir": project_relative(report_dir),
        }

        if not args.skip_eval:
            eval_args = eval_parser().parse_args(
                ["--model", model, "--run-id", run_id, "--eval-dir", str(eval_dir)]
            )
            started = time.perf_counter()
            eval_run(eval_args)
            row["eval_seconds"] = round(time.perf_counter() - started, 1)
            row.update(read_metrics(report_dir))

        rows.append(row)

    if args.dry_run:
        LOGGER.info("[monolingual] dry run - stopping before training.")
        return 0

    dump_json({"models": args.models, "rows": rows}, root / "monolingual_summary.json")
    write_markdown(rows, root / "monolingual_summary.md")

    LOGGER.info("=" * 78)
    LOGGER.info("Monolingual baseline results")
    for row in rows:
        LOGGER.info(
            "  %-14s Acc=%.4f  M-F1=%.4f  HATE-F1=%.4f",
            row["model"],
            row.get("accuracy", float("nan")),
            row.get("macro_f1", float("nan")),
            row.get("HATE_f1", float("nan")),
        )
    LOGGER.info("=" * 78)
    LOGGER.info("Artefacts: %s", project_relative(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
