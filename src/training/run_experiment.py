"""Training entry point.

Usage
-----
    python -m src.training.run_experiment --config configs/sensitiveai-vi.yaml
    python -m src.training.run_experiment --config configs/sensitiveai-vi-customdlr2stage.yaml \
        --set training.batch_size=8

What it does
------------
1. loads + validates the config and verifies the **fixed split fingerprint**
   (so a run can never silently use a different split than the other three);
2. seeds everything, builds the tokenized datasets;
3. computes class weights from the **training split only**;
4. builds the model (standard head or custom head);
5. resolves the training stages (1, 2 or 3);
6. trains with the callback stack (best checkpoint, early stopping, CSV log);
7. writes ``config.yaml``, ``training_log.csv``, ``training_summary.json``,
   ``class_weights.json`` and ``best_model/`` into
   ``reports/<model>/run_XXX/`` and the stage checkpoints into
   ``experiments/<model>/run_XXX/``.

Evaluation is a separate step (``python -m src.evaluation.run_evaluation``) so
that the test set is touched exactly once, at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..callbacks.base import CallbackList
from ..callbacks.csv_logger import CSVLoggerCallback
from ..callbacks.monitoring import BestCheckpointCallback, EarlyStoppingCallback, resolve_direction
from ..common.config import dump_json, dump_yaml, get, load_config, resolve_config_path, to_plain
from ..common.env_info import collect_environment
from ..common.folder_docs import ensure_explain
from ..common.logging_utils import add_file_handler, get_logger
from ..common.paths import (
    ensure_dir,
    experiment_dir,
    guard_free_run_dir,
    next_run_id,
    project_path,
    project_relative,
    report_dir,
    run_dir_name,
)
from ..common.seeding import seed_everything
from ..data.dataset import build_datasets
from ..data.split import split_fingerprints, verify_split_fingerprint
from ..losses.builder import class_weights_for_config
from ..losses.class_weights import describe as describe_class_weights
from ..losses.class_weights import label_counts
from ..models.model import build_model, count_parameters
from ..common.constants import ID2LABEL, NUM_LABELS
from .stages import resolve_stages, stages_summary
from .trainer import Trainer

LOGGER = get_logger("sensitiveai.train")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.training.run_experiment",
        description="Train one SensitiveAI model from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Config name (e.g. sensitiveai-vi) or path to a YAML file.")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Dotted config overrides, e.g. training.batch_size=8 loss.type=focal")
    p.add_argument("--run-id", default=None, help="Explicit run id (default: next free run_XXX).")
    p.add_argument("--reports-dir", default=None, help="Override reports root directory.")
    p.add_argument("--experiments-dir", default=None, help="Override experiments root directory.")
    p.add_argument("--output-dir", default=None, help="Explicit report directory (overrides --run-id).")
    p.add_argument("--no-save-checkpoints", action="store_true", help="Do not persist weights.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build everything, print the plan, and exit without training.")
    p.add_argument("--max-train-samples", type=int, default=None, help="Debug: cap the training set size.")
    p.add_argument("--max-eval-samples", type=int, default=None, help="Debug: cap validation/test set size.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Debug: tiny subsets, 1 epoch, checkpoints disabled.")
    p.add_argument("--device", default=None, help="Force 'cuda' or 'cpu'.")
    p.add_argument("--allow-run-reuse", action="store_true",
                   help="Append to an occupied run directory instead of refusing (logs are never "
                        "overwritten by default).")
    return p


# --------------------------------------------------------------------------- #
def collect_class_weights(cfg: Dict[str, Any], train_labels: List[int]) -> Dict[str, Any]:
    """Class weights for every stage, all derived from the train split only."""
    num_labels = int(get(cfg, "dataset.num_labels", NUM_LABELS))
    counts_arr = label_counts(train_labels, num_labels)
    counts = {ID2LABEL[i]: int(counts_arr[i]) for i in range(num_labels)}

    stages = resolve_stages(cfg)
    per_stage: Dict[str, Optional[Dict[str, float]]] = {}
    for stage in stages:
        per_stage[stage.name] = class_weights_for_config(stage.loss or cfg.get("loss", {}), train_labels, num_labels)

    primary = per_stage.get(stages[0].name)
    return {
        "estimated_on": "train split only",
        "train_label_counts": counts,
        "per_stage": per_stage,
        "primary": primary,
        "table": describe_class_weights(counts, primary or {name: 1.0 for name in ID2LABEL.values()}),
    }


def build_callbacks(cfg: Dict[str, Any], report_path: Path, save_checkpoints: bool) -> CallbackList:
    es_cfg = get(cfg, "early_stopping", {}) or {}
    metric = str(es_cfg.get("metric", "macro_f1"))
    mode = es_cfg.get("mode") or resolve_direction(metric)

    callbacks = CallbackList(
        [
            BestCheckpointCallback(report_path, metric=metric, mode=mode, save=save_checkpoints),
            EarlyStoppingCallback(
                metric=metric,
                mode=mode,
                patience=int(es_cfg.get("patience", 2)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                enabled=bool(es_cfg.get("enabled", True)),
            ),
            CSVLoggerCallback(report_path / "training_log.csv", model_name=str(cfg.get("model_name", ""))),
        ]
    )
    return callbacks


# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path, overrides=args.set)
    run_id = run_dir_name(args.run_id) if args.run_id else next_run_id(
        str(cfg["model_name"]), Path(args.reports_dir) if args.reports_dir else None
    )

    if args.smoke_test:
        LOGGER.warning(
            "SMOKE TEST: full dataset, 1 epoch, checkpoints disabled - "
            "pipeline check on the real data distribution."
        )
        # Run on the full dataset so metrics are meaningful.
        # Only cap if the caller explicitly passes --max-train-samples / --max-eval-samples.
        if args.max_train_samples:
            cfg["dataset"]["max_train_samples"] = args.max_train_samples
        if args.max_eval_samples:
            cfg["dataset"]["max_eval_samples"] = args.max_eval_samples
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 4
        cfg["training"]["eval_batch_size"] = 8
        cfg["training"]["num_workers"] = 0
        cfg["early_stopping"]["enabled"] = False
        cfg["output"]["save_checkpoints"] = False
        if cfg.get("stages"):
            for stage in cfg["stages"]:
                stage["epochs"] = 1
    else:
        if args.max_train_samples:
            cfg["dataset"]["max_train_samples"] = args.max_train_samples
        if args.max_eval_samples:
            cfg["dataset"]["max_eval_samples"] = args.max_eval_samples

    save_checkpoints = bool(cfg.get("output", {}).get("save_checkpoints", True)) and not args.no_save_checkpoints

    reports_root = Path(args.reports_dir) if args.reports_dir else project_path(get(cfg, "output.reports_dir", "outputs/reports"))
    experiments_root = Path(args.experiments_dir) if args.experiments_dir else project_path(
        get(cfg, "output.experiments_dir", "outputs/experiments")
    )
    report_path = Path(args.output_dir) if args.output_dir else report_dir(str(cfg["model_name"]), run_id, reports_root)
    experiment_path = experiment_dir(str(cfg["model_name"]), run_id, experiments_root)

    guard_free_run_dir(report_path, allow_reuse=bool(getattr(args, "allow_run_reuse", False)))
    guard_free_run_dir(experiment_path, allow_reuse=True)

    if args.dry_run:
        LOGGER.info("  report dir    : %s", report_path)
        LOGGER.info("  experiment dir: %s", experiment_path)
        LOGGER.info("=" * 78)
        LOGGER.info(
            "[dry-run] nothing written: no run folder, no train.log, no run id consumed. "
            "The real run will use %s.",
            report_dir(str(cfg["model_name"]), run_id, reports_root).name,
        )
    else:
        ensure_dir(report_path)
        ensure_dir(experiment_path)
        add_file_handler(LOGGER, report_path / "train.log")
        ensure_explain(report_path)
        ensure_explain(experiment_path)
        ensure_explain(report_path.parent)
        ensure_explain(experiment_path.parent)

    # ------------------------------------------------------------------ #
    LOGGER.info("=" * 78)
    LOGGER.info("SensitiveAI training run")
    LOGGER.info("  config        : %s", config_path)
    LOGGER.info("  model_name    : %s", cfg["model_name"])
    LOGGER.info("  run_id        : %s", run_id)
    LOGGER.info("  base_model    : %s", cfg["base_model"])
    LOGGER.info("  head          : %s", cfg["head"])
    LOGGER.info("  report dir    : %s", report_path)
    LOGGER.info("  experiment dir: %s", experiment_path)
    LOGGER.info("=" * 78)

    # ---- data leakage guard ------------------------------------------- #
    splits_dir = project_path(get(cfg, "dataset.splits_dir", "data/splits"))
    verify_split_fingerprint(splits_dir)
    fingerprints = split_fingerprints(splits_dir)
    LOGGER.info("[data] split fingerprints: %s", {k: v[:12] for k, v in fingerprints.items()})
    cfg["dataset"]["split_fingerprints"] = fingerprints

    seed_everything(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", True)))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        LOGGER.info("[env] GPU: %s", torch.cuda.get_device_name(device))

    # ---- data --------------------------------------------------------- #
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(cfg["base_model"]), use_fast=True)
    bundle = build_datasets(
        tokenizer=tokenizer,
        splits_dir=splits_dir,
        max_length=int(get(cfg, "tokenizer.max_length", 256)),
        seed=int(cfg.get("seed", 42)),
        text_column=str(get(cfg, "dataset.text_column", "text")),
        label_column=str(get(cfg, "dataset.label_column", "label")),
        max_train_samples=get(cfg, "dataset.max_train_samples"),
        max_eval_samples=get(cfg, "dataset.max_eval_samples"),
    )

    train_labels = [int(v) for v in bundle.train.labels.tolist()]
    class_weight_info = collect_class_weights(cfg, train_labels)
    LOGGER.info("\n%s", class_weight_info["table"])

    # ---- model -------------------------------------------------------- #
    head_options = dict(cfg.get("head_options") or {})
    model = build_model(
        head=str(cfg["head"]),
        base_model=str(cfg["base_model"]),
        num_labels=int(get(cfg, "dataset.num_labels", NUM_LABELS)),
        head_options=head_options or None,
        encoder_checkpoint=cfg.get("encoder_checkpoint"),
    )
    param_counts = model.parameter_counts()
    LOGGER.info(
        "[model] total=%.2fM trainable=%.2fM non-trainable=%.2fM",
        param_counts["total_parameters"] / 1e6,
        param_counts["trainable_parameters"] / 1e6,
        param_counts["non_trainable_parameters"] / 1e6,
    )

    stages = resolve_stages(cfg)
    LOGGER.info("\n%s", stages_summary(stages))

    if args.dry_run:
        LOGGER.info("[dry-run] stopping before writing config or training.")
        return {"dry_run": True, "report_dir": str(report_path), "config": to_plain(cfg)}

    # ---- persist the resolved configuration --------------------------- #
    header = (
        f"# Resolved configuration of {cfg['model_name']} ({run_id})\n"
        f"# Generated by src.training.run_experiment - do not edit by hand.\n"
    )
    dump_yaml(cfg, report_path / "config.yaml", header=header)
    dump_yaml(cfg, experiment_path / "config.yaml", header=header)
    dump_json(class_weight_info, report_path / "class_weights.json")
    dump_json(
        {"fingerprints": fingerprints, "sizes": bundle.sizes, "splits_dir": project_relative(splits_dir)},
        report_path / "split_manifest_used.json",
    )

    # ---- train -------------------------------------------------------- #
    callbacks = build_callbacks(cfg, report_path, save_checkpoints)
    LOGGER.info("[train] callbacks: %s", ", ".join(callbacks.describe()))

    trainer = Trainer(
        model=model,
        bundle=bundle,
        stages=stages,
        config=cfg,
        output_dir=report_path,
        experiment_dir=experiment_path,
        model_name=str(cfg["model_name"]),
        device=device,
        callbacks=callbacks,
    )
    trainer.tokenizer = tokenizer  # saved next to every checkpoint

    started = time.time()
    result = trainer.train()
    result_dict = result.to_dict()
    result_dict["wall_clock_seconds"] = round(time.time() - started, 3)

    summary = {
        "run_id": run_id,
        "model_name": cfg["model_name"],
        "base_model": cfg["base_model"],
        "head": cfg["head"],
        "config_file": project_relative(config_path),
        "report_dir": project_relative(report_path),
        "experiment_dir": project_relative(experiment_path),
        "dataset": {
            "name": get(cfg, "dataset.name"),
            "splits_dir": project_relative(splits_dir),
            "fingerprints": fingerprints,
            "num_train": bundle.sizes["train"],
            "num_validation": bundle.sizes["validation"],
            "num_test": bundle.sizes["test"],
            "max_length": bundle.max_length,
        },
        "class_weights": class_weight_info,
        "parameters": {**param_counts, "head_only": count_parameters(model.head_module)},
        "training": result_dict,
        "environment": collect_environment(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dump_json(summary, report_path / "training_summary.json")
    dump_json(summary, experiment_path / "training_summary.json")

    # mark the newest successful run so evaluation can find it automatically
    (report_path.parent / "run_pointer.txt").write_text(run_dir_name(run_id), encoding="utf-8")

    LOGGER.info("=" * 78)
    LOGGER.info("Training finished: %s", run_id)
    LOGGER.info("  best %s = %s at epoch %s (stage %s)",
                result.best_metric,
                f"{result.best_value:.6f}" if result.best_value is not None else "n/a",
                result.best_epoch, result.best_stage)
    LOGGER.info("  epochs run      : %d/%d", result.epochs_run, result.total_epochs_planned)
    LOGGER.info("  training time   : %.1f s", result.training_seconds)
    LOGGER.info("  best checkpoint : %s", report_path / "best_model")
    LOGGER.info("  training log    : %s", report_path / "training_log.csv")
    LOGGER.info("  run guide       : %s", report_path / "explain.md")
    LOGGER.info("")
    LOGGER.info("Next step (evaluation on the test set):")
    LOGGER.info("  python -m src.evaluation.run_evaluation --model %s --run-id %s",
                cfg["model_name"], run_id)
    LOGGER.info("=" * 78)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except FileExistsError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.warning("Interrupted by the user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
