"""Evaluation entry point: turns a trained run into the full artefact set.

Usage
-----
    python -m src.evaluation.run_evaluation --model sensitiveai-vi
    python -m src.evaluation.run_evaluation --model sensitiveai-vi-customdlr2stage --run-id run_001

What it produces (plan sections 16, 20-22, 27-31)
--------------------------------------------------
in ``reports/<model>/run_XXX/``

    confusion_matrix.png, confusion_matrix_raw.png, confusion_matrix_normalized.png
    classification_report.txt, Best_model_classification_report.txt
    pr_curve.png, pr_curve_data.json
    roc_curve.png, roc_curve_data.json
    val_accuracy_curve.png, val_loss_curve.png, train_loss_curve.png
    per_class_f1.png, model_summary.txt, metrics.json, thresholds.json
    test_predictions.csv, benchmark.json, evaluation.json

and copies of the key files in ``evaluation/per_model/<model>/`` plus one row in
``evaluation/logs/evaluation_log.csv``.

Protocol guarantees
-------------------
* the **test set is evaluated exactly once**, with the *best validation*
  checkpoint;
* label-specific thresholds are searched on **validation** and frozen before the
  test pass (``thresholds.json``);
* the split fingerprint recorded at training time must match the split on disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from ..callbacks.csv_logger import COLUMNS as LOG_COLUMNS
from ..common.config import dump_json, load_yaml
from ..common.constants import LABELS, MODEL_NAMES, NUM_LABELS
from ..common.env_info import collect_environment
from ..common.folder_docs import ensure_explain
from ..common.logging_utils import add_file_handler, get_logger
from ..common.paths import (
    archive_run_artifacts,
    ensure_dir,
    evaluation_log_path,
    find_best_report_dir,
    per_model_eval_dir,
    project_path,
)
from ..common.seeding import seed_everything
from ..data.dataset import build_datasets, make_dataloader
from ..data.split import load_manifest, split_fingerprints, verify_split_fingerprint
from ..losses.builder import build_loss
from ..models.model import CustomHeadModel, StandardHeadModel, count_parameters
from .benchmark import benchmark_model
from .curves import precision_recall_curve_data, roc_curve_data
from .metrics import (
    apply_decision_rule,
    classification_report_text,
    compute_metrics,
    confusion_matrix_text,
    predictions_dataframe,
    run_inference,
)
from .plots import (
    plot_confusion_matrix,
    plot_per_class_f1_bars,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_loss_curve,
    plot_validation_accuracy_curve,
    plot_validation_loss_curve,
)
from .report import build_model_summary, write_model_summary, write_text
from .thresholds import describe as describe_thresholds
from .thresholds import optimize_thresholds, save_thresholds

LOGGER = get_logger("sensitiveai.evaluate")

PRECISION_TO_DTYPE = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}

EVALUATION_LOG_COLUMNS = [
    "timestamp",
    "model",
    "run_id",
    "checkpoint",
    "dataset",
    "split",
    "decision_rule",
    "num_test_samples",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "HATE_f1",
    "OFFENSIVE_f1",
    "CLEAN_f1",
    "roc_auc_macro",
    "average_precision_macro",
    "latency_ms_per_sample",
    "fps",
    "parameters",
    "best_epoch",
]


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.run_evaluation",
        description="Evaluate a trained SensitiveAI model on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Model name (e.g. sensitiveai-vi).")
    p.add_argument("--run-id", default=None, help="Run id (default: the newest finished run).")
    p.add_argument("--reports-dir", default=None, help="Override the reports root.")
    p.add_argument("--checkpoint", default=None, help="Explicit checkpoint dir (default: <run>/best_model).")
    p.add_argument("--split", default="test", choices=["test", "validation"], help="Which split to score.")
    p.add_argument("--no-thresholds", action="store_true", help="Skip label-specific threshold search.")
    p.add_argument("--no-benchmark", action="store_true", help="Skip the latency/FPS benchmark.")
    p.add_argument("--force-argmax", action="store_true", help="Ignore thresholds even if thresholds.json exists.")
    p.add_argument("--max-eval-samples", type=int, default=None, help="Debug: cap the split size.")
    p.add_argument("--device", default=None, help="Force 'cuda' or 'cpu'.")
    return p


# --------------------------------------------------------------------------- #
def load_run_config(report_path: Path) -> Dict[str, Any]:
    config_file = report_path / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"{config_file} not found - the run has no resolved configuration. Was it trained?"
        )
    cfg = load_yaml(config_file)
    cfg["_report_dir"] = str(report_path)
    return cfg


def load_model_for_eval(cfg: Dict[str, Any], checkpoint: Path):
    head = str(cfg.get("head", "standard"))
    if head == "custom":
        model = CustomHeadModel.from_pretrained(checkpoint)
    else:
        model = StandardHeadModel.from_pretrained(checkpoint)
    LOGGER.info("[eval] loaded %s head from %s", head, checkpoint)
    return model


def resolve_checkpoint(report_path: Path, override: Optional[str], save_checkpoints: bool = True) -> Path:
    if override:
        return Path(override)
    best = report_path / "best_model"
    last = report_path / "last_model"
    if best.is_dir() and any(best.iterdir()):
        return best
    if last.is_dir() and any(last.iterdir()):
        LOGGER.warning("[eval] best_model not found - falling back to last_model (NOT the best epoch)")
        return last
    raise FileNotFoundError(
        f"No checkpoint in {report_path}. Re-train with output.save_checkpoints=true "
        "or pass --checkpoint <dir>."
    )


def read_training_log(report_path: Path) -> pd.DataFrame:
    path = report_path / "training_log.csv"
    if not path.is_file():
        LOGGER.warning("[eval] %s not found - validation curves will be skipped", path)
        return pd.DataFrame(columns=list(LOG_COLUMNS))
    return pd.read_csv(path)


def append_evaluation_log(row: Dict[str, Any], path: Optional[Path] = None) -> Path:
    out = Path(path) if path else evaluation_log_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.is_file()
    frame = pd.DataFrame([{c: row.get(c, "") for c in EVALUATION_LOG_COLUMNS}])
    frame.to_csv(out, mode="a", header=write_header, index=False, encoding="utf-8")
    LOGGER.info("[eval] evaluation log -> %s", out)
    return out


def _frozen_class_weights(report_path: Path) -> Optional[Dict[str, float]]:
    """Class weights recorded by the training run (estimated on train only).

    Reusing them keeps the reference loss comparable with training without
    re-reading (and re-tokenising) the training split.
    """
    path = Path(report_path) / "class_weights.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = payload.get("primary")
    return {str(k): float(v) for k, v in primary.items()} if primary else None


# --------------------------------------------------------------------------- #
#: Artifacts produced by one evaluation pass.  When a run is scored a second
#: time they are moved into ``history/`` first, so no previous number is lost.
EVALUATION_ARTIFACTS = (
    "metrics.json",
    "evaluation.json",
    "model_summary.txt",
    "test_predictions.csv",
    "classification_report.txt",
    "Best_model_classification_report.txt",
    "confusion_matrix*",
    "pr_curve*",
    "roc_curve*",
    "per_class_f1.png",
    "val_accuracy_curve.png",
    "val_loss_curve.png",
    "train_loss_curve.png",
    "benchmark.json",
    "thresholds.json",
    "thresholds.txt",
)


def archive_previous_evaluation(report_path: Path) -> Optional[Path]:
    """Move the artifacts of a previous evaluation into ``history/eval_<stamp>/``."""
    target = archive_run_artifacts(report_path, EVALUATION_ARTIFACTS)
    if target is not None:
        LOGGER.info("[eval] previous evaluation archived -> %s", target)
    return target


# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> Dict[str, Any]:
    reports_root = Path(args.reports_dir) if args.reports_dir else None
    report_path = find_best_report_dir(args.model, reports_root, args.run_id)
    run_id = report_path.name
    cfg = load_run_config(report_path)
    archive_previous_evaluation(report_path)
    add_file_handler(LOGGER, report_path / "evaluate.log")
    ensure_explain(report_path)

    LOGGER.info("=" * 78)
    LOGGER.info("Evaluating %s (%s)", args.model, run_id)
    LOGGER.info("  report dir : %s", report_path)
    LOGGER.info("=" * 78)

    # ---- data leakage guard ------------------------------------------- #
    splits_dir = project_path(cfg.get("dataset", {}).get("splits_dir", "data/splits"))
    verify_split_fingerprint(splits_dir, expected=cfg.get("dataset", {}).get("split_fingerprints"))
    LOGGER.info("[eval] split verified: %s", {k: v[:12] for k, v in split_fingerprints(splits_dir).items()})
    split_manifest = load_manifest(splits_dir) or {}

    seed_everything(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", True)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = str(cfg.get("training", {}).get("precision", "fp32")).lower()
    amp_dtype = PRECISION_TO_DTYPE.get(precision)

    # ---- model + tokenizer -------------------------------------------- #
    checkpoint = resolve_checkpoint(report_path, args.checkpoint)
    model = load_model_for_eval(cfg, checkpoint).to(device).eval()

    from transformers import AutoTokenizer

    tokenizer_dir = checkpoint if (checkpoint / "tokenizer_config.json").is_file() else None
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir or cfg["base_model"]), use_fast=True)

    bundle = build_datasets(
        tokenizer=tokenizer,
        splits_dir=splits_dir,
        max_length=int(cfg.get("tokenizer", {}).get("max_length", 256)),
        seed=int(cfg.get("seed", 42)),
        text_column=str(cfg.get("dataset", {}).get("text_column", "text")),
        label_column=str(cfg.get("dataset", {}).get("label_column", "label")),
        max_eval_samples=args.max_eval_samples,
        splits=("validation", "test"),
    )
    eval_batch_size = int(cfg.get("training", {}).get("eval_batch_size", 32))
    val_loader = make_dataloader(bundle.validation, eval_batch_size, shuffle=False, num_workers=0)
    test_loader = make_dataloader(bundle.test, eval_batch_size, shuffle=False, num_workers=0)

    # ---- validation pass: threshold search (never uses test) ---------- #
    thresholds_payload = None
    thresholds_list = None
    thresholds_enabled = bool(cfg.get("thresholds", {}).get("enabled", False)) and not args.no_thresholds
    val_outputs = run_inference(model, val_loader, device, amp_dtype=amp_dtype)

    if thresholds_enabled:
        tcfg = cfg.get("thresholds", {}) or {}
        thresholds_payload = optimize_thresholds(
            val_outputs["y_true"],
            val_outputs["y_proba"],
            objective=str(tcfg.get("search", "macro_f1")),
            grid_size=int(tcfg.get("grid_size", 101)),
            init=float(tcfg.get("init", 0.5)),
            coordinate_rounds=int(tcfg.get("coordinate_ascent_rounds", 3)),
        )
        save_thresholds(thresholds_payload, report_path / "thresholds.json")
        write_text(report_path / "thresholds.txt", describe_thresholds(thresholds_payload))
        thresholds_list = thresholds_payload["threshold_list"]
        LOGGER.info("\n%s", describe_thresholds(thresholds_payload))
    else:
        existing = report_path / "thresholds.json"
        if not args.force_argmax and not args.no_thresholds and existing.is_file():
            from .thresholds import load_thresholds

            thresholds_list = load_thresholds(existing)
            thresholds_payload = json.loads(existing.read_text(encoding="utf-8"))
            LOGGER.info("[eval] reusing frozen thresholds from %s", existing)

    decision_rule = str(cfg.get("evaluation", {}).get("decision_rule", "argmax"))
    if args.force_argmax or thresholds_list is None:
        decision_rule = "argmax"
        thresholds_list = None
    elif decision_rule == "argmax":
        LOGGER.warning(
            "[eval] config asks for decision_rule='argmax' although tuned thresholds exist - "
            "using the thresholds (set evaluation.decision_rule=thresholds to silence this)."
        )
        decision_rule = "thresholds"

    # validation metrics for the record
    val_pred = apply_decision_rule(val_outputs["y_proba"], decision_rule, thresholds_list)
    val_metrics = compute_metrics(val_outputs["y_true"], val_pred, val_outputs["y_proba"])
    val_pred_argmax = apply_decision_rule(val_outputs["y_proba"], "argmax")
    val_metrics_argmax = compute_metrics(val_outputs["y_true"], val_pred_argmax, val_outputs["y_proba"])

    # ---- scored pass: exactly one ------------------------------------- #
    # The loss under the training objective is accumulated inside that same
    # pass, so the scored split is never traversed twice.
    criterion = None
    try:
        criterion = build_loss(
            cfg.get("loss", {}),
            num_labels=int(cfg.get("dataset", {}).get("num_labels", NUM_LABELS)),
            class_weight_override=_frozen_class_weights(report_path),
            device=device,
        )
    except Exception as exc:  # pragma: no cover - optional reference number
        LOGGER.warning("[eval] could not build the training objective for the loss reference: %s", exc)

    scored_split = str(args.split)
    scored_loader = val_loader if scored_split == "validation" else test_loader
    LOGGER.info("[eval] running the single %s pass (rule=%s)", scored_split, decision_rule)
    scored_outputs = run_inference(model, scored_loader, device, amp_dtype=amp_dtype, criterion=criterion)
    y_true, y_proba = scored_outputs["y_true"], scored_outputs["y_proba"]
    y_pred = apply_decision_rule(y_proba, decision_rule, thresholds_list)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics_argmax = compute_metrics(y_true, apply_decision_rule(y_proba, "argmax"), y_proba)
    metrics.update(
        {
            "model_name": args.model,
            "run_id": run_id,
            "split": args.split,
            "decision_rule": decision_rule,
            "checkpoint": str(checkpoint),
            "num_test_samples": int(len(y_true)),
        }
    )

    # loss under the training objective (for reference only)
    if criterion is not None and "loss" in scored_outputs:
        metrics["test_loss_under_training_objective"] = float(scored_outputs["loss"])

    # ---- text reports -------------------------------------------------- #
    report_text = classification_report_text(y_true, y_pred)
    write_text(report_path / "classification_report.txt", report_text)
    write_text(report_path / "Best_model_classification_report.txt", report_text)
    write_text(report_path / "confusion_matrix.txt", confusion_matrix_text(y_true, y_pred))

    # ---- figures ------------------------------------------------------- #
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        report_path / "confusion_matrix.png",
        normalize=False,
        title=f"Confusion matrix (test set)\n{args.model}",
        model_name=args.model,
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        report_path / "confusion_matrix_raw.png",
        normalize=False,
        title=f"Confusion matrix - raw counts (test set)\n{args.model}",
        model_name=args.model,
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        report_path / "confusion_matrix_normalized.png",
        normalize=True,
        title=f"Confusion matrix - row-normalised (test set)\n{args.model}",
        model_name=args.model,
    )

    pr_data = precision_recall_curve_data(y_true, y_proba)
    roc_data = roc_curve_data(y_true, y_proba)
    dump_json(pr_data, report_path / "pr_curve_data.json")
    dump_json(roc_data, report_path / "roc_curve_data.json")
    plot_pr_curve(pr_data, report_path / "pr_curve.png", model_name=args.model)
    plot_roc_curve(roc_data, report_path / "roc_curve.png", model_name=args.model)
    plot_per_class_f1_bars(metrics, report_path / "per_class_f1.png", model_name=args.model)

    log_df = read_training_log(report_path)
    if len(log_df):
        plot_validation_accuracy_curve(
            log_df, report_path / "val_accuracy_curve.png", model_name=args.model,
            title=f"Validation accuracy per epoch\n{args.model}",
        )
        plot_validation_loss_curve(
            log_df, report_path / "val_loss_curve.png", model_name=args.model,
            title=f"Validation loss per epoch\n{args.model}",
        )
        plot_training_loss_curve(
            log_df, report_path / "train_loss_curve.png", model_name=args.model,
            title=f"Training loss per epoch\n{args.model}",
        )

    # ---- benchmark ------------------------------------------------------ #
    benchmark = None
    if not args.no_benchmark and bool(cfg.get("evaluation", {}).get("benchmark", True)):
        ecfg = cfg.get("evaluation", {}) or {}
        benchmark = benchmark_model(
            model,
            tokenizer,
            bundle.raw["test"][str(cfg.get("dataset", {}).get("text_column", "text"))].tolist(),
            device=device,
            batch_size=int(ecfg.get("benchmark_batch_size", 32)),
            max_length=int(cfg.get("tokenizer", {}).get("max_length", 256)),
            precision=str(ecfg.get("benchmark_precision", "fp32")),
            warmup_batches=int(ecfg.get("benchmark_warmup", 10)),
            timed_batches=int(ecfg.get("benchmark_runs", 50)),
            batch_sizes=ecfg.get("benchmark_batch_sizes"),
        )
        dump_json(benchmark, report_path / "benchmark.json")
    else:
        benchmark_path = report_path / "benchmark.json"
        if benchmark_path.is_file():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    # ---- predictions ---------------------------------------------------- #
    if bool(cfg.get("evaluation", {}).get("save_predictions", True)):
        predictions = predictions_dataframe(
            bundle.raw["test"][str(cfg.get("dataset", {}).get("text_column", "text"))].tolist(),
            y_true,
            y_pred,
            y_proba,
            indices=bundle.test.indices,
        )
        predictions.to_csv(report_path / "test_predictions.csv", index=False, encoding="utf-8")
        LOGGER.info("[eval] predictions -> %s", report_path / "test_predictions.csv")

    # ---- model summary -------------------------------------------------- #
    training_summary = {}
    summary_path = report_path / "training_summary.json"
    if summary_path.is_file():
        training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parameters = training_summary.get("parameters") or {
        **count_parameters(model),
        "estimated_parameter_memory_mb": round(
            count_parameters(model)["total_parameters"] * 4 / 1024**2, 2
        ),
        "head_only": count_parameters(model.head_module),
    }

    summary_text = build_model_summary(
        model_name=args.model,
        base_model=str(cfg.get("base_model", "")),
        head=str(cfg.get("head", "")),
        parameters=parameters,
        dataset={
            "name": cfg.get("dataset", {}).get("name"),
            "splits_dir": str(splits_dir),
            # sizes come from the manifest so the report shows the *full* split,
            # not the tokenized (possibly capped) subsets
            "num_train": int(split_manifest.get("num_train", len(bundle.train))),
            "num_validation": int(split_manifest.get("num_validation", len(bundle.validation))),
            "num_test": int(split_manifest.get("num_test", len(bundle.test))),
            "max_length": bundle.max_length,
        },
        training=training_summary.get("training", {}),
        benchmark=benchmark,
        metrics=metrics,
        decision_rule=decision_rule,
        thresholds=thresholds_payload,
        stages=training_summary.get("training", {}).get("stages"),
        environment=collect_environment(),
    )
    write_model_summary(report_path / "model_summary.txt", summary_text)

    # ---- metrics.json + evaluation.json --------------------------------- #
    metrics_payload = {
        **metrics,
        "metrics_with_argmax": {k: v for k, v in metrics_argmax.items() if k != "confusion_matrix"},
        "validation": {
            "decision_rule": decision_rule,
            **{k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
            "with_argmax": {k: v for k, v in val_metrics_argmax.items() if k != "confusion_matrix"},
        },
        "macro_average_precision_macro": metrics.get("macro_average_precision"),
    }
    dump_json(metrics_payload, report_path / "metrics.json")

    evaluation_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "decision_rule": decision_rule,
        "thresholds_file": str(report_path / "thresholds.json") if thresholds_payload else None,
        "metrics_file": str(report_path / "metrics.json"),
        "report_dir": str(report_path),
        "benchmark": benchmark,
        "environment": collect_environment(),
    }
    dump_json(evaluation_payload, report_path / "evaluation.json")

    # ---- evaluation/per_model mirror ------------------------------------ #
    mirror = ensure_dir(per_model_eval_dir(args.model))
    for name in (
        "metrics.json", "model_summary.txt", "classification_report.txt",
        "Best_model_classification_report.txt", "confusion_matrix.png",
        "confusion_matrix_raw.png", "confusion_matrix_normalized.png",
        "pr_curve.png", "roc_curve.png", "pr_curve_data.json", "roc_curve_data.json",
        "val_accuracy_curve.png", "val_loss_curve.png", "benchmark.json", "test_predictions.csv",
    ):
        src = report_path / name
        if src.is_file():
            shutil.copyfile(src, mirror / name)
    for name in ("training_log.csv", "config.yaml", "thresholds.json", "training_summary.json"):
        src = report_path / name
        if src.is_file():
            shutil.copyfile(src, mirror / name)
    LOGGER.info("[eval] mirrored artefacts -> %s", mirror)

    # ---- evaluation log -------------------------------------------------- #
    append_evaluation_log(
        {
            "timestamp": evaluation_payload["timestamp"],
            "model": args.model,
            "run_id": run_id,
            "checkpoint": str(checkpoint),
            "dataset": cfg.get("dataset", {}).get("name", ""),
            "split": args.split,
            "decision_rule": decision_rule,
            "num_test_samples": metrics["num_test_samples"],
            "accuracy": round(metrics["accuracy"], 6),
            "macro_precision": round(metrics["macro_precision"], 6),
            "macro_recall": round(metrics["macro_recall"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "weighted_f1": round(metrics["weighted_f1"], 6),
            "HATE_f1": round(metrics["HATE_f1"], 6),
            "OFFENSIVE_f1": round(metrics["OFFENSIVE_f1"], 6),
            "CLEAN_f1": round(metrics["CLEAN_f1"], 6),
            "roc_auc_macro": round(metrics.get("macro_roc_auc", float("nan")), 6),
            "average_precision_macro": round(metrics.get("macro_average_precision", float("nan")), 6),
            "latency_ms_per_sample": (benchmark or {}).get("latency_ms_per_sample", ""),
            "fps": (benchmark or {}).get("fps", ""),
            "parameters": parameters.get("total_parameters", ""),
            "best_epoch": training_summary.get("training", {}).get("best_epoch", ""),
        }
    )

    LOGGER.info("-" * 78)
    LOGGER.info("Test results for %s (%s, rule=%s)", args.model, run_id, decision_rule)
    LOGGER.info("  accuracy      : %.4f", metrics["accuracy"])
    LOGGER.info("  macro F1      : %.4f", metrics["macro_f1"])
    LOGGER.info("  weighted F1   : %.4f", metrics["weighted_f1"])
    LOGGER.info("  macro ROC-AUC : %.4f", metrics.get("macro_roc_auc", float("nan")))
    for name in LABELS:
        LOGGER.info("  %-10s P=%.4f R=%.4f F1=%.4f (n=%d)",
                    name, metrics[f"{name}_precision"], metrics[f"{name}_recall"],
                    metrics[f"{name}_f1"], metrics[f"{name}_support"])
    if benchmark:
        LOGGER.info("  latency       : %.4f ms/sample | %.1f FPS", benchmark["latency_ms_per_sample"], benchmark["fps"])
    LOGGER.info("  report dir    : %s", report_path)
    LOGGER.info("-" * 78)
    if args.model in MODEL_NAMES:
        remaining = [m for m in MODEL_NAMES if m != args.model]
        LOGGER.info("Remaining models: %s", ", ".join(remaining))
        LOGGER.info("When all four are done, build the comparison with:")
        LOGGER.info("  python -m src.comparison.compare_models")
    LOGGER.info("-" * 78)

    return evaluation_payload


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.warning("Interrupted by the user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
