"""Per-model report generation (``model_summary.txt`` and friends).

Plan section 27 requires every model to report: name, base model, total /
trainable / non-trainable parameters, estimated parameter memory, dataset sizes,
best epoch, training time, average inference latency, FPS, GPU, CUDA version,
PyTorch version and Transformers version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..common.constants import LABELS
from ..common.env_info import collect_environment, format_environment
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

WIDTH = 74


def _hms(seconds: float) -> str:
    seconds = int(max(seconds or 0, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _row(label: str, value: Any) -> str:
    return f"{label:<34}: {value}"


def build_model_summary(
    model_name: str,
    base_model: str,
    head: str,
    parameters: Mapping[str, Any],
    dataset: Mapping[str, Any],
    training: Mapping[str, Any],
    benchmark: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    decision_rule: str = "argmax",
    thresholds: Optional[Mapping[str, Any]] = None,
    stages: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, Any]] = None,
) -> str:
    env = dict(environment or collect_environment())
    head_only = parameters.get("head_only") or {}
    bench = dict(benchmark or {})

    lines: list[str] = []
    lines.append("=" * WIDTH)
    lines.append(f"Model summary - {model_name}")
    lines.append("=" * WIDTH)
    lines.append("")
    lines.append("[Model]")
    lines.append(_row("Model Name", model_name))
    lines.append(_row("Base Model", base_model))
    lines.append(_row("Classification Head", head))
    if stages:
        lines.append(_row("Training Stages", f"{len(stages.get('stages', []))} stage(s)"))
        for stage in stages.get("stages", []):
            focus = stage.get("focus_label")
            lines.append(
                _row(
                    f"  stage {stage.get('name')}",
                    f"epochs {stage.get('global_epoch_start')}-{stage.get('global_epoch_end')}"
                    f"{f', focus={focus}' if focus else ''}",
                )
            )
    lines.append("")

    lines.append("[Parameters]")
    lines.append(_row("Total Parameters", f"{parameters.get('total_parameters', 0):,}"))
    lines.append(_row("Trainable Parameters", f"{parameters.get('trainable_parameters', 0):,}"))
    lines.append(_row("Non-trainable Parameters", f"{parameters.get('non_trainable_parameters', 0):,}"))
    lines.append(_row("Estimated Parameter Memory", f"{parameters.get('estimated_parameter_memory_mb', 0):.2f} MB (fp32 weights)"))
    if head_only:
        lines.append(_row("  head parameters", f"{head_only.get('total_parameters', 0):,}"))
    if bench.get("peak_gpu_memory_mb") is not None:
        peak = bench.get("peak_gpu_memory_mb")
        lines.append(_row("Peak GPU Memory (inference)", f"{peak:.2f} MB" if isinstance(peak, float) else peak))
    lines.append("")

    lines.append("[Dataset]")
    lines.append(_row("Dataset", dataset.get("name", "uitnlp/vihsd")))
    lines.append(_row("Split directory", dataset.get("splits_dir", "")))
    lines.append(_row("Label Mapping", "HATE=0, OFFENSIVE=1, CLEAN=2"))
    lines.append(_row("Training Samples", f"{dataset.get('num_train', 0):,}"))
    lines.append(_row("Validation Samples", f"{dataset.get('num_validation', 0):,}"))
    lines.append(_row("Test Samples", f"{dataset.get('num_test', 0):,}"))
    lines.append(_row("Max sequence length", dataset.get("max_length", "")))
    lines.append("")

    lines.append("[Training]")
    lines.append(_row("Best Metric", training.get("best_metric", "")))
    lines.append(_row("Best Value", f"{training['best_value']:.6f}" if isinstance(training.get("best_value"), float) else "n/a"))
    lines.append(_row("Best Epoch", training.get("best_epoch", "n/a")))
    lines.append(_row("Best Stage", training.get("best_stage", "")))
    lines.append(_row("Epochs Run", f"{training.get('epochs_run', 0)} / {training.get('total_epochs_planned', 0)}"))
    lines.append(_row("Training Time", f"{_hms(training.get('training_seconds', 0))} (h:mm:ss)"))
    lines.append(_row("Early Stopped", training.get("stopped_early", False)))
    lines.append("")

    lines.append("[Inference Benchmark]")
    if bench:
        lines.append(_row("Batch Size", bench.get("batch_size")))
        lines.append(_row("Average Latency / Batch", f"{bench.get('latency_ms_per_batch', float('nan')):.3f} ms"))
        lines.append(_row("Average Latency / Sample", f"{bench.get('latency_ms_per_sample', float('nan')):.4f} ms"))
        lines.append(_row("Latency p95 / Batch", f"{bench.get('latency_p95_ms_per_batch', float('nan')):.3f} ms"))
        lines.append(_row("FPS (samples / second)", f"{bench.get('fps', float('nan')):.2f}"))
        lines.append(_row("Precision", bench.get("precision")))
        lines.append(_row("Timed Batches", bench.get("timed_batches")))
    else:
        lines.append(_row("Benchmark", "not run"))
    lines.append("")

    if metrics:
        lines.append("[Test Results]  (decision rule: " + decision_rule + ")")
        lines.append(_row("Accuracy", f"{metrics.get('accuracy', float('nan')):.4f}"))
        lines.append(_row("Macro Precision", f"{metrics.get('macro_precision', float('nan')):.4f}"))
        lines.append(_row("Macro Recall", f"{metrics.get('macro_recall', float('nan')):.4f}"))
        lines.append(_row("Macro F1", f"{metrics.get('macro_f1', float('nan')):.4f}"))
        lines.append(_row("Weighted F1", f"{metrics.get('weighted_f1', float('nan')):.4f}"))
        lines.append(_row("Macro ROC-AUC", f"{metrics.get('macro_roc_auc', float('nan')):.4f}"))
        lines.append(_row("Macro Average Precision", f"{metrics.get('macro_average_precision', float('nan')):.4f}"))
        lines.append("")
        lines.append(f"{'Class':<12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
        for name in LABELS:
            lines.append(
                f"{name:<12}"
                f"{metrics.get(f'{name}_precision', float('nan')):>12.4f}"
                f"{metrics.get(f'{name}_recall', float('nan')):>12.4f}"
                f"{metrics.get(f'{name}_f1', float('nan')):>12.4f}"
                f"{metrics.get(f'{name}_support', 0):>12d}"
            )
        lines.append("")

    if thresholds:
        lines.append("[Label-specific Thresholds]  (searched on VALIDATION only)")
        for name in LABELS:
            lines.append(_row(f"threshold {name}", thresholds.get("thresholds", {}).get(name)))
        lines.append(_row("objective before -> after",
                          f"{thresholds.get('objective_before')} -> {thresholds.get('objective_after')}"))
        lines.append("")

    lines.append("[Hardware / Software]")
    lines.append(_row("GPU", env.get("gpu")))
    lines.append(_row("GPU Memory (GB)", env.get("gpu_memory_gb")))
    lines.append(_row("CUDA Version (torch)", env.get("cuda_version")))
    lines.append(_row("cuDNN Version", env.get("cudnn_version")))
    lines.append(_row("NVIDIA Driver", env.get("nvidia_driver")))
    lines.append(_row("PyTorch Version", env.get("torch")))
    lines.append(_row("Transformers Version", env.get("transformers")))
    lines.append(_row("scikit-learn Version", env.get("sklearn")))
    lines.append(_row("Python Version", env.get("python")))
    lines.append(_row("Platform", env.get("platform")))
    lines.append("")
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def write_model_summary(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    LOGGER.info("[report] %s", out)
    return out


def write_text(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    LOGGER.info("[report] %s", out)
    return out


def environment_block() -> str:
    return format_environment()


__all__ = ["build_model_summary", "environment_block", "write_model_summary", "write_text"]
