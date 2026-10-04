"""Evaluation: metrics, curves, figures, thresholds, benchmarking, reports."""

from .curves import precision_recall_curve_data, roc_curve_data  # noqa: F401
from .metrics import (  # noqa: F401
    apply_decision_rule,
    classification_report_text,
    compute_metrics,
    confusion_matrix_text,
    evaluate_model,
    predictions_dataframe,
    run_inference,
)
from .report import build_model_summary, write_model_summary  # noqa: F401
from .thresholds import optimize_thresholds, save_thresholds  # noqa: F401

__all__ = [
    "apply_decision_rule",
    "build_model_summary",
    "classification_report_text",
    "compute_metrics",
    "confusion_matrix_text",
    "evaluate_model",
    "optimize_thresholds",
    "precision_recall_curve_data",
    "predictions_dataframe",
    "roc_curve_data",
    "run_inference",
    "save_thresholds",
    "write_model_summary",
]
