"""``training_log.csv`` writer (plan section 17).

One row per epoch, with the columns required by the plan plus the extra columns
needed by the DLR / curriculum variants.  Every number in the validation-accuracy
and validation-loss figures of the paper is read from this file - nothing is
re-plotted from an in-memory buffer.

Columns
-------
always
    ``model_name, stage, stage_index, epoch, epoch_in_stage, learning_rate,
    encoder_lr, head_lr, loss, accuracy, val_loss, val_accuracy,
    val_macro_f1, val_weighted_f1, grad_norm, epoch_seconds, cumulative_seconds,
    is_best``
per-class validation detail (macro friendly)
    ``val_HATE_f1, val_OFFENSIVE_f1, val_CLEAN_f1, val_HATE_recall,
    val_OFFENSIVE_recall, val_CLEAN_recall``
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from ..common.logging_utils import get_logger
from .base import Callback

LOGGER = get_logger(__name__)

BASE_COLUMNS: Sequence[str] = (
    "model_name",
    "stage",
    "stage_index",
    "epoch",
    "epoch_in_stage",
    "learning_rate",
    "encoder_lr",
    "head_lr",
    "loss",
    "accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "grad_norm",
    "epoch_seconds",
    "cumulative_seconds",
    "is_best",
)

EXTRA_COLUMNS: Sequence[str] = (
    "val_HATE_precision",
    "val_HATE_recall",
    "val_HATE_f1",
    "val_OFFENSIVE_precision",
    "val_OFFENSIVE_recall",
    "val_OFFENSIVE_f1",
    "val_CLEAN_precision",
    "val_CLEAN_recall",
    "val_CLEAN_f1",
    "val_macro_roc_auc",
)

COLUMNS: Sequence[str] = BASE_COLUMNS + EXTRA_COLUMNS

PER_CLASS = ("HATE", "OFFENSIVE", "CLEAN")


def format_number(value: Any, digits: int = 6) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return round(value, digits)
    return value


class CSVLoggerCallback(Callback):
    """Append one CSV row per epoch to ``<report_dir>/training_log.csv``."""

    name = "csv_logger"

    def __init__(self, path: str | Path, model_name: str = "", columns: Sequence[str] = COLUMNS) -> None:
        super().__init__()
        self.path = Path(path)
        self.model_name = model_name
        self.columns = list(columns)
        self.rows: list[Dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # truncate: a run_id is never reused
        with self.path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.columns).writeheader()

    # ------------------------------------------------------------------ #
    def build_row(self, trainer: Any, epoch: int, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        stage = getattr(trainer, "current_stage", None)
        row: Dict[str, Any] = {
            "model_name": self.model_name or getattr(trainer, "model_name", ""),
            "stage": getattr(stage, "name", "single"),
            "stage_index": getattr(trainer, "current_stage_index", 1),
            "epoch": epoch,
            "epoch_in_stage": getattr(trainer, "epoch_in_stage", 1),
            "learning_rate": format_number(getattr(trainer, "current_lr", None)),
            "encoder_lr": format_number(getattr(trainer, "current_encoder_lr", None)),
            "head_lr": format_number(getattr(trainer, "current_head_lr", None)),
            "loss": format_number(metrics.get("loss")),
            "accuracy": format_number(metrics.get("accuracy")),
            "val_loss": format_number(metrics.get("val_loss")),
            "val_accuracy": format_number(metrics.get("val_accuracy")),
            "val_macro_f1": format_number(metrics.get("val_macro_f1")),
            "val_weighted_f1": format_number(metrics.get("val_weighted_f1")),
            "grad_norm": format_number(getattr(trainer, "last_grad_norm", None)),
            "epoch_seconds": format_number(getattr(trainer, "last_epoch_seconds", None), 3),
            "cumulative_seconds": format_number(getattr(trainer, "elapsed_seconds", None), 3),
            "is_best": format_number(bool(metrics.get("is_best", False))),
        }
        for name in PER_CLASS:
            for metric in ("precision", "recall", "f1"):
                row[f"val_{name}_{metric}"] = format_number(metrics.get(f"val_{name}_{metric}"))
        row["val_macro_roc_auc"] = format_number(metrics.get("val_macro_roc_auc"))
        return {k: row.get(k, "") for k in self.columns}

    def on_epoch_end(self, trainer: Any = None, epoch: int = 0, metrics: Dict[str, Any] | None = None,
                     stage: Any = None, **_: Any) -> None:
        row = self.build_row(trainer, epoch, metrics or {})
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.columns).writerow(row)
        LOGGER.info(
            "[log] stage=%s epoch=%d loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f%s",
            row["stage"],
            epoch,
            float(row["loss"] or 0.0),
            float(row["val_loss"] or 0.0),
            float(row["val_accuracy"] or 0.0),
            float(row["val_macro_f1"] or 0.0),
            "  *best*" if row["is_best"] else "",
        )

    # ------------------------------------------------------------------ #
    def dataframe(self):
        """The log as a pandas DataFrame (used by the plotting code)."""
        import pandas as pd

        return pd.DataFrame(self.rows, columns=self.columns)

    def read(self) -> "Any":
        import pandas as pd

        if not self.path.is_file():
            return pd.DataFrame(columns=list(self.columns))
        return pd.read_csv(self.path)


def append_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str] = COLUMNS) -> Path:
    """Append rows to an existing log (used when merging runs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return path


__all__ = ["BASE_COLUMNS", "COLUMNS", "EXTRA_COLUMNS", "CSVLoggerCallback", "append_rows", "format_number"]
