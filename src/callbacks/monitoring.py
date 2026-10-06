"""Early stopping and best-checkpoint tracking (plan section 5).

Both decisions are made on the **validation** metric configured by
``early_stopping.metric`` (default ``macro_f1``).  The test split is never
consulted here - that is the whole point of the data-leakage rules in the plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..common.logging_utils import get_logger
from .base import Callback

LOGGER = get_logger(__name__)

HIGHER_IS_BETTER = {"macro_f1", "weighted_f1", "accuracy", "macro_roc_auc", "macro_average_precision", "f1"}
LOWER_IS_BETTER = {"loss", "val_loss"}


def resolve_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "max"
    if metric in LOWER_IS_BETTER:
        return "min"
    raise ValueError(
        f"Unknown early_stopping.metric {metric!r}. Use one of {sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)}."
    )


def is_better(value: float, best: Optional[float], mode: str) -> bool:
    if best is None or value != value:
        return True
    return value > best if mode == "max" else value < best


class EarlyStoppingCallback(Callback):
    """Stop training when the monitored validation metric stops improving."""

    name = "early_stopping"

    def __init__(
        self,
        metric: str = "macro_f1",
        mode: Optional[str] = None,
        patience: int = 2,
        min_delta: float = 0.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.mode = mode or resolve_direction(metric)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.enabled = bool(enabled)

        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.num_bad_epochs = 0
        self.should_stop = False
        self.stage_resets: list[Dict[str, Any]] = []
        self.history: list[Dict[str, Any]] = []

    def observe(self, epoch: int, value: float) -> bool:
        """Update the internal state; return ``True`` when this epoch is a new best."""
        if value != value:  # NaN
            return False
        improved = is_better(value, self.best_value, self.mode)
        if improved and (self.best_value is None or abs(value - self.best_value) >= self.min_delta):
            self.best_value = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        self.history.append({"epoch": epoch, "value": float(value), "is_best": bool(improved)})
        return improved and self.best_epoch == epoch

    def on_stage_begin(self, trainer: Any = None, stage: Any = None, **_: Any) -> None:
        """Give the new stage its own plateau budget.

        Every stage is a different optimisation problem - the loss may change
        (cross-entropy -> focal, concept OFFENSIVE -> concept HATE) and the
        learning rates are usually lowered - so the first epochs of a stage
        legitimately score below the best value of the *previous* stage.  The
        ``num_bad_epochs`` counter is therefore reset here instead of being
        inherited, otherwise a plateau that started before the transition would
        stop the whole run and the later stages would never execute.

        ``best_value`` is deliberately *not* reset: it still tracks the best
        epoch of the entire run, which is what ``best_model/`` must contain.
        """
        if not self.enabled:
            return
        if self.num_bad_epochs:
            LOGGER.info(
                "[early-stop] %s carries a %d-epoch plateau into the next stage - "
                "counter reset, %d more allowed",
                self.metric, self.num_bad_epochs, self.patience,
            )
            self.stage_resets.append(
                {
                    "stage": getattr(stage, "name", None),
                    "epoch": getattr(stage, "global_epoch_start", None),
                    "bad_epochs_carried": self.num_bad_epochs,
                }
            )
        self.num_bad_epochs = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Any = None, epoch: int = 0, metrics: Dict[str, Any] | None = None,
                     stage: Any = None, **_: Any) -> None:
        if not self.enabled:
            return
        metrics = metrics or {}
        value = metrics.get(self.metric)
        if value is None:
            value = metrics.get(f"val_{self.metric}")
        if value is None:
            LOGGER.warning("[early-stop] metric %r not found in epoch metrics", self.metric)
            return
        improved = self.observe(epoch, float(value))
        if improved:
            LOGGER.info("[early-stop] new best %s=%.6f at epoch %d", self.metric, float(value), epoch)
        else:
            LOGGER.info(
                "[early-stop] %s=%.6f (best %.6f @ epoch %s) - %d/%d epochs without improvement",
                self.metric, float(value),
                self.best_value if self.best_value is not None else float("nan"),
                self.best_epoch, self.num_bad_epochs, self.patience,
            )
        if self.patience > 0 and self.num_bad_epochs >= self.patience:
            self.should_stop = True
            LOGGER.info("[early-stop] stopping after epoch %d (patience exhausted)", epoch)

    def summary(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "stopped_early": bool(self.should_stop),
            "stage_resets": self.stage_resets,
            "history": self.history,
        }


class BestCheckpointCallback(Callback):
    """Persist the weights of the best validation epoch to ``best_model/``."""

    name = "best_checkpoint"

    def __init__(self, output_dir: str | Path, metric: str = "macro_f1", mode: Optional[str] = None,
                 save: bool = True) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.mode = mode or resolve_direction(metric)
        self.save = bool(save)
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_stage: Optional[str] = None
        self.history: list[Dict[str, Any]] = []

    def on_epoch_end(self, trainer: Any = None, epoch: int = 0, metrics: Dict[str, Any] | None = None,
                     stage: Any = None, **_: Any) -> None:
        metrics = metrics or {}
        value = metrics.get(self.metric, metrics.get(f"val_{self.metric}"))
        if value is None or value != value:
            return
        value = float(value)
        improved = is_better(value, self.best_value, self.mode)
        self.history.append({"epoch": epoch, "value": value, "is_best": bool(improved)})
        metrics["is_best"] = bool(improved)
        if not improved:
            return

        self.best_value = value
        self.best_epoch = epoch
        self.best_stage = getattr(stage, "name", None)
        if self.save and trainer is not None:
            path = self.output_dir / "best_model"
            trainer.save_model(path)
            trainer.save_training_state(self.output_dir / "best_training_state.pt")
            LOGGER.info("[best] %s=%.6f at epoch %d -> %s", self.metric, value, epoch, path)

    def summary(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_stage": self.best_stage,
            "saved_to": str(self.output_dir / "best_model"),
            "history": self.history,
        }


__all__ = ["BestCheckpointCallback", "EarlyStoppingCallback", "is_better", "resolve_direction"]
