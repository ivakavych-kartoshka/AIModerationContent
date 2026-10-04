"""Training Controller: owns the Stage 1 -> Stage 2 -> ... transitions.

Plan section 8.3 requires the transition between stages to be handled by a
callback / training controller rather than by two unrelated scripts.  This class
is that controller: it decides when a stage is over, persists the stage
checkpoint, restores a stage's ``init_from`` checkpoint and keeps a
machine-readable transition log.

Guarantees:

* epoch numbering is **global and continuous** (stage 2 epoch 1 is epoch
  ``stage_1.epochs + 1``), so the validation-accuracy / validation-loss figures
  of the multi-stage models are directly comparable with the single-stage ones;
* ``epoch_in_stage`` is logged as well, for per-stage analysis;
* every transition is written to ``stage_transitions.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from ..common.folder_docs import ensure_explain
from ..common.logging_utils import get_logger

if TYPE_CHECKING:  # avoids a circular import with src.training.stages
    from ..training.stages import StageSpec

LOGGER = get_logger(__name__)


class StageController:
    """Drives the stage sequence of a run."""

    def __init__(
        self,
        stages: Sequence[StageSpec],
        checkpoints_root: Optional[Path] = None,
        save_transitions_to: Optional[Path] = None,
    ) -> None:
        if not stages:
            raise ValueError("StageController needs at least one stage")
        self.stages: List[StageSpec] = list(stages)
        self.checkpoints_root = Path(checkpoints_root) if checkpoints_root else None
        self.transitions_path = Path(save_transitions_to) if save_transitions_to else None

        self.index = 0
        self.epoch_in_stage = 0
        self.global_epoch = 0
        self.transitions: List[Dict[str, Any]] = []
        self.stop_requested = False
        # whether the most recent ``end_stage`` advanced into a further stage
        self.advanced = False

    # ------------------------------------------------------------------ #
    @property
    def current(self) -> StageSpec:
        return self.stages[self.index]

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def total_epochs(self) -> int:
        return sum(s.num_epochs for s in self.stages)

    @property
    def is_last_stage(self) -> bool:
        return self.index == len(self.stages) - 1

    def is_stage_finished(self) -> bool:
        return self.epoch_in_stage >= self.current.num_epochs

    # ------------------------------------------------------------------ #
    def begin_train(self) -> None:
        self._record("train_begin", {"stages": [s.name for s in self.stages],
                                     "total_epochs": self.total_epochs})
        self.begin_stage()

    def begin_stage(self) -> StageSpec:
        stage = self.current
        self.epoch_in_stage = 0
        LOGGER.info(
            "\n=== STAGE %d/%d :: %s ===\n%s",
            self.index + 1, self.num_stages, stage.name, stage.describe(),
        )
        self._record("stage_begin", {"stage": stage.name, "index": self.index + 1,
                                     "spec": stage.to_dict()})
        return stage

    def begin_epoch(self) -> int:
        """Advance the epoch counters and return the global epoch number."""
        self.epoch_in_stage += 1
        self.global_epoch += 1
        return self.global_epoch

    def end_stage(self, model=None) -> Optional[Path]:
        """Persist the stage checkpoint (when requested) and record the transition."""
        stage = self.current
        saved: Optional[Path] = None
        if stage.save_checkpoint and self.checkpoints_root is not None and model is not None:
            saved = self.checkpoints_root / f"{stage.name}"
            model.save_pretrained(saved)
            ensure_explain(saved)
            ensure_explain(self.checkpoints_root)
            LOGGER.info("[stage] checkpoint for %s -> %s", stage.name, saved)
        self._record(
            "stage_end",
            {
                "stage": stage.name,
                "index": self.index + 1,
                "epochs_run": self.epoch_in_stage,
                "last_global_epoch": self.global_epoch,
                "checkpoint": str(saved) if saved else None,
            },
        )
        if not self.is_last_stage:
            self.index += 1
            self.advanced = True
            LOGGER.info("[stage] transition %s -> %s", stage.name, self.current.name)
        else:
            self.advanced = False
        return saved

    def next_stage_exists(self) -> bool:
        """``True`` when the last :meth:`end_stage` moved on to another stage.

        ``end_stage`` advances ``self.index`` itself, so this must report the
        transition that just happened - checking ``index`` against the stage
        count would always look at the *next* stage and skip the final one.
        """
        return self.advanced

    def end_train(self, reason: str = "completed") -> None:
        self._record("train_end", {"reason": reason, "epochs_run": self.global_epoch})

    # ------------------------------------------------------------------ #
    def _record(self, event: str, payload: Dict[str, Any]) -> None:
        entry = {"event": event, "global_epoch": self.global_epoch, **payload}
        self.transitions.append(entry)
        LOGGER.debug("[stage-controller] %s", entry)
        if self.transitions_path is not None:
            self.transitions_path.parent.mkdir(parents=True, exist_ok=True)
            self.transitions_path.write_text(
                json.dumps(self.transitions, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "num_stages": self.num_stages,
            "total_epochs_planned": self.total_epochs,
            "epochs_run": self.global_epoch,
            "stages": [
                {
                    "name": s.name,
                    "description": s.description,
                    "focus_label": s.focus_label,
                    "epochs": s.num_epochs,
                    "global_epoch_start": s.global_epoch_start,
                    "global_epoch_end": s.global_epoch_end,
                    "init_from": s.init_from,
                    "save_checkpoint": s.save_checkpoint,
                    "training": s.training,
                    "dlr": s.dlr,
                    "loss": s.loss,
                }
                for s in self.stages
            ],
            "transitions": self.transitions,
        }


__all__ = ["StageController"]
