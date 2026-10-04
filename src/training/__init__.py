"""Training package: stages, optimizer/DLR, trainer, CLI."""

from .optim import (  # noqa: F401
    build_optimizer,
    build_param_groups,
    build_scheduler,
    representative_lrs,
)
from .stages import StageSpec, build_single_stage, resolve_stages, stages_summary  # noqa: F401
from .trainer import TrainResult, Trainer  # noqa: F401

__all__ = [
    "StageSpec",
    "TrainResult",
    "Trainer",
    "build_optimizer",
    "build_param_groups",
    "build_scheduler",
    "build_single_stage",
    "representative_lrs",
    "resolve_stages",
    "stages_summary",
]
