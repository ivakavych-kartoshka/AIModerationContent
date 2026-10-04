"""Stage specifications for multi-stage training.

Plan section 8.3 (``sensitiveai-vi-customdlr2stage``) and section 13
(``sensitiveai-vi-custom-curriculum``) both describe training as a sequence of
stages.  A stage is a self-contained training recipe:

    name              unique id used in the logs ("stage_1", "stage_2", ...)
    description       human readable summary stored in the config
    epochs            how many epochs this stage runs
    focus_label       curriculum concept focus (None -> plain 3-class)
    init_from         checkpoint the stage starts from (None -> continue in place)
    save_checkpoint   write ``stage_<n>`` weights when the stage ends
    training          overrides of the global ``training`` block
    dlr               overrides of the global ``dlr`` block
    loss              overrides of the global ``loss`` block

When a config has no ``stages`` key, a single implicit stage is derived from the
top-level ``training`` / ``dlr`` / ``loss`` blocks - which is exactly what
``sensitiveai-vi`` and ``sensitiveai-vi-custom`` use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..common.config import deep_merge, get
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

TRAINING_KEYS = (
    "batch_size",
    "eval_batch_size",
    "gradient_accumulation",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "optimizer",
    "max_grad_norm",
    "scheduler",
    "precision",
    "label_smoothing",
)
DLR_KEYS = ("enabled", "encoder_lr", "layer_lr", "head_lr", "embedding_lr", "layer_split", "layer_decay")
LOSS_KEYS = ("type", "use_class_weight", "class_weight", "class_weight_manual", "class_weight_beta", "focal", "concept", "label_smoothing")


@dataclass
class StageSpec:
    """One stage of a (possibly multi-stage) training run."""

    name: str
    description: str = ""
    epochs: int = 1
    focus_label: Optional[str] = None
    init_from: Optional[str] = None
    save_checkpoint: bool = False
    training: Dict[str, Any] = field(default_factory=dict)
    dlr: Dict[str, Any] = field(default_factory=dict)
    loss: Dict[str, Any] = field(default_factory=dict)

    # filled in by the trainer at runtime
    global_epoch_start: int = 0
    global_epoch_end: int = 0

    @property
    def num_epochs(self) -> int:
        return int(self.epochs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        bits = [f"{self.name} ({self.epochs} epoch{'s' if self.epochs != 1 else ''})"]
        if self.focus_label:
            bits.append(f"focus={self.focus_label}")
        if self.dlr.get("enabled"):
            bits.append(
                "DLR[enc={encoder_lr}, layer={layer_lr}, head={head_lr}]".format(
                    encoder_lr=self.dlr.get("encoder_lr"),
                    layer_lr=self.dlr.get("layer_lr"),
                    head_lr=self.dlr.get("head_lr"),
                )
            )
        else:
            bits.append(f"lr={self.training.get('learning_rate')}")
        bits.append(f"loss={self.loss.get('type', 'cross_entropy')}")
        if self.init_from:
            bits.append(f"init_from={self.init_from}")
        return " | ".join(bits)


def _subset(source: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {k: source[k] for k in keys if k in source and source[k] is not None}


def build_single_stage(cfg: Mapping[str, Any], name: str = "single") -> StageSpec:
    """The implicit single-stage recipe used by the baseline models."""
    return StageSpec(
        name=name,
        description="single-stage training on the 3-class task",
        epochs=int(get(cfg, "training.epochs", 1)),
        focus_label=None,
        init_from=None,
        save_checkpoint=False,
        training=_subset(get(cfg, "training", {}) or {}, TRAINING_KEYS),
        dlr=_subset(get(cfg, "dlr", {}) or {}, DLR_KEYS),
        loss=_subset(get(cfg, "loss", {}) or {}, LOSS_KEYS),
    )


def resolve_stages(cfg: Mapping[str, Any]) -> List[StageSpec]:
    """Turn the config's ``stages`` list into :class:`StageSpec` objects."""
    raw_stages: Optional[List[Mapping[str, Any]]] = cfg.get("stages")

    if not raw_stages:
        stage = build_single_stage(cfg)
        stage.global_epoch_start = 1
        stage.global_epoch_end = stage.num_epochs
        return [stage]

    global_training = get(cfg, "training", {}) or {}
    global_dlr = get(cfg, "dlr", {}) or {}
    global_loss = get(cfg, "loss", {}) or {}

    stages: List[StageSpec] = []
    epoch_cursor = 1
    for i, raw in enumerate(raw_stages, start=1):
        default_name = f"stage_{i}"
        stage_training = deep_merge(global_training, raw.get("training") or {})
        stage_dlr = deep_merge(global_dlr, raw.get("dlr") or {})
        stage_loss = deep_merge(global_loss, raw.get("loss") or {})
        focus = raw.get("focus_label")
        if focus is None and str(stage_loss.get("type", "cross_entropy")).lower() == "concept":
            concept_cfg = stage_loss.get("concept") or {}
            focus = concept_cfg.get("focus") or concept_cfg.get("focus_label")

        spec = StageSpec(
            name=str(raw.get("name", default_name)),
            description=str(raw.get("description", "")),
            epochs=int(raw.get("epochs", global_training.get("epochs", 1))),
            focus_label=focus,
            init_from=raw.get("init_from"),
            save_checkpoint=bool(raw.get("save_checkpoint", True)),
            training=_subset(stage_training, TRAINING_KEYS),
            dlr=_subset(stage_dlr, DLR_KEYS),
            loss=_subset(stage_loss, LOSS_KEYS),
        )
        if spec.num_epochs < 1:
            raise ValueError(f"Stage {spec.name!r} has epochs={spec.num_epochs}; must be >= 1")
        spec.global_epoch_start = epoch_cursor
        epoch_cursor += spec.num_epochs
        spec.global_epoch_end = epoch_cursor - 1
        stages.append(spec)

    if not any(s.init_from for s in stages):
        stages[0].init_from = None  # stage 1 always starts from the base model

    return stages


def stages_summary(stages: Sequence[StageSpec]) -> str:
    total = sum(s.num_epochs for s in stages)
    lines = [f"Training plan: {len(stages)} stage(s), {total} epoch(s) total", "-" * 72]
    for s in stages:
        lines.append(
            f"  [{s.name}] epochs {s.global_epoch_start}-{s.global_epoch_end} :: {s.describe()}"
        )
    return "\n".join(lines)


__all__ = [
    "DLR_KEYS",
    "LOSS_KEYS",
    "TRAINING_KEYS",
    "StageSpec",
    "build_single_stage",
    "resolve_stages",
    "stages_summary",
]
