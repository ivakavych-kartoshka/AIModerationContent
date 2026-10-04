"""Loss factory.

One entry point, :func:`build_loss`, turns a (possibly stage-specific) loss
config block into a ready-to-use ``nn.Module`` with the signature
``loss(logits, labels) -> scalar``.

Supported ``loss.type`` values
-----------------------------
``cross_entropy``
    Standard ``nn.CrossEntropyLoss``, optionally with label-specific class
    weights (plan section 10).
``focal``
    :class:`~src.losses.focal.FocalLoss` with ``gamma`` / ``alpha``
    (plan section 11).
``concept``
    :class:`~src.losses.curriculum.ConceptFocusedLoss` for the binary
    OFFENSIVE-/HATE-focused curriculum stages (plan section 13).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..common.config import get
from ..common.constants import NUM_LABELS
from .class_weights import compute_class_weights, weights_to_tensor
from .curriculum import ConceptFocusedLoss, positive_class_weight
from .focal import FocalLoss

VALID_TYPES = ("cross_entropy", "focal", "concept")


def build_loss(
    loss_cfg: Mapping[str, Any],
    num_labels: int = NUM_LABELS,
    train_labels: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    description: Optional[str] = None,
    class_weight_override: Optional[Mapping[str, float]] = None,
) -> nn.Module:
    """Instantiate the loss described by ``loss_cfg``.

    ``train_labels`` is required whenever class weights (or the concept's
    positive weight) must be estimated.  It must contain **training labels
    only** - the caller is responsible for that (see ``run_experiment``).
    ``class_weight_override`` injects already-computed weights (keyed by label
    name); evaluation uses it to reproduce the training objective without
    re-reading the training split.
    """
    loss_cfg = dict(loss_cfg or {})
    loss_type = str(loss_cfg.get("type", "cross_entropy")).lower()
    if loss_type not in VALID_TYPES:
        raise ValueError(f"loss.type must be one of {VALID_TYPES}, got {loss_type!r}")

    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)
    use_class_weight = bool(loss_cfg.get("use_class_weight", False))
    device = device or torch.device("cpu")

    weights_map: Optional[Dict[str, float]] = (
        {str(k): float(v) for k, v in class_weight_override.items()}
        if class_weight_override
        else None
    )
    if use_class_weight and weights_map is None:
        if train_labels is None:
            raise ValueError("use_class_weight=true requires the training labels")
        weights_map = compute_class_weights(
            train_labels,
            strategy=str(loss_cfg.get("class_weight", "inverse_frequency")),
            beta=float(loss_cfg.get("class_weight_beta", 0.999)),
            manual=loss_cfg.get("class_weight_manual"),
            num_labels=num_labels,
        )

    if loss_type == "cross_entropy":
        weight_tensor = (
            weights_to_tensor(weights_map, num_labels, device=device) if weights_map else None
        )
        module: nn.Module = nn.CrossEntropyLoss(
            weight=weight_tensor, label_smoothing=label_smoothing
        )

    elif loss_type == "focal":
        focal_cfg = loss_cfg.get("focal") or {}
        weight_tensor = (
            weights_to_tensor(weights_map, num_labels, device=device) if weights_map else None
        )
        module = FocalLoss(
            num_labels=num_labels,
            gamma=float(focal_cfg.get("gamma", 2.0)),
            alpha=focal_cfg.get("alpha"),
            class_weights=weight_tensor,
            label_smoothing=label_smoothing,
        )

    else:  # concept
        concept_cfg = loss_cfg.get("concept") or {}
        focus = concept_cfg.get("focus") or concept_cfg.get("focus_label")
        if not focus:
            raise ValueError("loss.type='concept' requires loss.concept.focus (e.g. OFFENSIVE)")
        if train_labels is None:
            raise ValueError("loss.type='concept' requires the training labels to set positive_weight")
        pos_weight_cfg = concept_cfg.get("positive_weight", "auto")
        if pos_weight_cfg == "auto" or pos_weight_cfg is None:
            pos_weight = positive_class_weight(
                train_labels,
                focus,
                num_labels=num_labels,
                normalize=bool(concept_cfg.get("normalize_positive_weight", True)),
                cap=concept_cfg.get("positive_weight_cap"),
            )
        else:
            pos_weight = float(pos_weight_cfg)
        module = ConceptFocusedLoss(
            focus_label=str(focus),
            num_labels=num_labels,
            positive_weight=pos_weight,
            gamma=float(concept_cfg.get("gamma", 0.0)),
        )

    module = module.to(device)
    if description:
        module = _DescribedLoss(module, description)  # type: ignore[assignment]
    return module


class _DescribedLoss(nn.Module):
    """Thin wrapper that carries a human-readable description of the loss."""

    def __init__(self, loss: nn.Module, description: str) -> None:
        super().__init__()
        self.loss = loss
        self.description = description

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, labels)

    def extra_repr(self) -> str:
        return f"description={self.description!r}"


def describe_loss(loss: nn.Module) -> str:
    if isinstance(loss, _DescribedLoss):
        return f"{loss.description}\n{loss.loss}"
    inner = getattr(loss, "loss", loss)
    return repr(inner)


def resolve_stage_loss(
    cfg: Mapping[str, Any],
    stage_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge the global ``loss`` block with a stage's ``loss`` overrides."""
    from ..common.config import deep_merge

    base = dict(get(cfg, "loss", {}) or {})
    if stage_cfg and stage_cfg.get("loss"):
        base = deep_merge(base, stage_cfg["loss"])
    return base


def class_weights_for_config(
    loss_cfg: Mapping[str, Any],
    train_labels: Sequence[int],
    num_labels: int = NUM_LABELS,
) -> Optional[Dict[str, float]]:
    """Class weights implied by ``loss_cfg`` (``None`` when disabled)."""
    if not loss_cfg.get("use_class_weight", False):
        return None
    return compute_class_weights(
        train_labels,
        strategy=str(loss_cfg.get("class_weight", "inverse_frequency")),
        beta=float(loss_cfg.get("class_weight_beta", 0.999)),
        manual=loss_cfg.get("class_weight_manual"),
        num_labels=num_labels,
    )


__all__ = [
    "VALID_TYPES",
    "build_loss",
    "class_weights_for_config",
    "describe_loss",
    "resolve_stage_loss",
]
