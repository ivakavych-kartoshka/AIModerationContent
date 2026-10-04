"""Focal loss (Lin et al., 2017) used by ``sensitiveai-vi-customdlr2stage``.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

* ``gamma`` down-weights easy samples (their ``(1-p_t)^gamma`` factor is close to 0);
* hard / misclassified samples keep a large contribution;
* ``alpha`` may be a scalar or one value per class (label-specific, plan section 11).

Both ``gamma`` and ``alpha`` are stored in the run config, as required.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.constants import ID2LABEL, NUM_LABELS


def _to_class_tensor(
    value: Union[float, Sequence[float], Mapping[str, float], torch.Tensor, None],
    num_labels: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    name: str = "alpha",
) -> Optional[torch.Tensor]:
    """Accept a scalar / list / ``{label_name: value}`` mapping -> tensor."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, Mapping):
        return torch.tensor(
            [float(value.get(ID2LABEL[i], 1.0)) for i in range(num_labels)],
            dtype=dtype,
            device=device,
        )
    if isinstance(value, (int, float)):
        return torch.full((num_labels,), float(value), dtype=dtype, device=device)
    seq = list(value)
    if len(seq) != num_labels:
        raise ValueError(f"{name} must have {num_labels} entries, got {len(seq)}")
    return torch.tensor([float(v) for v in seq], dtype=dtype, device=device)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights and label smoothing."""

    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        gamma: float = 2.0,
        alpha: Union[float, Sequence[float], Mapping[str, float], None] = None,
        class_weights: Union[torch.Tensor, Sequence[float], None] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")

        self.num_labels = num_labels
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self._alpha_spec = alpha
        self._weights_spec = class_weights
        self.register_buffer("alpha", _to_class_tensor(alpha, num_labels, torch.device("cpu")), persistent=False)
        self.register_buffer(
            "class_weights",
            None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    def _move(self, module: torch.nn.Module, device: torch.device) -> None:
        for name in ("alpha", "class_weights"):
            buf = getattr(self, name, None)
            if buf is not None:
                setattr(self, name, buf.to(device))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2 or logits.size(-1) != self.num_labels:
            raise ValueError(
                f"Expected logits of shape (B, {self.num_labels}), got {tuple(logits.shape)}"
            )
        self._move(self, logits.device)

        log_probs = F.log_softmax(logits, dim=-1)
        # log p_t : log-probability of the target class
        log_pt = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        if self.label_smoothing > 0:
            smooth = -log_probs.mean(dim=-1)
            log_pt = (1.0 - self.label_smoothing) * log_pt + self.label_smoothing * smooth

        pt = log_pt.exp()
        focal_term = (1.0 - pt).pow(self.gamma)

        loss = -focal_term * log_pt

        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[labels]
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device)[labels]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"num_labels={self.num_labels}, gamma={self.gamma}, reduction={self.reduction}"


__all__ = ["FocalLoss"]
