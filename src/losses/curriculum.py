"""Concept-focused binary loss used by the curriculum stages (plan section 13).

The curriculum keeps a **single 3-way head** (so the label mapping never
changes and no head has to be reshaped between stages) and re-expresses it as a
binary concept for the focus label ``c``:

    z_pos = logits[:, c]
    z_neg = logsumexp(logits[:, all other classes])
    binary logits = [z_neg, z_pos]
    target = 1 if y == c else 0

Consequences, matching the plan exactly:

* ``Stage 1`` -> focus = ``OFFENSIVE``: OFFENSIVE is positive, HATE and CLEAN are
  negative;
* ``Stage 2`` -> focus = ``HATE``: HATE is positive, OFFENSIVE and CLEAN are
  negative;
* the **whole dataset** is used in every stage - negatives are the other two
  classes, we never train on OFFENSIVE samples only;
* the softmax over ``z_neg`` keeps the relative ordering of the two negative
  classes in the gradient, so no information about HATE vs CLEAN is discarded.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.constants import ID2LABEL, LABEL2ID, NUM_LABELS


class ConceptFocusedLoss(nn.Module):
    """Binary concept loss computed from a 3-class logit vector."""

    def __init__(
        self,
        focus_label: str,
        num_labels: int = NUM_LABELS,
        positive_weight: Optional[float] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if focus_label not in LABEL2ID:
            raise ValueError(f"focus_label must be one of {sorted(LABEL2ID)}, got {focus_label!r}")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be mean|sum, got {reduction!r}")

        self.focus_label = focus_label
        self.num_labels = num_labels
        self.positive_index = LABEL2ID[focus_label]
        self.negative_index = [i for i in range(num_labels) if i != self.positive_index]
        self.gamma = float(gamma)
        self.reduction = reduction
        self.positive_weight = None if positive_weight is None else float(positive_weight)
        self.register_buffer(
            "class_weight",
            None
            if self.positive_weight is None
            else torch.tensor([1.0, self.positive_weight], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.size(-1) != self.num_labels:
            raise ValueError(
                f"Expected {self.num_labels}-way logits for the concept loss, got {tuple(logits.shape)}"
            )
        neg = torch.stack([logits[:, i] for i in self.negative_index], dim=-1)
        z_pos = logits[:, self.positive_index].unsqueeze(-1)
        z_neg = torch.logsumexp(neg, dim=-1, keepdim=True)
        binary_logits = torch.cat([z_neg, z_pos], dim=-1)  # index 1 == positive class

        target = (labels == self.positive_index).long()
        weight = None
        if self.class_weight is not None:
            weight = self.class_weight.to(logits.device)

        loss = F.cross_entropy(binary_logits, target, weight=weight, reduction="none")
        if self.gamma > 0:
            with torch.no_grad():
                pt = torch.exp(-loss.detach())
            loss = loss * (1.0 - pt).pow(self.gamma)

        return loss.mean() if self.reduction == "mean" else loss.sum()

    def extra_repr(self) -> str:
        return (
            f"focus_label={self.focus_label}, positive_weight={self.positive_weight}, "
            f"gamma={self.gamma}"
        )

    def describe(self) -> str:
        positives = self.focus_label
        negatives = ", ".join(ID2LABEL[i] for i in self.negative_index)
        return (
            f"ConceptFocusedLoss(focus={positives}; negative={{{negatives}}}; "
            f"positive_weight={self.positive_weight})"
        )


def positive_class_weight(
    labels: Sequence[int] | np.ndarray,
    focus_label: str,
    num_labels: int = NUM_LABELS,
    normalize: bool = True,
    cap: Optional[float] = None,
) -> float:
    """Neg/pos ratio inside the concept, estimated on the **train split only**.

    ``cap`` guards against an extreme ratio on very small debug subsets.
    """
    focus_id = LABEL2ID[focus_label]
    arr = np.asarray(
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    ).reshape(-1).astype(int)
    n_pos = float((arr == focus_id).sum())
    n_neg = float((arr != focus_id).sum())
    if n_pos <= 0:
        raise ValueError(f"No positive samples for focus label {focus_label!r} in the given split")
    weight = n_neg / n_pos
    if normalize:
        # [neg=1, pos=w] -> mean 1 so the loss scale stays comparable to stage 3
        weight = 2.0 * weight / (1.0 + weight)
    if cap is not None:
        weight = min(weight, cap)
    return round(float(weight), 6)


def concept_stage_plan() -> Dict[str, Dict[str, object]]:
    """The three curriculum stages as described in plan section 13."""
    return {
        "stage_1": {
            "name": "OFFENSIVE-focused",
            "focus_label": "OFFENSIVE",
            "concept": {"OFFENSIVE": "positive", "HATE": "negative", "CLEAN": "negative"},
            "uses_full_dataset": True,
        },
        "stage_2": {
            "name": "HATE-focused",
            "focus_label": "HATE",
            "concept": {"HATE": "positive", "OFFENSIVE": "negative", "CLEAN": "negative"},
            "uses_full_dataset": True,
        },
        "stage_3": {
            "name": "final-3-class",
            "focus_label": None,
            "concept": {"HATE": "positive", "OFFENSIVE": "positive", "CLEAN": "positive"},
            "uses_full_dataset": True,
        },
    }


__all__ = ["ConceptFocusedLoss", "concept_stage_plan", "positive_class_weight"]
