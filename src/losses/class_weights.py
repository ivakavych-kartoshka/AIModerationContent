"""Class weights computed **from the training split only** (plan sections 9 & 10).

Rules enforced here:

* weights are estimated from ``train`` only - never from validation or test;
* every weight that ends up in a run is written to ``class_weights.json`` and to
  the run config (plan section 10);
* the weight vector is normalised to mean 1.0 so that switching class weighting
  on/off does not silently rescale the loss and therefore does not change the
  effective learning rate - this keeps the four compared systems fair.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from ..common.constants import ID2LABEL, LABEL2ID, NUM_LABELS

STRATEGIES = ("none", "inverse_frequency", "inverse_sqrt_frequency", "effective_number", "manual")


def label_counts(labels: Sequence[int] | np.ndarray | torch.Tensor, num_labels: int = NUM_LABELS) -> np.ndarray:
    """Absolute sample count per class id, ordered 0..num_labels-1."""
    if isinstance(labels, torch.Tensor):
        arr = labels.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(labels).reshape(-1)
    return np.bincount(arr.astype(int), minlength=num_labels)[:num_labels]


def _weights_from_counts(
    counts: np.ndarray,
    strategy: str,
    beta: float = 0.999,
) -> np.ndarray:
    counts = counts.astype(float)
    safe = np.where(counts > 0, counts, 1.0)
    if strategy == "none":
        weights = np.ones_like(safe)
    elif strategy == "inverse_frequency":
        weights = 1.0 / safe
    elif strategy == "inverse_sqrt_frequency":
        weights = 1.0 / np.sqrt(safe)
    elif strategy == "effective_number":
        # Cui et al. 2019 class-balanced loss (used with focal loss in RetinaNet)
        weights = (1.0 - beta) / (1.0 - np.power(beta, safe))
        weights = np.where(counts > 0, weights, 0.0)
        weights = np.where(weights <= 0, np.zeros_like(weights), weights)
        if weights.sum() <= 0:
            weights = np.ones_like(safe)
    else:
        raise ValueError(f"Unknown class-weight strategy {strategy!r}; expected one of {STRATEGIES}")
    return weights


def _normalize_to_unit_mean(weights: np.ndarray) -> np.ndarray:
    mean = float(weights.mean())
    if mean <= 0:
        raise ValueError(f"Degenerate class weights (mean={mean}); cannot normalise.")
    return weights / mean


def compute_class_weights(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    strategy: str = "inverse_frequency",
    beta: float = 0.999,
    manual: Optional[Mapping[str, float]] = None,
    normalize: bool = True,
    num_labels: int = NUM_LABELS,
) -> Dict[str, float]:
    """Return ``{"HATE": w, "OFFENSIVE": w, "CLEAN": w}``.

    ``manual`` overrides the computed vector; keys are label *names*.
    """
    counts = label_counts(labels, num_labels)

    if strategy == "manual":
        if not manual:
            raise ValueError("class_weight strategy 'manual' requires loss.class_weight_manual")
        weights = np.array(
            [float(manual.get(ID2LABEL[i], 1.0)) for i in range(num_labels)], dtype=float
        )
    else:
        weights = _weights_from_counts(counts, strategy, beta)
        if manual:
            # partial override: e.g. tune CLEAN only, keep the computed ratio for the rest
            weights = weights.copy()
            for i in range(num_labels):
                name = ID2LABEL[i]
                if name in manual:
                    weights[i] = float(manual[name])

    if normalize:
        weights = _normalize_to_unit_mean(weights)

    return {ID2LABEL[i]: round(float(weights[i]), 6) for i in range(num_labels)}


def weights_to_tensor(
    weights: Mapping[str, float],
    num_labels: int = NUM_LABELS,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``{"HATE": w, ...}`` -> tensor ordered by canonical label id."""
    return torch.tensor(
        [float(weights[ID2LABEL[i]]) for i in range(num_labels)],
        dtype=dtype,
        device=device,
    )


def weights_to_dict_array(weights: Mapping[str, float], num_labels: int = NUM_LABELS) -> np.ndarray:
    return np.array([float(weights[ID2LABEL[i]]) for i in range(num_labels)], dtype=float)


def describe(counts: Mapping[str, int], weights: Mapping[str, float]) -> str:
    lines = ["Class weights (estimated on TRAIN only)", "-" * 46]
    for label in LABEL2ID:
        lines.append(f"{label:<10} n={counts.get(label, 0):>7}  weight={weights.get(label, 1.0):.4f}")
    return "\n".join(lines)


__all__ = [
    "STRATEGIES",
    "compute_class_weights",
    "describe",
    "label_counts",
    "weights_to_dict_array",
    "weights_to_tensor",
]
