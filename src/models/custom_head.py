"""Custom classification head used by ``sensitiveai-vi-custom`` and its descendants.

Why a custom head?
------------------
The standard Hugging Face head for DeBERTa-v2 / mDeBERTa-v3 is

    last_hidden_state[:, 0]  ->  ContextPooler(Linear 768->768 + tanh)
                             ->  Dropout
                             ->  Linear 768 -> num_labels

i.e. the *decision is made from a single token* (the first one).  On short,
heavily code-switched social-media comments the first token carries most of the
sentence function word, and the pooled representation is a single 768-d vector
with no explicit normalisation between the representation and the classifier.

The custom head implemented here keeps the encoder untouched and only replaces
everything after it:

    last_hidden_state (B, L, 768)  +  attention_mask (B, L)
              |
              +--> z_cls  = H[:, 0]                          (768)  sentence view
              +--> z_mean = masked mean over L              (768)  content view
              +--> z_max  = masked max over L               (768)  salience view
              (+ optional) z_att = attention pooling        (768)  learned view
              |
              v
    z = LayerNorm( concat[z_cls, z_mean, z_max(, z_att)] )  (2304 or 3072)
              |
              v
    z = Linear(-> hidden) -> GELU -> Dropout -> LayerNorm     (hidden = 512)
              |
              v
    logits = Dropout -> Linear(hidden -> num_labels)          (+ optional class prior bias)

Design rationale, one line per component:

``z_cls``   captures the global / syntactic role of the utterance;
``z_mean``  averages the topical content, robust for short comments where a
            single toxic token decides the label;
``z_max``   keeps the strongest local signal - exactly what a hate-speech
            trigger word should dominate;
``LayerNorm`` before the fusion MLP removes the scale mismatch between the
            three pooling views (their magnitudes differ by an order of
            magnitude after a pre-trained encoder);
``hidden``  bottleneck (512) before the output layer adds a non-linear,
            regularised decision boundary instead of a single linear hyperplane.

Nothing here touches the dataset, the split or the label mapping, so
comparing ``sensitiveai-vi`` with ``sensitiveai-vi-custom`` isolates the effect
of the head (plan section 7).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid positions only."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def masked_max_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Max over valid positions only (invalid positions pushed to ``-inf``)."""
    mask = attention_mask.unsqueeze(-1)
    very_negative = torch.finfo(hidden.dtype).min
    return hidden.masked_fill(~mask.bool(), very_negative).max(dim=1).values


class AttentionPool(nn.Module):
    """Single learned query attention pooling (optional 4th view)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bld,d->bl", self.proj(hidden), self.query) / (hidden.size(-1) ** 0.5)
        scores = scores.masked_fill(~attention_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)


class CustomClassificationHead(nn.Module):
    """Multi-view pooling + bottleneck MLP classifier.

    Parameters
    ----------
    hidden_size:
        Encoder hidden size (768 for mDeBERTa-v3-base).
    num_labels:
        Number of classes (3: HATE / OFFENSIVE / CLEAN).
    head_hidden_size:
        Width of the fusion bottleneck (512).
    dropout:
        Dropout applied inside the fusion MLP and before the output layer.
    use_mean_pool / use_max_pool / use_cls_pool:
        Toggle the individual views.  At least one must stay enabled.
    use_attention_pool:
        Add the learned attention-pooled view.
    use_class_prior:
        Add a learnable per-class bias initialised to the (train-set only)
        log-prior.  Helps the rare HATE class in the first epochs.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 3,
        head_hidden_size: int = 512,
        dropout: float = 0.1,
        use_cls_pool: bool = True,
        use_mean_pool: bool = True,
        use_max_pool: bool = True,
        use_attention_pool: bool = False,
        use_class_prior: bool = False,
        class_prior: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not (use_cls_pool or use_mean_pool or use_max_pool or use_attention_pool):
            raise ValueError("CustomClassificationHead needs at least one pooling view enabled.")

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.head_hidden_size = head_hidden_size
        self.use_cls_pool = use_cls_pool
        self.use_mean_pool = use_mean_pool
        self.use_max_pool = use_max_pool
        self.use_attention_pool = use_attention_pool
        self.use_class_prior = use_class_prior

        self.views: Dict[str, int] = {}
        fused_size = 0
        if use_cls_pool:
            self.views["cls"] = hidden_size
            fused_size += hidden_size
        if use_mean_pool:
            self.views["mean"] = hidden_size
            fused_size += hidden_size
        if use_max_pool:
            self.views["max"] = hidden_size
            fused_size += hidden_size
        if use_attention_pool:
            self.attention_pool = AttentionPool(hidden_size)
            self.views["attention"] = hidden_size
            fused_size += hidden_size
        self.fused_size = fused_size

        # --- fusion block -------------------------------------------------
        self.fusion_norm = nn.LayerNorm(fused_size)
        self.fusion_linear = nn.Linear(fused_size, head_hidden_size)
        self.fusion_activation = nn.GELU()
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_out_norm = nn.LayerNorm(head_hidden_size)

        # --- classifier ----------------------------------------------------
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_hidden_size, num_labels, bias=not use_class_prior)
        self.dropout = dropout

        if use_class_prior:
            prior = (
                class_prior.detach().clone().float()
                if class_prior is not None
                else torch.zeros(num_labels)
            )
            self.classifier.bias = nn.Parameter(prior)

    # ------------------------------------------------------------------ #
    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Build the fused representation from the enabled views."""
        parts = []
        if self.use_cls_pool:
            parts.append(hidden[:, 0])
        if self.use_mean_pool:
            parts.append(masked_mean_pool(hidden, attention_mask))
        if self.use_max_pool:
            parts.append(masked_max_pool(hidden, attention_mask))
        if self.use_attention_pool:
            parts.append(self.attention_pool(hidden, attention_mask))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long)

        z = self.pool(hidden_states, attention_mask)
        z = self.fusion_norm(z)
        z = self.fusion_linear(z)
        z = self.fusion_activation(z)
        z = self.fusion_dropout(z)
        z = self.fusion_out_norm(z)
        z = self.classifier_dropout(z)
        return self.classifier(z)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        views = " + ".join(f"{k}({v})" for k, v in self.views.items())
        return (
            "CustomClassificationHead\n"
            f"  pooling views      : {views}  -> fused {self.fused_size}\n"
            f"  fusion_norm        : LayerNorm({self.fused_size})\n"
            f"  fusion_mlp         : Linear({self.fused_size} -> {self.head_hidden_size}) + GELU"
            f" + Dropout({self.dropout}) + LayerNorm({self.head_hidden_size})\n"
            f"  classifier         : Dropout({self.dropout}) + Linear({self.head_hidden_size} -> {self.num_labels})\n"
            f"  class_prior bias   : {self.use_class_prior}"
        )

    def config_dict(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "num_labels": self.num_labels,
            "head_hidden_size": self.head_hidden_size,
            "dropout": self.dropout,
            "use_cls_pool": self.use_cls_pool,
            "use_mean_pool": self.use_mean_pool,
            "use_max_pool": self.use_max_pool,
            "use_attention_pool": self.use_attention_pool,
            "use_class_prior": self.use_class_prior,
        }


def build_custom_head(
    hidden_size: int = 768,
    num_labels: int = 3,
    options: Optional[Dict[str, object]] = None,
    class_prior: Optional[torch.Tensor] = None,
) -> CustomClassificationHead:
    """Instantiate the head from a plain config dict (YAML friendly)."""
    options = dict(options or {})
    allowed = {
        "head_hidden_size",
        "dropout",
        "use_cls_pool",
        "use_mean_pool",
        "use_max_pool",
        "use_attention_pool",
        "use_class_prior",
    }
    kwargs = {k: v for k, v in options.items() if k in allowed}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown custom-head options: {sorted(unknown)}")
    return CustomClassificationHead(
        hidden_size=hidden_size,
        num_labels=num_labels,
        class_prior=class_prior,
        **kwargs,
    )


__all__ = [
    "AttentionPool",
    "CustomClassificationHead",
    "build_custom_head",
    "masked_max_pool",
    "masked_mean_pool",
]
