"""Optimizer, discriminative learning rates and LR schedules.

Discriminative learning rates (plan section 8.2)
-----------------------------------------------
A single learning rate for the whole network is not used by
``sensitiveai-vi-customdlr2stage``.  Parameters are bucketed into groups with
their own learning rate:

    embeddings (word embeddings, embedding LayerNorm)  -> embedding_lr  (lowest)
    encoder.layer.0 ... encoder.layer.k                -> encoder_lr
    encoder.layer.k ... encoder.layer.L-1              -> layer_lr     (medium)
    encoder.rel_embeddings / encoder.LayerNorm         -> encoder_lr
    classification head                                 -> head_lr      (highest)

``layer_split`` decides where the "low" part ends (``k = round(layer_split * L)``)
and ``layer_decay`` optionally applies a geometric decay across the blocks so
that the very first blocks move even more slowly than ``encoder_lr``.

Every group also gets its own weight-decay subgroup (``bias`` and ``LayerNorm.weight``
are excluded from weight decay, following the BERT/DeBERTa convention).

The three learning rates required by the plan (``encoder_lr``, ``layer_lr``,
``head_lr``) are stored in the run config and echoed into ``training_log.csv``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

LAYER_RE = re.compile(r"(?:^|\.)layer\.(\d+)\.")
NO_DECAY_KEYWORDS = ("bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight")

OPTIMIZERS = ("adamw_torch", "adamw_torch_fused", "adamw_bnb_8bit", "sgd")
SCHEDULERS = ("linear", "cosine", "constant", "constant_with_warmup")


# --------------------------------------------------------------------------- #
# parameter grouping
# --------------------------------------------------------------------------- #
def _layer_index(name: str) -> Optional[int]:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _is_head(name: str, head_prefixes: Iterable[str]) -> bool:
    return any(name.startswith(p) or f".{p}." in name for p in head_prefixes)


def split_encoder_layer(
    encoder: nn.Module,
) -> Dict[str, List[nn.Parameter]]:
    """Split the encoder parameters into embeddings / blocks / other."""
    groups: Dict[str, List[nn.Parameter]] = {"embeddings": [], "blocks": {}, "other": []}
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("embeddings"):
            groups["embeddings"].append(param)
            continue
        idx = _layer_index(name)
        if idx is not None:
            groups["blocks"].setdefault(idx, []).append(param)
        else:
            groups["other"].append(param)
    return groups


def build_param_groups(
    model: nn.Module,
    learning_rate: float,
    head_lr: Optional[float] = None,
    encoder_lr: Optional[float] = None,
    layer_lr: Optional[float] = None,
    embedding_lr: Optional[float] = None,
    layer_split: float = 0.5,
    layer_decay: float = 1.0,
    weight_decay: float = 0.01,
    discriminative: bool = False,
    head_prefixes: Iterable[str] = ("head", "classifier", "pooler", "model.classifier", "model.pooler"),
) -> List[Dict[str, Any]]:
    """Create AdamW parameter groups, optionally with discriminative LRs."""
    head_prefixes = tuple(head_prefixes)
    encoder = model.encoder_module
    head = model.head_module

    head_params = [p for p in head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}

    groups: List[Dict[str, Any]] = []
    seen: set[int] = set()
    name_by_id = {id(p): n for n, p in model.named_parameters()}

    def _no_decay(name: str) -> bool:
        return any(k in name for k in NO_DECAY_KEYWORDS)

    def add(params: List[nn.Parameter], lr: float, group_name: str) -> None:
        """Split ``params`` into a weight-decay and a no-weight-decay subgroup."""
        if not params:
            return
        named = [(name_by_id.get(id(p), "?"), p) for p in params]
        decay = [p for n, p in named if not _no_decay(n)]
        no_decay = [p for n, p in named if _no_decay(n)]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "group_name": group_name})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "group_name": group_name})
        seen.update(id(p) for p in params)

    if discriminative:
        enc_lr = float(encoder_lr if encoder_lr is not None else learning_rate)
        top_lr = float(layer_lr if layer_lr is not None else enc_lr)
        emb_lr = float(embedding_lr if embedding_lr is not None else enc_lr)
        h_lr = float(head_lr if head_lr is not None else learning_rate)

        split = split_encoder_layer(encoder)
        n_layers = max(split["blocks"]) + 1 if split["blocks"] else 0
        boundary = int(round(float(layer_split) * n_layers)) if n_layers else 0
        boundary = max(1, min(boundary, n_layers)) if n_layers else 0

        add(split["embeddings"], emb_lr, "encoder.embeddings")
        add(split["other"], enc_lr, "encoder.misc")

        for idx in sorted(split["blocks"]):
            if idx < boundary:
                lr = enc_lr
                name = f"encoder.layer.{idx}.lower"
            else:
                lr = top_lr
                name = f"encoder.layer.{idx}.upper"
            if layer_decay and layer_decay != 1.0:
                # the earlier the block, the more its lr is shrunk
                lr = lr * (float(layer_decay) ** max(n_layers - 1 - idx, 0))
            add(split["blocks"][idx], lr, name)

        add(head_params, h_lr, "head")
        LOGGER.info(
            "[dlr] groups: embeddings=%.2e  lower(%d blocks)=%.2e  upper(%d blocks)=%.2e  head=%.2e",
            emb_lr, boundary, enc_lr, max(n_layers - boundary, 0), top_lr, h_lr,
        )
    else:
        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        add(encoder_params, float(learning_rate), "encoder")
        add(head_params, float(head_lr if head_lr is not None else learning_rate), "head")

    # safety net: anything not captured above (e.g. tied/extra buffers)
    leftovers = [p for p in model.parameters() if p.requires_grad and id(p) not in seen and id(p) not in head_ids]
    if leftovers:
        LOGGER.warning("[optim] %d parameters were not assigned to any group", len(leftovers))
        add(leftovers, float(learning_rate), "unassigned")

    return groups


# --------------------------------------------------------------------------- #
# optimizer / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(
    param_groups: List[Dict[str, Any]],
    name: str = "adamw_torch",
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    name = str(name).lower()
    if name in {"adamw_torch", "adamw_torch_fused"}:
        kwargs = {"betas": betas, "eps": eps, "weight_decay": 0.0}  # per-group wd already set
        cls = torch.optim.AdamW
        if name == "adamw_torch_fused":
            cls = getattr(torch.optim, "AdamW", cls)
            kwargs["fused"] = True
        return cls(param_groups, **kwargs)
    if name == "adamw_bnb_8bit":
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "training.optimizer='adamw_bnb_8bit' needs bitsandbytes: pip install bitsandbytes"
            ) from exc
        return bnb.optim.AdamW8bit(param_groups, betas=betas, eps=eps, weight_decay=0.0)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=float(param_groups[0]["lr"]), momentum=momentum, weight_decay=0.0)
    raise ValueError(f"Unknown optimizer {name!r}; expected one of {OPTIMIZERS}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: str = "linear",
    num_training_steps: int = 1,
    warmup_ratio: float = 0.1,
    warmup_steps: Optional[int] = None,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear / cosine / constant schedule with linear warmup.

    The warmup is applied to every param group independently, so discriminative
    learning rates keep their ratios throughout training.
    """
    scheduler = str(scheduler).lower()
    if scheduler not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler {scheduler!r}; expected one of {SCHEDULERS}")

    num_training_steps = max(int(num_training_steps), 1)
    warmup = int(warmup_steps) if warmup_steps is not None else int(round(warmup_ratio * num_training_steps))
    warmup = max(0, min(warmup, num_training_steps - 1)) if num_training_steps > 1 else 0

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step) / float(max(warmup, 1))
        progress = float(step - warmup) / float(max(num_training_steps - warmup, 1))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler == "linear":
            return max(min_lr_ratio, 1.0 - progress)
        if scheduler == "cosine":
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def current_group_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """``{"encoder": 1.2e-5, "head": 1.0e-4, ...}`` for the current step."""
    return {
        str(group.get("group_name", f"group_{i}")): float(group["lr"])
        for i, group in enumerate(optimizer.param_groups)
    }


def representative_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Collapse the groups into the ``encoder_lr`` / ``head_lr`` pair used in the log."""
    lrs = current_group_lrs(optimizer)
    encoder_lrs = [v for k, v in lrs.items() if not k.startswith("head")]
    head_lrs = [v for k, v in lrs.items() if k.startswith("head")]
    out: Dict[str, float] = {}
    if encoder_lrs:
        out["encoder_lr"] = max(encoder_lrs)
        out["encoder_lr_mean"] = sum(encoder_lrs) / len(encoder_lrs)
    if head_lrs:
        out["head_lr"] = head_lrs[0]
    out["learning_rate"] = max(lrs.values()) if lrs else 0.0
    return out


__all__ = [
    "OPTIMIZERS",
    "SCHEDULERS",
    "build_optimizer",
    "build_param_groups",
    "build_scheduler",
    "current_group_lrs",
    "representative_lrs",
    "split_encoder_layer",
]
