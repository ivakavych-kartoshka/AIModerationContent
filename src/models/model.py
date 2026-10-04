"""The two model variants compared in the research plan.

``sensitiveai-vi``
    ``AutoModelForSequenceClassification`` on ``microsoft/mdeberta-v3-base`` -
    i.e. the *standard* Hugging Face classification head
    (ContextPooler(Linear+tanh) on the first token -> Dropout -> Linear).
    This is the baseline.

``sensitiveai-vi-custom`` and its descendants
    the same pre-trained encoder plus the :class:`CustomClassificationHead`
    defined in :mod:`src.models.custom_head`.

Both wrappers expose the same interface so the training loop, the losses, the
callbacks and the evaluation code are literally shared. Only the head differs,
which is what makes the comparison in the paper controlled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from ..common.constants import NUM_LABELS
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

CUSTOM_CONFIG_NAME = "sensitiveai_config.json"
CUSTOM_HEAD_WEIGHTS = "head.safetensors"


def load_encoder(loader, name_or_path: str, **kwargs):
    """Load a transformer checkpoint with **fp32 master weights**.

    ``microsoft/mdeberta-v3-base`` is published in float16.  Recent versions of
    ``transformers`` keep the checkpoint dtype when instantiating the model,
    which would train with half-precision weights and no optimiser state in
    full precision - unstable, and it silently changes the memory budget.  The
    dtype is therefore pinned to float32 here and mixed precision is handled by
    ``torch.autocast`` during training instead.
    """
    kwargs.setdefault("cache_dir", None)
    try:
        model = loader.from_pretrained(name_or_path, dtype=torch.float32, **kwargs)
    except TypeError:  # transformers < 5 uses `torch_dtype`
        kwargs.pop("cache_dir", None)
        model = loader.from_pretrained(name_or_path, torch_dtype=torch.float32, **kwargs)
    return model.float()


class BaseSensitiveAIModel(nn.Module):
    """Common interface for both heads."""

    head_type: str = "base"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> SequenceClassifierOutput:  # pragma: no cover - interface
        raise NotImplementedError

    # -------------------------------------------------------------- #
    @property
    def encoder_module(self) -> nn.Module:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def head_module(self) -> nn.Module:  # pragma: no cover - interface
        raise NotImplementedError

    def parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "non_trainable_parameters": int(total - trainable),
            "estimated_parameter_memory_mb": round(total * 4 / 1024**2, 2),  # fp32 weights
        }

    # -------------------------------------------------------------- #
    def save_pretrained(self, path: str | Path) -> Path:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs: Any):  # pragma: no cover - interface
        raise NotImplementedError


class StandardHeadModel(BaseSensitiveAIModel):
    """Baseline: pre-trained encoder + Hugging Face standard classification head."""

    head_type = "standard"

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_base_model(
        cls,
        base_model: str,
        num_labels: int = 3,
        dropout: Optional[float] = None,
        cache_dir: Optional[str] = None,
    ) -> "StandardHeadModel":
        kwargs: Dict[str, Any] = {"num_labels": num_labels, "cache_dir": cache_dir}
        if dropout is not None:
            kwargs["classifier_dropout"] = dropout
        model = load_encoder(AutoModelForSequenceClassification, base_model, **kwargs)
        LOGGER.info("[model] standard head loaded from %s (num_labels=%d)", base_model, num_labels)
        return cls(model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @property
    def encoder_module(self) -> nn.Module:
        return getattr(self.model, self.model.base_model_prefix)

    @property
    def head_module(self) -> nn.Module:
        return self.model.classifier

    def save_pretrained(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        return out

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs: Any) -> "StandardHeadModel":
        model = load_encoder(AutoModelForSequenceClassification, str(path), **kwargs)
        return cls(model)


class CustomHeadModel(BaseSensitiveAIModel):
    """Pre-trained encoder + :class:`CustomClassificationHead`."""

    head_type = "custom"

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        base_model_name: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.base_model_name = base_model_name

    @classmethod
    def from_base_model(
        cls,
        base_model: str,
        num_labels: int = 3,
        head_options: Optional[Dict[str, Any]] = None,
        class_prior: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
    ) -> "CustomHeadModel":
        from .custom_head import build_custom_head

        encoder = load_encoder(AutoModel, base_model, cache_dir=cache_dir)
        hidden_size = encoder.config.hidden_size
        head = build_custom_head(
            hidden_size=hidden_size,
            num_labels=num_labels,
            options=head_options,
            class_prior=class_prior,
        )
        LOGGER.info("[model] custom head on top of %s:\n%s", base_model, head.describe())
        return cls(encoder, head, base_model)

    @classmethod
    def from_encoder_checkpoint(
        cls,
        base_model: str,
        checkpoint: Optional[str | Path],
        num_labels: int = 3,
        head_options: Optional[Dict[str, Any]] = None,
        class_prior: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
    ) -> "CustomHeadModel":
        """Initialise the encoder from a previous run (plan section 7).

        ``checkpoint`` may point at a saved ``StandardHeadModel`` /
        ``CustomHeadModel`` directory or at a plain HF checkpoint.  It is
        recorded in the run config so the provenance is never ambiguous.
        """
        if checkpoint is None:
            return cls.from_base_model(base_model, num_labels, head_options, class_prior, cache_dir)

        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"encoder_checkpoint does not exist: {ckpt}")

        if (ckpt / CUSTOM_CONFIG_NAME).is_file():
            return cls.from_pretrained(ckpt, class_prior=class_prior)

        LOGGER.info("[model] initialising encoder from checkpoint %s", ckpt)
        encoder = load_encoder(AutoModel, str(ckpt))
        from .custom_head import build_custom_head

        head = build_custom_head(
            hidden_size=encoder.config.hidden_size,
            num_labels=num_labels,
            options=head_options,
            class_prior=class_prior,
        )
        return cls(encoder, head, str(ckpt))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = self.head(outputs.last_hidden_state, attention_mask)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.last_hidden_state)

    @property
    def encoder_module(self) -> nn.Module:
        return self.encoder

    @property
    def head_module(self) -> nn.Module:
        return self.head

    # -------------------------------------------------------------- #
    def save_pretrained(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out / "encoder")
        payload = {"base_model_name": self.base_model_name, "head": self.head.config_dict()}
        (out / CUSTOM_CONFIG_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from safetensors.torch import save_file

        state = {k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()}
        save_file(state, str(out / CUSTOM_HEAD_WEIGHTS))
        return out

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs: Any) -> "CustomHeadModel":
        from safetensors.torch import load_file

        from .custom_head import build_custom_head

        src = Path(path)
        payload = json.loads((src / CUSTOM_CONFIG_NAME).read_text(encoding="utf-8"))
        head_options = dict(payload["head"])
        num_labels = int(head_options.pop("num_labels", NUM_LABELS))
        head_options.pop("hidden_size", None)
        encoder = load_encoder(AutoModel, str(src / "encoder"))
        head = build_custom_head(
            hidden_size=encoder.config.hidden_size,
            num_labels=num_labels,
            options=head_options,
            class_prior=kwargs.get("class_prior"),
        )
        head.load_state_dict(load_file(str(src / CUSTOM_HEAD_WEIGHTS)))
        return cls(encoder, head, payload.get("base_model_name", str(src)))

    @classmethod
    def from_hub_id(cls, hub_id: str, **kwargs: Any) -> "CustomHeadModel":
        return cls.from_base_model(hub_id, **kwargs)


def count_parameters(module: nn.Module) -> Dict[str, int]:
    """Standalone parameter counter (also used for the head alone)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
    }


def build_model(
    head: str,
    base_model: str,
    num_labels: int = 3,
    head_options: Optional[Dict[str, Any]] = None,
    encoder_checkpoint: Optional[str] = None,
    class_prior: Optional[torch.Tensor] = None,
    cache_dir: Optional[str] = None,
) -> BaseSensitiveAIModel:
    """Model factory used by every entry point."""
    if head == "standard":
        if encoder_checkpoint:
            LOGGER.warning(
                "[model] encoder_checkpoint is ignored for the standard head "
                "(the baseline must start from the pure pre-trained encoder)."
            )
        return StandardHeadModel.from_base_model(base_model, num_labels, cache_dir=cache_dir)
    if head == "custom":
        if encoder_checkpoint:
            return CustomHeadModel.from_encoder_checkpoint(
                base_model,
                encoder_checkpoint,
                num_labels=num_labels,
                head_options=head_options,
                class_prior=class_prior,
                cache_dir=cache_dir,
            )
        return CustomHeadModel.from_base_model(
            base_model,
            num_labels=num_labels,
            head_options=head_options,
            class_prior=class_prior,
            cache_dir=cache_dir,
        )
    raise ValueError(f"head must be 'standard' or 'custom', got {head!r}")


__all__ = [
    "AutoConfig",
    "BaseSensitiveAIModel",
    "CustomHeadModel",
    "StandardHeadModel",
    "build_model",
    "count_parameters",
]
