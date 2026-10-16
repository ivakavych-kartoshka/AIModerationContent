"""The training loop.

A single explicit PyTorch loop is used for all four systems so that the only
differences between them are the ones declared in the configuration
(plan section 1: same dataset split, same label mapping, same test set).

Features
--------
* gradient accumulation, mixed precision (``fp16`` / ``bf16`` / ``fp32``),
  gradient clipping, gradient checkpointing;
* discriminative learning rates through param groups (plan section 8.2);
* multi-stage training driven by :class:`StageController`
  (2-stage DLR run and 3-stage curriculum run);
* per-stage loss rebuild - cross-entropy / focal / curriculum concept loss,
  with class weights estimated on the training split only;
* callbacks: best-checkpoint saving, early stopping, ``training_log.csv``;
* every decision (best model, early stop, thresholds) is taken on validation;
* early stopping on a non-final stage only ends that stage early - the run
  continues with the next stage, initialised from the *best* epoch of the
  stopped stage rather than the last one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..callbacks.base import CallbackList
from ..callbacks.monitoring import (
    BestCheckpointCallback,
    EarlyStoppingCallback,
    is_better,
    resolve_direction,
)
from ..callbacks.stage_controller import StageController
from ..common.constants import LABELS
from ..common.logging_utils import get_logger
from ..data.dataset import DataBundle, make_dataloader
from ..evaluation.metrics import evaluate_model
from ..losses.builder import build_loss
from .optim import (
    build_optimizer,
    build_param_groups,
    build_scheduler,
    representative_lrs,
)
from .stages import StageSpec

LOGGER = get_logger(__name__)

PRECISION_TO_DTYPE = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class TrainResult:
    """Everything the report generator needs to know about a finished run."""

    model_name: str
    best_metric: Optional[str] = None
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    best_stage: Optional[str] = None
    epochs_run: int = 0
    total_epochs_planned: int = 0
    training_seconds: float = 0.0
    stopped_early: bool = False
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    best_checkpoint: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, Any] = field(default_factory=dict)
    lr_by_group: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "best_metric": self.best_metric,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_stage": self.best_stage,
            "epochs_run": self.epochs_run,
            "total_epochs_planned": self.total_epochs_planned,
            "training_seconds": round(self.training_seconds, 3),
            "training_time_hms": _hms(self.training_seconds),
            "stopped_early": self.stopped_early,
            "early_stopping": self.early_stopping,
            "best_checkpoint": self.best_checkpoint,
            "stages": self.stages,
            "lr_by_group": self.lr_by_group,
        }


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Trainer:
    """Fine-tune one model according to a resolved configuration."""

    def __init__(
        self,
        model: nn.Module,
        bundle: DataBundle,
        stages: Sequence[StageSpec],
        config: Dict[str, Any],
        output_dir: str | Path,
        experiment_dir: str | Path | None = None,
        model_name: str = "model",
        device: Optional[torch.device] = None,
        callbacks: Optional[CallbackList] = None,
    ) -> None:
        self.model = model
        self.bundle = bundle
        self.stages = list(stages)
        self.config = config
        self.output_dir = Path(output_dir)
        self.experiment_dir = Path(experiment_dir) if experiment_dir else self.output_dir
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = int(config.get("seed", 42))
        self.train_labels: List[int] = [int(v) for v in bundle.train.labels.tolist()]

        precision = str(config.get("training", {}).get("precision", "bf16")).lower()
        self.amp_dtype = PRECISION_TO_DTYPE.get(precision, None)
        if self.amp_dtype is torch.float16 and self.device.type == "cpu":
            LOGGER.warning("fp16 on CPU is not supported - falling back to fp32.")
            self.amp_dtype = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype is torch.float16)

        self.callbacks = callbacks or CallbackList()
        self.controller = StageController(
            self.stages,
            checkpoints_root=self.experiment_dir / "stage_checkpoints",
            save_transitions_to=self.output_dir / "stage_transitions.json",
        )
        self._resolve_stage_checkpoints()

        # runtime state used by the CSV logger
        self.current_stage: Optional[StageSpec] = None
        self.current_stage_index = 1
        self.epoch_in_stage = 0
        self.global_epoch = 0
        self.current_lr = 0.0
        self.current_encoder_lr: Optional[float] = None
        self.current_head_lr: Optional[float] = None
        self.last_grad_norm: Optional[float] = None
        self.last_epoch_seconds: Optional[float] = None
        self.elapsed_seconds = 0.0
        self.history: List[Dict[str, Any]] = []
        self.lr_by_group: Dict[str, Any] = {}

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.criterion: Optional[nn.Module] = None

        es_cfg = config.get("early_stopping", {}) or {}
        self.monitor_metric = str(es_cfg.get("metric", "macro_f1"))
        self.monitor_mode = str(es_cfg.get("mode") or resolve_direction(self.monitor_metric))

        self._prepare_model()

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _resolve_stage_checkpoints(self) -> None:
        """Turn ``init_from`` placeholders into real paths.

        Supported forms:

        ``stage:<name>``      the ``<name>`` checkpoint written by an earlier stage
        ``auto:previous``     the checkpoint of the previous stage
        anything else         a literal path (relative paths resolve to the project root)
        """
        from ..common.paths import project_path

        checkpoint_root = self.experiment_dir / "stage_checkpoints"
        for index, stage in enumerate(self.stages):
            if not stage.init_from:
                continue
            token = str(stage.init_from)
            if token in {"auto:previous", "${auto:previous}"}:
                if index == 0:
                    raise ValueError("auto:previous cannot be used by the first stage")
                stage.init_from = str(checkpoint_root / self.stages[index - 1].name)
            elif token.startswith("stage:") or token.startswith("${stage:"):
                name = token.split(":", 1)[1].rstrip("}")
                stage.init_from = str(checkpoint_root / name)
            else:
                stage.init_from = str(project_path(token))

    def _prepare_model(self) -> None:
        self.model.to(self.device)
        if self.config.get("training", {}).get("gradient_checkpointing", False):
            target = getattr(self.model, "encoder_module", self.model)
            enable = getattr(target, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()
                LOGGER.info("[train] gradient checkpointing enabled")
            else:  # pragma: no cover
                LOGGER.warning("[train] gradient checkpointing requested but not available")
        use_cache = getattr(self.model, "config", None)
        if use_cache is not None and hasattr(use_cache, "use_cache"):
            use_cache.use_cache = False

    def _dataloader(self, split: str, batch_size: int, shuffle: bool, training: bool) -> DataLoader:
        dataset = getattr(self.bundle, split)
        return make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=self.seed,
            num_workers=int(self.config.get("training", {}).get("num_workers", 2)) if training else 0,
            drop_last=bool(self.config.get("training", {}).get("dataloader_drop_last", False)) if training else False,
        )

    # ------------------------------------------------------------------ #
    def build_stage_runtime(self, stage: StageSpec) -> Dict[str, Any]:
        """Optimizer + scheduler + loss for one stage."""
        tcfg = stage.training or self.config.get("training", {})
        dcfg = stage.dlr or self.config.get("dlr", {})
        discriminative = bool(dcfg.get("enabled", False))

        train_loader = self._dataloader(
            "train",
            batch_size=int(tcfg.get("batch_size", 16)),
            shuffle=True,
            training=True,
        )
        steps_per_epoch = max(1, math.ceil(len(train_loader) / max(int(tcfg.get("gradient_accumulation", 1)), 1)))
        total_steps = steps_per_epoch * stage.num_epochs

        param_groups = build_param_groups(
            self.model,
            learning_rate=float(tcfg.get("learning_rate", 3e-5)),
            head_lr=float(dcfg.get("head_lr", tcfg.get("learning_rate", 3e-5))) if discriminative else float(tcfg.get("learning_rate", 3e-5)),
            encoder_lr=float(dcfg.get("encoder_lr", tcfg.get("learning_rate", 3e-5))) if discriminative else None,
            layer_lr=float(dcfg.get("layer_lr", tcfg.get("learning_rate", 3e-5))) if discriminative else None,
            embedding_lr=dcfg.get("embedding_lr") if discriminative else None,
            layer_split=float(dcfg.get("layer_split", 0.5)),
            layer_decay=float(dcfg.get("layer_decay", 1.0)),
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
            discriminative=discriminative,
        )
        optimizer = build_optimizer(param_groups, name=str(tcfg.get("optimizer", "adamw_torch")))
        scheduler = build_scheduler(
            optimizer,
            scheduler=str(tcfg.get("scheduler", "linear")),
            num_training_steps=total_steps,
            warmup_ratio=float(tcfg.get("warmup_ratio", 0.1)),
        )

        loss_cfg = dict(stage.loss or self.config.get("loss", {}))
        criterion = build_loss(
            loss_cfg,
            num_labels=int(self.config.get("dataset", {}).get("num_labels", 3)),
            train_labels=self.train_labels,
            device=self.device,
            description=stage.describe(),
        )

        self.lr_by_group[stage.name] = {
            "discriminative": discriminative,
            "encoder_lr": dcfg.get("encoder_lr") if discriminative else None,
            "layer_lr": dcfg.get("layer_lr") if discriminative else None,
            "head_lr": dcfg.get("head_lr") if discriminative else None,
            "learning_rate": float(tcfg.get("learning_rate", 3e-5)),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        }

        return {
            "train_loader": train_loader,
            "val_loader": self._dataloader(
                "validation",
                batch_size=int(tcfg.get("eval_batch_size", 32)),
                shuffle=False,
                training=False,
            ),
            "optimizer": optimizer,
            "scheduler": scheduler,
            "criterion": criterion,
            "training": tcfg,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        }

    # ------------------------------------------------------------------ #
    def train_epoch(self, runtime: Dict[str, Any]) -> Dict[str, float]:
        """One pass over the training split."""
        model = self.model
        loader = runtime["train_loader"]
        optimizer = runtime["optimizer"]
        scheduler = runtime["scheduler"]
        criterion = runtime["criterion"]
        tcfg = runtime["training"]

        accum = max(int(tcfg.get("gradient_accumulation", 1)), 1)
        clip = float(tcfg.get("max_grad_norm", 1.0) or 0.0)
        log_every = int(tcfg.get("log_every_n_steps", 50))

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        step = 0

        start = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype or torch.float32,
                enabled=self.amp_dtype is not None,
            ):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits.float(), labels)
                loss = loss / accum

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            batch_size = labels.size(0)
            total_loss += float(loss.detach()) * accum * batch_size
            total_correct += int((logits.detach().argmax(dim=-1) == labels).sum())
            total_seen += batch_size

            is_last = batch_index == len(loader) - 1
            if (batch_index + 1) % accum == 0 or is_last:
                if clip > 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(optimizer)
                    self.last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    )
                else:
                    self.last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    )
                if self.scaler.is_enabled():
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                self._track_lrs(optimizer)
                if log_every and step % log_every == 0:
                    LOGGER.info(
                        "[train] step %d/%d  loss=%.4f  lr=%.2e",
                        step, int(runtime["total_steps"]), total_loss / max(total_seen, 1), self.current_lr,
                    )
                self.callbacks.on_step_end(trainer=self, step=step)

        seconds = time.perf_counter() - start
        self.last_epoch_seconds = seconds
        return {
            "loss": total_loss / max(total_seen, 1),
            "accuracy": total_correct / max(total_seen, 1),
            "optimizer_steps": step,
            "seconds": seconds,
        }

    def _track_lrs(self, optimizer: torch.optim.Optimizer) -> None:
        lrs = representative_lrs(optimizer)
        self.current_lr = lrs.get("learning_rate", 0.0)
        self.current_encoder_lr = lrs.get("encoder_lr")
        self.current_head_lr = lrs.get("head_lr")

    # ------------------------------------------------------------------ #
    def evaluate(self, runtime: Dict[str, Any]) -> Dict[str, Any]:
        metrics = evaluate_model(
            self.model,
            runtime["val_loader"],
            self.device,
            criterion=runtime["criterion"],
            decision_rule="argmax",
            amp_dtype=self.amp_dtype,
        )
        epoch_metrics: Dict[str, Any] = {
            "val_loss": metrics.get("loss"),
            "val_accuracy": metrics.get("accuracy"),
            "val_macro_f1": metrics.get("macro_f1"),
            "val_weighted_f1": metrics.get("weighted_f1"),
            "val_macro_precision": metrics.get("macro_precision"),
            "val_macro_recall": metrics.get("macro_recall"),
            "val_macro_roc_auc": metrics.get("macro_roc_auc"),
        }
        for name in LABELS:
            for metric in ("precision", "recall", "f1", "support"):
                epoch_metrics[f"val_{name}_{metric}"] = metrics.get(f"{name}_{metric}")
        epoch_metrics["val_confusion_matrix"] = metrics.get("confusion_matrix")
        return epoch_metrics

    # ------------------------------------------------------------------ #
    def train(self) -> TrainResult:
        t0 = time.perf_counter()
        early_stopping: Optional[EarlyStoppingCallback] = None
        best_ckpt: Optional[BestCheckpointCallback] = None
        for cb in self.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                early_stopping = cb
            elif isinstance(cb, BestCheckpointCallback):
                best_ckpt = cb

        self.callbacks.on_train_begin(trainer=self)

        while True:
            stage = self.controller.begin_stage()
            self.current_stage = stage
            self.current_stage_index = self.controller.index + 1
            is_last_stage = self.controller.is_last_stage

            if stage.init_from:
                self.load_model(stage.init_from)
                LOGGER.info("[stage] initialised weights from %s", stage.init_from)

            runtime = self.build_stage_runtime(stage)
            self.optimizer = runtime["optimizer"]
            self.scheduler = runtime["scheduler"]
            self.criterion = runtime["criterion"]
            self._track_lrs(self.optimizer)

            self.callbacks.on_stage_begin(trainer=self, stage=stage)

            # The stage checkpoint handed to the *next* stage is the best epoch
            # of this stage (not the last one) so the next stage always starts
            # from the strongest base.  Weights are cloned to CPU memory so GPU
            # vRAM is not doubled while training continues.
            stage_best_value: Optional[float] = None
            stage_best_epoch: Optional[int] = None
            stage_best_state: Optional[Dict[str, Any]] = None

            while not self.controller.is_stage_finished():
                self.global_epoch = self.controller.begin_epoch()
                self.epoch_in_stage = self.controller.epoch_in_stage
                self.callbacks.on_epoch_begin(trainer=self, epoch=self.global_epoch, stage=stage)

                train_stats = self.train_epoch(runtime)
                self.elapsed_seconds = time.perf_counter() - t0
                val_metrics = self.evaluate(runtime)

                metrics: Dict[str, Any] = {
                    "loss": train_stats["loss"],
                    "accuracy": train_stats["accuracy"],
                    **val_metrics,
                }
                metrics["is_best"] = False

                LOGGER.info(
                    "[epoch %2d/%2d | %s] train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f (%.1fs)",
                    self.epoch_in_stage, stage.num_epochs, stage.name,
                    train_stats["loss"], train_stats["accuracy"],
                    float(val_metrics.get("val_loss") or float("nan")),
                    float(val_metrics.get("val_accuracy") or float("nan")),
                    float(val_metrics.get("val_macro_f1") or float("nan")),
                    train_stats["seconds"],
                )

                self.callbacks.on_epoch_end(trainer=self, epoch=self.global_epoch, metrics=metrics, stage=stage)

                # recorded after the callbacks so `is_best` reflects the
                # BestCheckpointCallback decision rather than the default
                self.history.append(
                    {
                        "epoch": self.global_epoch,
                        "stage": stage.name,
                        **{k: v for k, v in metrics.items() if k != "val_confusion_matrix"},
                    }
                )

                value = metrics.get(self.monitor_metric, metrics.get(f"val_{self.monitor_metric}"))
                if value is not None and value == value:  # ignore NaN
                    value = float(value)
                    if is_better(value, stage_best_value, self.monitor_mode):
                        stage_best_value = value
                        stage_best_epoch = self.global_epoch
                        stage_best_state = {
                            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                        }

                if early_stopping is not None and early_stopping.should_stop:
                    if is_last_stage:
                        LOGGER.info("[train] early stopping triggered after epoch %d (last stage)", self.global_epoch)
                    else:
                        LOGGER.info(
                            "[train] early stopping ended stage %s at epoch %d - "
                            "best epoch kept as the base for the next stage",
                            stage.name, self.global_epoch,
                        )
                    break

            if stage_best_state is not None and not is_last_stage:
                LOGGER.info(
                    "[stage] restoring best %s=%.6f at epoch %d for the %s checkpoint",
                    self.monitor_metric, stage_best_value, stage_best_epoch, stage.name,
                )
                self.model.load_state_dict(stage_best_state)

            self.controller.end_stage(model=self.model)
            self.callbacks.on_stage_end(trainer=self, stage=stage)

            # early stopping only ends the whole run when it fires on the last
            # stage; an earlier stage simply finishes early and training continues
            if early_stopping is not None and early_stopping.should_stop and is_last_stage:
                break
            if not self.controller.next_stage_exists():
                break

        self.elapsed_seconds = time.perf_counter() - t0
        self.controller.end_train(reason="early_stopping" if (early_stopping and early_stopping.should_stop) else "completed")
        self.callbacks.on_train_end(trainer=self)

        if bool(self.config.get("output", {}).get("save_last_checkpoint", True)):
            self.save_model(self.output_dir / "last_model")

        # early stopping can be disabled; then the best-checkpoint callback is the
        # only component that knows which epoch actually won.
        primary = best_ckpt
        if early_stopping is not None and early_stopping.best_value is not None:
            primary = early_stopping

        return TrainResult(
            model_name=self.model_name,
            best_metric=(primary.metric if primary is not None else None),
            best_value=(primary.best_value if primary is not None else None),
            best_epoch=(primary.best_epoch if primary is not None else None),
            best_stage=(best_ckpt.best_stage if best_ckpt else None),
            epochs_run=self.controller.global_epoch,
            total_epochs_planned=self.controller.total_epochs,
            training_seconds=self.elapsed_seconds,
            stopped_early=bool(early_stopping.should_stop) if early_stopping else False,
            early_stopping=early_stopping.summary() if early_stopping else {},
            best_checkpoint=best_ckpt.summary() if best_ckpt else {},
            stages=self.controller.summary(),
            lr_by_group=self.lr_by_group,
        )

    # ------------------------------------------------------------------ #
    def save_model(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(out)
        return out

    def load_model(self, path: str | Path) -> None:
        """Restore weights from a directory produced by ``save_pretrained``."""
        from ..models.model import CustomHeadModel

        src = Path(path)
        if not src.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {src}")
        loaded = CustomHeadModel.from_pretrained(src)
        self.model.load_state_dict(loaded.state_dict())
        LOGGER.info("[train] weights loaded from %s", src)

    def save_training_state(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "epoch": self.global_epoch,
            "stage_index": self.current_stage_index,
            "stage_name": getattr(self.current_stage, "name", None),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "history": self.history,
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(payload, out)
        return out


__all__ = ["TrainResult", "Trainer"]
