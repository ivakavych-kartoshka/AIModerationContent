"""YAML configuration loading / merging / dumping.

A config file fully describes one experiment. The resolved config is copied
verbatim into ``reports/<model>/run_XXX/config.yaml`` so every number reported
in the paper can be traced back to the exact run that produced it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from .constants import CONFIGS_DIR

# --------------------------------------------------------------------------- #
# Defaults. A missing key in the YAML falls back to the value below, so a
# config only has to declare what it changes.
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_name": "sensitiveai-vi",
    "run_id": "run_001",
    "base_model": "microsoft/mdeberta-v3-base",
    "head": "standard",  # standard | custom
    "seed": 42,
    "deterministic": True,

    "dataset": {
        "name": "uitnlp/vihsd",
        "splits_dir": "data/splits",
        "train_file": "data/splits/train.csv",
        "validation_file": "data/splits/validation.csv",
        "test_file": "data/splits/test.csv",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 3,
        "max_train_samples": None,  # debug only - never use for the paper
        "max_eval_samples": None,  # debug only
    },

    "tokenizer": {
        "max_length": 256,
        "padding": "max_length",
        "truncation": True,
        "lowercase": True,
    },

    "training": {
        "batch_size": 16,
        "eval_batch_size": 32,
        "gradient_accumulation": 2,
        "epochs": 5,
        "learning_rate": 3.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "optimizer": "adamw_torch",
        "max_grad_norm": 1.0,
        "scheduler": "linear",
        "label_smoothing": 0.0,
        "precision": "bf16",  # fp32 | fp16 | bf16
        "gradient_checkpointing": True,
        "num_workers": 2,
        "dataloader_drop_last": False,
        "log_every_n_steps": 50,
    },

    "loss": {
        "type": "cross_entropy",  # cross_entropy | focal
        "use_class_weight": False,
        "class_weight": "inverse_frequency",  # inverse_frequency | inverse_sqrt_frequency | manual | none
        "class_weight_manual": None,  # {"HATE": w, "OFFENSIVE": w, "CLEAN": w}
        "class_weight_beta": 0.999,
        "focal": {"gamma": 2.0, "alpha": None},
    },

    "dlr": {
        "enabled": False,
        "encoder_lr": 1.0e-5,
        "layer_lr": 3.0e-5,
        "head_lr": 1.0e-4,
        "embedding_lr": None,  # defaults to encoder_lr
        "layer_split": 0.5,  # fraction of blocks treated as "upper"
        "layer_decay": 1.0,  # geometric decay across blocks (<1 shrinks lower layers)
    },

    "stages": None,  # None -> single implicit stage built from `training`
    "curriculum": {"enabled": False},

    "thresholds": {
        "enabled": False,
        "search": "macro_f1",  # macro_f1 | f1 | accuracy
        "grid_size": 101,
        "coordinate_ascent_rounds": 3,
        "init": 0.5,
    },

    "early_stopping": {
        "enabled": True,
        "metric": "macro_f1",  # metric to monitor for best checkpoint + early stop
        "mode": "max",
        "patience": 2,
        "min_delta": 0.0,
    },

    "evaluation": {
        "decision_rule": "argmax",  # argmax | thresholds
        "benchmark": True,
        "benchmark_batch_size": 32,
        "benchmark_warmup": 10,
        "benchmark_runs": 50,
        "save_predictions": True,
    },

    "output": {
        "experiments_dir": "experiments",
        "reports_dir": "reports",
        "checkpoints_dir": "checkpoints",
        "save_checkpoints": True,
        "save_last_checkpoint": True,
    },
}


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


# --------------------------------------------------------------------------- #
# dict helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def load_config(
    path: str | Path,
    overrides: Optional[Iterable[str]] = None,
    use_defaults: bool = True,
) -> Dict[str, Any]:
    """Load ``path``, merge onto :data:`DEFAULT_CONFIG`, apply CLI overrides.

    Overrides use dotted syntax, e.g. ``--set training.batch_size=8 loss.type=focal``.
    Values are parsed as JSON when possible, otherwise kept as strings.
    """
    path = Path(path)
    user_cfg = load_yaml(path)
    cfg = deep_merge(DEFAULT_CONFIG, user_cfg) if use_defaults else dict(user_cfg)
    cfg["_config_path"] = str(path)
    cfg["_config_name"] = path.stem

    if not cfg.get("model_name"):
        cfg["model_name"] = path.stem

    for item in overrides or []:
        apply_override(cfg, item)
    validate_config(cfg)
    return cfg


def apply_override(cfg: Dict[str, Any], item: str) -> None:
    """Apply one ``dotted.key=value`` override in place."""
    if "=" not in item:
        raise ConfigError(f"Override must look like key.path=value, got: {item!r}")
    key, raw_value = item.split("=", 1)
    try:
        value: Any = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    keys = key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``cfg`` with a dotted path."""
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, Mapping) or k not in node:
            return default
        node = node[k]
    return node


def set_(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
_VALID_LOSS = {"cross_entropy", "focal"}
_VALID_OPTIM = {"adamw_torch", "adamw_torch_fused", "adamw_bnb_8bit", "sgd"}
_VALID_SCHED = {"linear", "cosine", "constant", "constant_with_warmup"}
_VALID_PRECISION = {"fp32", "fp16", "bf16"}


def validate_config(cfg: Dict[str, Any]) -> None:
    """Fail fast on the most common configuration mistakes."""
    problems = []

    if cfg.get("head") not in {"standard", "custom"}:
        problems.append(f"head must be 'standard' or 'custom', got {cfg.get('head')!r}")
    if get(cfg, "dataset.num_labels") != 3:
        problems.append("dataset.num_labels must be 3 (HATE / OFFENSIVE / CLEAN)")
    if get(cfg, "loss.type") not in _VALID_LOSS:
        problems.append(f"loss.type must be one of {sorted(_VALID_LOSS)}")
    if get(cfg, "training.optimizer") not in _VALID_OPTIM:
        problems.append(f"training.optimizer must be one of {sorted(_VALID_OPTIM)}")
    if get(cfg, "training.scheduler") not in _VALID_SCHED:
        problems.append(f"training.scheduler must be one of {sorted(_VALID_SCHED)}")
    if get(cfg, "training.precision") not in _VALID_PRECISION:
        problems.append(f"training.precision must be one of {sorted(_VALID_PRECISION)}")
    if int(get(cfg, "training.batch_size", 0)) < 1:
        problems.append("training.batch_size must be >= 1")
    if int(get(cfg, "training.epochs", 0)) < 1 and not cfg.get("stages"):
        problems.append("training.epochs must be >= 1")

    mode = get(cfg, "early_stopping.mode")
    if mode not in {"min", "max"}:
        problems.append("early_stopping.mode must be 'min' or 'max'")

    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
def _strip_private(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def to_plain(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Config as plain YAML-safe python objects (private keys removed)."""
    return _strip_private(dict(cfg))


def dump_yaml(cfg: Mapping[str, Any], path: str | Path, header: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if header:
            fh.write(header.rstrip() + "\n")
        yaml.safe_dump(to_plain(cfg), fh, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return path


def dump_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")
    return path


def resolve_config_path(name_or_path: str) -> Path:
    """Accept either a bare model name or an explicit path to a YAML file."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.is_file():
        return candidate
    candidate = CONFIGS_DIR / name_or_path
    if candidate.is_file():
        return candidate
    candidate = CONFIGS_DIR / f"{name_or_path}.yaml"
    if candidate.is_file():
        return candidate
    raise ConfigError(
        f"Cannot resolve config {name_or_path!r}. Looked for it in {CONFIGS_DIR} "
        f"and as a path. Available: {sorted(p.name for p in CONFIGS_DIR.glob('*.yaml'))}"
    )
