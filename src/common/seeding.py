"""Reproducibility helpers.

Every entry point calls :func:`seed_everything` so that the four compared
systems share the same seed (plan section 4: "fixed random seed").
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed python / numpy / torch and (optionally) force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover - depends on the torch build
            pass
    return seed


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - dataloader hook
    """Make DataLoader workers deterministic."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Return a seeded generator (used by the DataLoader shuffle)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
