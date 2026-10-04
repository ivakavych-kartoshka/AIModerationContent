"""Filesystem layout helpers (run directories, artifact paths)."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .constants import (
    CHECKPOINTS_DIR,
    EVALUATION_DIR,
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)


def project_path(path_like: str | Path) -> Path:
    """Resolve ``path_like`` against the project root when it is relative."""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def run_dir_name(run_id: str) -> str:
    """Normalise a run id such as ``1`` / ``run_1`` / ``run_001`` -> ``run_001``."""
    digits = re.sub(r"\D", "", str(run_id))
    return f"run_{int(digits):03d}" if digits else "run_000"


def next_run_id(model_name: str, reports_root: Path | None = None) -> str:
    """Return the next free ``run_XXX`` id for ``model_name``.

    ``reports_root`` is the *base* reports directory; ``model_name`` is appended
    to it, so ``--reports-dir D:/tmp`` yields ``D:/tmp/<model>/run_XXX``.
    """
    root = _model_root(model_name, reports_root, REPORTS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    used = []
    for child in root.glob("run_*"):
        digits = re.sub(r"\D", "", child.name)
        if digits:
            used.append(int(digits))
    return f"run_{max(used) + 1 if used else 1:03d}"


def _model_root(model_name: str, root: Path | None, default: Path) -> Path:
    return (Path(root) if root else default) / model_name


def experiment_dir(model_name: str, run_id: str, root: Path | None = None) -> Path:
    return _model_root(model_name, root, EXPERIMENTS_DIR) / run_dir_name(run_id)


def report_dir(model_name: str, run_id: str, root: Path | None = None) -> Path:
    return _model_root(model_name, root, REPORTS_DIR) / run_dir_name(run_id)


def checkpoint_dir(model_name: str, run_id: str, root: Path | None = None) -> Path:
    return _model_root(model_name, root, CHECKPOINTS_DIR) / run_dir_name(run_id)


def per_model_eval_dir(model_name: str, root: Path | None = None) -> Path:
    base = Path(root) if root else EVALUATION_DIR / "per_model"
    return base / model_name


def comparison_dir(root: Path | None = None) -> Path:
    return (Path(root) if root else EVALUATION_DIR / "comparison")


def evaluation_log_path(root: Path | None = None) -> Path:
    base = Path(root) if root else EVALUATION_DIR
    return base / "logs" / "evaluation_log.csv"


def find_best_report_dir(model_name: str, reports_root: Path | None = None, run_id: Optional[str] = None) -> Path:
    """Locate the report directory of a finished run.

    Order of preference: explicit ``run_id`` -> ``run_pointer.txt`` -> highest
    numbered run that contains ``metrics.json``.
    """
    root = _model_root(model_name, reports_root, REPORTS_DIR)
    if run_id:
        candidate = report_dir(model_name, run_id, reports_root)
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Report directory not found: {candidate}")

    pointer = root / "run_pointer.txt"
    if pointer.is_file():
        candidate = root / pointer.read_text(encoding="utf-8").strip()
        if candidate and (root / candidate).is_dir():
            return root / candidate

    # A run counts as finished once training produced its summary, whether or
    # not it has been evaluated yet (evaluation is the *next* step).
    candidates = [
        d
        for d in root.glob("run_*")
        if (d / "training_summary.json").is_file() or (d / "metrics.json").is_file()
    ]
    if candidates:
        return sorted(candidates)[-1]
    raise FileNotFoundError(
        f"No finished run found for '{model_name}' under {root}. Train the model first."
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Log preservation
# --------------------------------------------------------------------------- #
#: Files that mark a run directory as "already used" - writing into such a
#: directory would overwrite the log of a previous experiment.
RUN_MARKER_FILES = (
    "train.log",
    "training_log.csv",
    "training_summary.json",
    "metrics.json",
    "evaluate.log",
)


def guard_free_run_dir(run_dir: str | Path, *, allow_reuse: bool = False) -> Path:
    """Refuse to write into a run directory that already holds a past run.

    Every experiment gets its own ``run_XXX`` folder, so an occupied folder
    means an ``--run-id`` collision.  Silently appending to it would mix two
    experiments in one set of logs, which is exactly what this project must not
    do, so the run is stopped with an actionable message instead.

    Returns the directory when it is free (or when ``allow_reuse`` is set).
    """
    path = Path(run_dir)
    if not path.is_dir():
        return path
    occupied = [name for name in RUN_MARKER_FILES if (path / name).is_file()]
    if not occupied or allow_reuse:
        return path
    listing = ", ".join(occupied)
    raise FileExistsError(
        f"Run directory {path} already contains a previous experiment ({listing}).\n"
        "Logs are never overwritten on purpose. Use one of:\n"
        "  * drop --run-id so the next free run_XXX is allocated automatically, or\n"
        "  * pass a run id that is not in use yet (see reports/<model>/explain.md), or\n"
        "  * pass --allow-run-reuse to append to this run on purpose."
    )


def archive_run_artifacts(
    run_dir: str | Path,
    patterns: Iterable[str],
    *,
    history_root: Optional[str | Path] = None,
    stamp: Optional[str] = None,
) -> Optional[Path]:
    """Move already-produced artifacts of a run into ``history/<stamp>/``.

    Used by the evaluation entry point so that re-scoring the same run keeps
    the previous numbers instead of destroying them.  Returns the history
    directory, or ``None`` when there was nothing to archive.
    """
    path = Path(run_dir)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(p for p in path.glob(pattern) if p.exists()))
    if not matches:
        return None
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(history_root) if history_root else path / "history"
    target = ensure_dir(root / f"eval_{stamp}")
    for item in matches:
        destination = target / item.name
        if destination.exists():
            destination = target / f"{item.stem}__{stamp}{item.suffix}"
        shutil.move(str(item), str(destination))
    return target
