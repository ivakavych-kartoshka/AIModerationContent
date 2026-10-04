"""Stage 2 of the data pipeline: ONE fixed, stratified split shared by all models.

The plan is explicit about this (sections 4 & 5):

* one single split directory ``data/splits/{train,validation,test}.csv``;
* stratified by HATE / OFFENSIVE / CLEAN;
* fixed random seed;
* **never** re-split per model;
* the test set must keep its full sample count - nothing may be dropped after
  the split.

Two strategies are supported:

``official`` (default)
    Keep the ViHSD authors' own train / dev / test partition.  The published
    test set is preserved 1:1, which makes the numbers directly comparable with
    prior Vietnamese hate-speech work and structurally guarantees zero leakage
    between the official train and test portions.

``stratified``
    Concatenate all three official files and re-split 80/10/10 with
    ``sklearn.model_selection.train_test_split(stratify=labels,
    random_state=seed)``.  Use this only for ablations; note that in this mode
    the test set contains rows that the original authors had used for training,
    so absolute numbers are *not* comparable with the literature.

Both modes write ``data/splits/split_manifest.json`` containing a SHA-256
fingerprint of each csv.  :func:`verify_split_fingerprint` is called by every
training / evaluation entry point so a mismatching split can never go unnoticed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..common.constants import (
    ID2LABEL,
    LABEL_COLUMN,
    LABEL_NAME_COLUMN,
    PROJECT_ROOT,
    SPLITS_DIR,
    TEXT_COLUMN,
    VIHSD_PROCESSED_DIR,
)
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
MANIFEST_NAME = "split_manifest.json"
COLUMNS = [TEXT_COLUMN, LABEL_COLUMN, LABEL_NAME_COLUMN, "source", "source_row"]

VALID_STRATEGIES = ("official", "stratified")


class SplitError(RuntimeError):
    """Raised when the split cannot be built or fails verification."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    return {ID2LABEL[int(k)]: int(v) for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))}


def label_ratios(df: pd.DataFrame) -> Dict[str, float]:
    total = max(len(df), 1)
    return {k: round(v / total, 6) for k, v in label_distribution(df).items()}


def _read_processed(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for split in ("train", "dev", "test"):
        path = processed_dir / f"{split}.csv"
        if not path.is_file():
            raise SplitError(
                f"Processed file missing: {path}\nRun the preprocessing step first:\n"
                "    python -m src.data.preprocess"
            )
        frames[split] = pd.read_csv(path, dtype={TEXT_COLUMN: "string"})
    return frames


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def _split_official(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Keep the authors' partition: train -> train, dev -> validation, test -> test."""
    return {
        "train": frames["train"].reset_index(drop=True),
        "validation": frames["dev"].reset_index(drop=True),
        "test": frames["test"].reset_index(drop=True),
    }


def _split_stratified(
    frames: Dict[str, pd.DataFrame],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> Dict[str, pd.DataFrame]:
    """Re-split the pooled data with stratification on the canonical label."""
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"Ratios must sum to 1.0, got {total:.4f}")

    pooled = pd.concat(
        [frames["train"], frames["dev"], frames["test"]], ignore_index=True, sort=False
    )
    # original index -> stable position, so `source_row` stays unique per source file
    positions = list(range(len(pooled)))

    train_pos, holdout_pos = train_test_split(
        positions,
        test_size=validation_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=pooled[LABEL_COLUMN].to_numpy(),
    )
    rel_val = test_ratio / (validation_ratio + test_ratio)
    val_pos, test_pos = train_test_split(
        holdout_pos,
        test_size=rel_val,
        random_state=seed,
        shuffle=True,
        stratify=pooled.iloc[holdout_pos][LABEL_COLUMN].to_numpy(),
    )

    out = {}
    for name, pos in (("train", train_pos), ("validation", val_pos), ("test", test_pos)):
        idx = sorted(pos)
        out[name] = pooled.iloc[idx].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_split(
    processed_dir: str | Path = VIHSD_PROCESSED_DIR,
    splits_dir: str | Path = SPLITS_DIR,
    strategy: str = "official",
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Create the fixed split once and write it to ``splits_dir``."""
    if strategy not in VALID_STRATEGIES:
        raise SplitError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")

    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = splits_dir / MANIFEST_NAME
    if manifest_path.is_file() and not overwrite:
        LOGGER.info("[split] manifest already exists -> %s (use overwrite=True to rebuild)", manifest_path)
        return {name: splits_dir / fname for name, fname in SPLIT_FILES.items()}

    frames = _read_processed(processed_dir)
    if strategy == "official":
        parts = _split_official(frames)
    else:
        parts = _split_stratified(frames, seed, train_ratio, validation_ratio, test_ratio)

    outputs: Dict[str, Path] = {}
    manifest: Dict[str, object] = {
        "strategy": strategy,
        "seed": seed,
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "label_mapping": {str(k): v for k, v in ID2LABEL.items()},
        "columns": COLUMNS,
        "splits": {},
    }

    for name, df in parts.items():
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            df = df.copy()
            for col in missing:
                df[col] = "" if col == TEXT_COLUMN else (-1 if col == LABEL_COLUMN else "")
        df = df[COLUMNS].copy()
        # The split is the last chance to lose rows: guard it explicitly.
        if df[TEXT_COLUMN].isna().any() or (df[TEXT_COLUMN].astype(str).str.strip() == "").any():
            raise SplitError(f"split '{name}' contains rows with empty text - refusing to write it")

        path = splits_dir / SPLIT_FILES[name]
        df.to_csv(path, index=False, encoding="utf-8")
        outputs[name] = path
        manifest["splits"][name] = {
            "file": path.name,
            "num_samples": int(len(df)),
            "sha256": sha256_of(path),
            "label_distribution": label_distribution(df),
            "label_ratios": label_ratios(df),
        }
        LOGGER.info(
            "[split] %-10s n=%6d  %s",
            name,
            len(df),
            manifest["splits"][name]["label_distribution"],
        )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[split] manifest -> %s", manifest_path)

    _assert_test_integrity(manifest, parts)
    return outputs


def _assert_test_integrity(manifest: Dict, parts: Dict[str, pd.DataFrame]) -> None:
    """Plan rule: the test set must not lose samples after the split."""
    strategy = manifest["strategy"]
    n_test_written = int(manifest["splits"]["test"]["num_samples"])
    if strategy == "official":
        n_test_source = len(parts["test"])
        if n_test_written != n_test_source:
            raise SplitError(
                f"Test set size changed during splitting: source={n_test_source} written={n_test_written}"
            )
    if n_test_written == 0:
        raise SplitError("Test split is empty")


def load_split(splits_dir: str | Path = SPLITS_DIR) -> Dict[str, pd.DataFrame]:
    """Load the fixed split as ``{"train": df, "validation": df, "test": df}``."""
    splits_dir = Path(splits_dir)
    out: Dict[str, pd.DataFrame] = {}
    for name, fname in SPLIT_FILES.items():
        path = splits_dir / fname
        if not path.is_file():
            raise SplitError(
                f"Missing split file {path}\nBuild the fixed split first:\n"
                "    python -m src.data.split"
            )
        out[name] = pd.read_csv(path, dtype={TEXT_COLUMN: "string"})
    return out


def load_manifest(splits_dir: str | Path = SPLITS_DIR) -> Optional[Dict]:
    path = Path(splits_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_split_fingerprint(
    splits_dir: str | Path = SPLITS_DIR,
    expected: Optional[Dict[str, str]] = None,
    raise_on_mismatch: bool = True,
) -> Tuple[bool, Dict[str, object]]:
    """Check the split files on disk against the manifest fingerprints.

    ``expected`` may be a ``{split_name: sha256}`` mapping coming from a run's
    saved config; when provided, the current split must match it exactly.
    """
    splits_dir = Path(splits_dir)
    manifest = load_manifest(splits_dir)
    problems: List[str] = []

    if manifest is None:
        problems.append(f"missing manifest {splits_dir / MANIFEST_NAME}")
    else:
        for name, fname in SPLIT_FILES.items():
            path = splits_dir / fname
            if not path.is_file():
                problems.append(f"missing split file {path}")
                continue
            recorded = manifest["splits"].get(name, {}).get("sha256")
            actual = sha256_of(path)
            if recorded and actual != recorded:
                problems.append(
                    f"fingerprint mismatch for {fname}: manifest={recorded[:12]} actual={actual[:12]}"
                )

    if expected:
        for name, sha in expected.items():
            path = splits_dir / SPLIT_FILES[name]
            if not path.is_file():
                problems.append(f"missing split file {path}")
                continue
            actual = sha256_of(path)
            if actual != sha:
                problems.append(f"{name}: expected {sha[:12]} but found {actual[:12]}")

    ok = not problems
    report = {"ok": ok, "problems": problems, "splits_dir": str(splits_dir)}
    if problems:
        message = "Split verification failed:\n  - " + "\n  - ".join(problems)
        if raise_on_mismatch:
            raise SplitError(message)
        LOGGER.warning(message)
    return ok, report


def split_fingerprints(splits_dir: str | Path = SPLITS_DIR) -> Dict[str, str]:
    """``{split_name: sha256}`` for the current split files."""
    splits_dir = Path(splits_dir)
    return {name: sha256_of(splits_dir / fname) for name, fname in SPLIT_FILES.items() if (splits_dir / fname).is_file()}


def dataset_summary(splits_dir: str | Path = SPLITS_DIR) -> Dict[str, object]:
    """Counts + distributions used by ``model_summary.txt`` (plan section 27)."""
    frames = load_split(splits_dir)
    manifest = load_manifest(splits_dir)
    return {
        "splits_dir": str(Path(splits_dir)),
        "strategy": (manifest or {}).get("strategy"),
        "seed": (manifest or {}).get("seed"),
        "num_train": int(len(frames["train"])),
        "num_validation": int(len(frames["validation"])),
        "num_test": int(len(frames["test"])),
        "train_label_distribution": label_distribution(frames["train"]),
        "validation_label_distribution": label_distribution(frames["validation"]),
        "test_label_distribution": label_distribution(frames["test"]),
    }


def default_splits_dir(config: Optional[Dict] = None) -> Path:
    """Resolve the splits directory of a model config (project-relative)."""
    if config:
        raw = config.get("dataset", {}).get("splits_dir", "data/splits")
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return SPLITS_DIR


if __name__ == "__main__":  # pragma: no cover
    build_split()
