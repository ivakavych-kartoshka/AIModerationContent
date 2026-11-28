"""Stage 2 (English): ONE fixed stratified split for the Davidson dataset.

The dataset ships no official split, so the split is 80/10/10 stratified by the
canonical label with a fixed seed.  The files and manifest mirror the Vietnamese
pipeline exactly:

    data/splits_en/train.csv
    data/splits_en/validation.csv
    data/splits_en/test.csv
    data/splits_en/split_manifest.json

Every training/evaluation entry point calls ``verify_split_fingerprint()``, so a
changed English split is detected the same way as the Vietnamese one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from ..common.constants import (
    HATE_SPEECH_OFFENSIVE_PROCESSED_DIR,
    HATE_SPEECH_OFFENSIVE_SPLITS_DIR,
    ID2LABEL,
    LABEL_COLUMN,
    TEXT_COLUMN,
)
from ..common.logging_utils import get_logger
from .split import (  # reuse the exact same fingerprint/format helpers
    COLUMNS,
    MANIFEST_NAME,
    SPLIT_FILES,
    SplitError,
    label_distribution,
    label_ratios,
    sha256_of,
)

LOGGER = get_logger(__name__)


def build_split_en(
    processed_csv: str | Path = HATE_SPEECH_OFFENSIVE_PROCESSED_DIR / "all.csv",
    splits_dir: str | Path = HATE_SPEECH_OFFENSIVE_SPLITS_DIR,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    overwrite: bool = False,
) -> Dict[str, Path]:
    processed_csv = Path(processed_csv)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = splits_dir / MANIFEST_NAME

    if manifest_path.is_file() and not overwrite:
        LOGGER.info("[split-en] manifest already exists -> %s (use overwrite=True to rebuild)", manifest_path)
        return {name: splits_dir / fname for name, fname in SPLIT_FILES.items()}

    if not processed_csv.is_file():
        raise SplitError(
            f"Processed English csv missing: {processed_csv}\n"
            "Run 'python scripts/52_prepare_english_data.py --stage download preprocess' first."
        )
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise SplitError("ratios must sum to 1.0")
    df = pd.read_csv(processed_csv, dtype={TEXT_COLUMN: "string"})
    if df[TEXT_COLUMN].isna().any() or (df[TEXT_COLUMN].astype(str).str.strip() == "").any():
        raise SplitError("processed English data contains empty text")

    positions = list(range(len(df)))
    train_pos, holdout_pos = train_test_split(
        positions,
        test_size=validation_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=df[LABEL_COLUMN].to_numpy(),
    )
    rel_val = test_ratio / (validation_ratio + test_ratio)
    val_pos, test_pos = train_test_split(
        holdout_pos,
        test_size=rel_val,
        random_state=seed,
        shuffle=True,
        stratify=df.iloc[holdout_pos][LABEL_COLUMN].to_numpy(),
    )

    parts = {
        "train": df.iloc[sorted(train_pos)].reset_index(drop=True),
        "validation": df.iloc[sorted(val_pos)].reset_index(drop=True),
        "test": df.iloc[sorted(test_pos)].reset_index(drop=True),
    }
    manifest: Dict[str, object] = {
        "strategy": "stratified",
        "dataset": "tdavidson/hate_speech_offensive",
        "seed": seed,
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "label_mapping": {str(k): v for k, v in ID2LABEL.items()},
        "columns": COLUMNS,
        "splits": {},
    }

    outputs: Dict[str, Path] = {}
    for name, frame in parts.items():
        for col in COLUMNS:
            if col not in frame.columns:
                frame[col] = "" if col != LABEL_COLUMN else -1
        frame = frame[COLUMNS].copy()
        path = splits_dir / SPLIT_FILES[name]
        frame.to_csv(path, index=False, encoding="utf-8")
        outputs[name] = path
        manifest["splits"][name] = {
            "file": path.name,
            "num_samples": int(len(frame)),
            "sha256": sha256_of(path),
            "label_distribution": label_distribution(frame),
            "label_ratios": label_ratios(frame),
        }
        LOGGER.info("[split-en] %-10s n=%6d  %s", name, len(frame), manifest["splits"][name]["label_distribution"])

    if int(manifest["splits"]["test"]["num_samples"]) == 0:
        raise SplitError("English test split is empty")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[split-en] manifest -> %s", manifest_path)
    return outputs


if __name__ == "__main__":  # pragma: no cover
    build_split_en()
