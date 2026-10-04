"""Stage 1 of the data pipeline: raw ViHSD csv -> cleaned csv with canonical labels.

Outputs (plan sections 2 & 3)
-----------------------------
``data/processed/vihsd/train.csv``, ``dev.csv``, ``test.csv``
    columns: ``text``, ``label`` (0=HATE, 1=OFFENSIVE, 2=CLEAN), ``label_name``,
    ``source`` (which raw file the row came from), ``source_row``.

``data/processed/vihsd/preprocessing_report.json``
    cleaning options, per-split row counts before/after, label distribution and
    a small before/after sample.  This file is the evidence that no test row
    was silently dropped for any reason other than "has no usable text".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..common.constants import (
    ID2LABEL,
    LABEL_COLUMN,
    LABEL_NAME_COLUMN,
    TEXT_COLUMN,
    VIHSD_ID2LABEL,
    VIHSD_PROCESSED_DIR,
    VIHSD_RAW_DIR,
    VIHSD_TO_CANONICAL,
)
from ..common.logging_utils import get_logger
from ..preprocessing.text_cleaning import (
    CleaningConfig,
    build_cleaning_config,
    clean_dataframe,
)
from .download import NATIVE_LABEL_COLUMN, NATIVE_TEXT_COLUMN, RAW_FILES, download_vihsd

LOGGER = get_logger(__name__)

RAW_SPLIT_NAMES = {"train.csv": "train", "dev.csv": "dev", "test.csv": "test"}


def remap_labels(label_ids: pd.Series) -> pd.Series:
    """ViHSD native ids -> canonical ids (HATE=0, OFFENSIVE=1, CLEAN=2)."""
    mapped = label_ids.map(VIHSD_TO_CANONICAL)
    unmapped = mapped[mapped.isna()].unique().tolist()
    if unmapped:
        raise ValueError(
            f"Unexpected label_id values {unmapped}. Known ViHSD ids: {sorted(VIHSD_ID2LABEL)}"
        )
    return mapped.astype("int64")


def _label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    return {ID2LABEL[int(k)]: int(v) for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))}


def load_raw_split(path: Path) -> pd.DataFrame:
    """Read one raw csv and normalise its column names."""
    df = pd.read_csv(path, dtype={NATIVE_TEXT_COLUMN: "string"})
    missing = {NATIVE_TEXT_COLUMN, NATIVE_LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}; found {list(df.columns)}")
    df = df.rename(columns={NATIVE_TEXT_COLUMN: TEXT_COLUMN, NATIVE_LABEL_COLUMN: "vihsd_label"})
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype("string")
    return df


def preprocess(
    raw_dir: str | Path = VIHSD_RAW_DIR,
    processed_dir: str | Path = VIHSD_PROCESSED_DIR,
    cleaning: Optional[CleaningConfig] = None,
    download_if_missing: bool = True,
) -> Dict[str, Path]:
    """Clean every raw split and write canonical csv files.

    Returns ``{split_name: path}`` for ``train``, ``dev``, ``test``.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaning = cleaning or CleaningConfig()

    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        if not download_if_missing:
            raise FileNotFoundError(f"Missing raw files in {raw_dir}: {missing}")
        LOGGER.warning("Raw files missing (%s) - downloading from the Hugging Face Hub.", missing)
        download_vihsd(raw_dir)

    report: Dict[str, object] = {
        "cleaning_config": cleaning.to_dict(),
        "label_mapping": {
            "canonical": {str(k): v for k, v in ID2LABEL.items()},
            "vihsd_native": {str(k): v for k, v in VIHSD_ID2LABEL.items()},
            "vihsd_to_canonical": {str(k): int(v) for k, v in VIHSD_TO_CANONICAL.items()},
        },
        "splits": {},
        "examples": {},
    }

    outputs: Dict[str, Path] = {}
    for file_name, split_name in RAW_SPLIT_NAMES.items():
        raw_path = raw_dir / file_name
        if not raw_path.is_file():
            LOGGER.warning("Skipping missing raw file %s", raw_path)
            continue

        df = load_raw_split(raw_path)
        df["source_row"] = df.index.astype(int)
        df["source"] = split_name

        before_example = str(df[TEXT_COLUMN].iloc[0]) if len(df) else ""

        df[LABEL_COLUMN] = remap_labels(df["vihsd_label"])
        cleaned_df, stats = clean_dataframe(df, text_column=TEXT_COLUMN, cfg=cleaning)

        cleaned_df[LABEL_NAME_COLUMN] = cleaned_df[LABEL_COLUMN].map(ID2LABEL)
        cleaned_df = cleaned_df[[TEXT_COLUMN, LABEL_COLUMN, LABEL_NAME_COLUMN, "source", "source_row"]]

        out_path = processed_dir / f"{split_name}.csv"
        cleaned_df.to_csv(out_path, index=False, encoding="utf-8")
        outputs[split_name] = out_path

        after_example = str(cleaned_df[TEXT_COLUMN].iloc[0]) if len(cleaned_df) else ""
        report["splits"][split_name] = {
            **stats,
            "dropped_rows": int(stats["input_rows"] - stats["output_rows"]),
            "drop_rate": round((stats["input_rows"] - stats["output_rows"]) / max(stats["input_rows"], 1), 6),
            "label_distribution": _label_distribution(cleaned_df),
            "output_file": str(out_path),
        }
        report["examples"][split_name] = {"before": before_example, "after": after_example}

        LOGGER.info(
            "[preprocess] %-5s  in=%6d  out=%6d  dropped=%4d  labels=%s",
            split_name,
            stats["input_rows"],
            stats["output_rows"],
            stats["input_rows"] - stats["output_rows"],
            report["splits"][split_name]["label_distribution"],
        )

    report_path = processed_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[preprocess] report -> %s", report_path)
    return outputs


def build_cleaning_from_config(config: Dict) -> CleaningConfig:
    """Read the optional ``preprocessing`` block of a model config."""
    return build_cleaning_config(config.get("preprocessing") or {})


if __name__ == "__main__":  # pragma: no cover
    preprocess()
