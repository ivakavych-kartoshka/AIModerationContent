"""Stage 1 (English): raw Davidson parquet -> cleaned csv with canonical labels.

The dataset has a single annotated pool (no official split), so this stage only
cleans, remaps and de-duplicates.  The fixed train/validation/test split is
created by :mod:`src.data.split_en`.

Output:
    data/processed/hate_speech_offensive/all.csv
        columns: ``text``, ``label`` (0=HATE, 1=OFFENSIVE, 2=CLEAN),
        ``label_name``, ``source``, ``source_row``
    data/processed/hate_speech_offensive/preprocessing_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..common.constants import (
    HATE_SPEECH_OFFENSIVE_PROCESSED_DIR,
    HATE_SPEECH_OFFENSIVE_RAW_DIR,
    ID2LABEL,
    LABEL_COLUMN,
    LABEL_NAME_COLUMN,
    TEXT_COLUMN,
)
from ..common.logging_utils import get_logger
from ..preprocessing.text_cleaning import CleaningConfig, clean_dataframe
from .download_en import RAW_LABEL_COLUMN, RAW_TEXT_COLUMN, download_hate_speech_offensive, load_raw

LOGGER = get_logger(__name__)

#: Davidson native ids and the project's canonical ids happen to coincide.
DAVIDSON_TO_CANONICAL = {0: 0, 1: 1, 2: 2}
DAVIDSON_LABELS = {0: "HATE", 1: "OFFENSIVE", 2: "CLEAN"}


def remap_labels(series: pd.Series) -> pd.Series:
    mapped = series.map(DAVIDSON_TO_CANONICAL)
    unmapped = mapped[mapped.isna()].unique().tolist()
    if unmapped:
        raise ValueError(f"Unexpected Davidson labels {unmapped}; known: {sorted(DAVIDSON_LABELS)}")
    return mapped.astype("int64")


def _distribution(df: pd.DataFrame) -> Dict[str, int]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    return {ID2LABEL[int(k)]: int(v) for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))}


def preprocess_en(
    raw_dir: str | Path = HATE_SPEECH_OFFENSIVE_RAW_DIR,
    processed_dir: str | Path = HATE_SPEECH_OFFENSIVE_PROCESSED_DIR,
    cleaning: Optional[CleaningConfig] = None,
    download_if_missing: bool = True,
    dedupe: bool = True,
) -> Dict[str, Path]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaning = cleaning or CleaningConfig()

    parquet = raw_dir / "hate_speech_offensive.parquet"
    if not parquet.is_file():
        if not download_if_missing:
            raise FileNotFoundError(f"Missing {parquet}")
        LOGGER.warning("[preprocess-en] raw file missing - downloading from the Hugging Face Hub")
        download_hate_speech_offensive(raw_dir)

    df = load_raw(parquet)
    missing = {RAW_TEXT_COLUMN, RAW_LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing columns {sorted(missing)}; found {list(df.columns)}")

    df = df.rename(columns={RAW_TEXT_COLUMN: TEXT_COLUMN, RAW_LABEL_COLUMN: "davidson_label"})
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype("string")
    df["source_row"] = df.index.astype(int)
    df["source"] = "davidson"

    input_rows = len(df)
    df[LABEL_COLUMN] = remap_labels(df["davidson_label"])
    cleaned, stats = clean_dataframe(df, text_column=TEXT_COLUMN, cfg=cleaning)
    cleaned[LABEL_NAME_COLUMN] = cleaned[LABEL_COLUMN].map(ID2LABEL)

    duplicates_removed = 0
    if dedupe:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=[TEXT_COLUMN], keep="first")
        duplicates_removed = before - len(cleaned)

    columns = [TEXT_COLUMN, LABEL_COLUMN, LABEL_NAME_COLUMN, "source", "source_row"]
    cleaned = cleaned[columns].reset_index(drop=True)

    out_path = processed_dir / "all.csv"
    cleaned.to_csv(out_path, index=False, encoding="utf-8")

    report = {
        "dataset": "tdavidson/hate_speech_offensive",
        "paper": "Davidson et al., ICWSM 2017",
        "raw_label_map": {"0": "hate-speech", "1": "offensive-language", "2": "neither"},
        "canonical_label_map": {str(k): v for k, v in ID2LABEL.items()},
        "cleaning_config": cleaning.to_dict(),
        "input_rows": int(input_rows),
        "output_rows": int(len(cleaned)),
        "dropped_rows": int(input_rows - len(cleaned)),
        "duplicates_removed": int(duplicates_removed),
        "label_distribution": _distribution(cleaned),
        "label_ratios": {
            label: round(count / max(len(cleaned), 1), 6)
            for label, count in _distribution(cleaned).items()
        },
        "output_file": str(out_path),
    }
    report_path = processed_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(
        "[preprocess-en] in=%d out=%d dedupe=%d labels=%s",
        input_rows,
        len(cleaned),
        duplicates_removed,
        report["label_distribution"],
    )
    return {"all": out_path, "report": report_path}


if __name__ == "__main__":  # pragma: no cover
    preprocess_en()
