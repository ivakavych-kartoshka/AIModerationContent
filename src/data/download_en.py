"""Download the English Davidson hate-speech/offensive-language dataset.

Source: https://huggingface.co/datasets/tdavidson/hate_speech_offensive
Paper: Davidson et al., ICWSM 2017 - Automated Hate Speech Detection and
       the Problem of Offensive Language

Mapping to canonical (same as Vietnamese project):
    hate-speech (0) -> HATE      (0)
    offensive-language (1) -> OFFENSIVE (1)
    neither (2) -> CLEAN         (2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from ..common.constants import HATE_SPEECH_OFFENSIVE_RAW_DIR
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

DATASET_HF_ID = "tdavidson/hate_speech_offensive"
FILE_NAME = "hate_speech_offensive.parquet"
RAW_TEXT_COLUMN = "tweet"
RAW_LABEL_COLUMN = "class"


def download_hate_speech_offensive(
    output_dir: str | Path = HATE_SPEECH_OFFENSIVE_RAW_DIR,
    repo_id: str = DATASET_HF_ID,
    force: bool = False,
) -> Dict[str, Path]:
    from huggingface_hub import hf_hub_download

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / FILE_NAME
    if target.is_file() and not force:
        LOGGER.info("[download-en] %s already present -> %s", FILE_NAME, target)
        return {"parquet": target}
    LOGGER.info("[download-en] fetching from %s ...", repo_id)
    cached = hf_hub_download(repo_id=repo_id, filename="data/train-00000-of-00001.parquet", repo_type="dataset")
    import shutil

    shutil.copyfile(cached, target)
    LOGGER.info("[download-en] saved %s (%d bytes)", target, target.stat().st_size)
    meta = out / "dataset_info.json"
    meta.write_text(
        json.dumps(
            {
                "source": f"https://huggingface.co/datasets/{repo_id}",
                "paper": "Davidson et al., ICWSM 2017",
                "raw_label_map": {"0": "hate-speech", "1": "offensive-language", "2": "neither"},
                "canonical_map": {"0": "HATE", "1": "OFFENSIVE", "2": "CLEAN"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"parquet": target}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else Path(HATE_SPEECH_OFFENSIVE_RAW_DIR) / FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run download_hate_speech_offensive()")
    return pd.read_parquet(path)


if __name__ == "__main__":
    download_hate_speech_offensive()
