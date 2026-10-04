"""Download the raw ViHSD csv files from the Hugging Face Hub.

Source of truth: https://huggingface.co/datasets/uitnlp/vihsd
Paper / repo:   https://github.com/sonlam1102/vihsd

Files fetched: ``train.csv``, ``dev.csv``, ``test.csv`` with columns
``free_text`` (str) and ``label_id`` (int, native ViHSD ids
0=CLEAN, 1=OFFENSIVE, 2=HATE).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

from ..common.constants import DATASET_HF_ID, VIHSD_RAW_DIR
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

RAW_FILES = ("train.csv", "dev.csv", "test.csv")
NATIVE_TEXT_COLUMN = "free_text"
NATIVE_LABEL_COLUMN = "label_id"


def download_vihsd(
    output_dir: str | Path = VIHSD_RAW_DIR,
    repo_id: str = DATASET_HF_ID,
    force: bool = False,
) -> Dict[str, Path]:
    """Fetch the three raw csv files. Returns ``{file_name: local_path}``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download ViHSD.  pip install huggingface_hub"
        ) from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    for name in RAW_FILES:
        target = out / name
        if target.is_file() and not force:
            LOGGER.info("[download] %s already present -> %s", name, target)
            paths[name] = target
            continue
        LOGGER.info("[download] fetching %s from %s ...", name, repo_id)
        cached = hf_hub_download(repo_id=repo_id, filename=name, repo_type="dataset")
        shutil.copyfile(cached, target)
        LOGGER.info("[download] saved %s (%d bytes)", target, target.stat().st_size)
        paths[name] = target

    write_raw_readme(out)
    return paths


def write_raw_readme(output_dir: str | Path = VIHSD_RAW_DIR) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# ViHSD (raw)\n\n"
            "Auto-downloaded from https://huggingface.co/datasets/uitnlp/vihsd\n\n"
            "Files: `train.csv`, `dev.csv`, `test.csv`\n"
            "Columns: `free_text`, `label_id`\n\n"
            "Native ViHSD label ids: `0 = CLEAN`, `1 = OFFENSIVE`, `2 = HATE`\n\n"
            "Citation: Luu, Son T. and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy (2021).\n"
            "*A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media "
            "Texts.* IEA/AIE 2021, Springer, pp. 415-426.\n",
            encoding="utf-8",
        )
    return readme


def local_raw_files(output_dir: str | Path = VIHSD_RAW_DIR) -> List[Path]:
    out = Path(output_dir)
    return [out / name for name in RAW_FILES if (out / name).is_file()]


if __name__ == "__main__":  # pragma: no cover
    download_vihsd()
