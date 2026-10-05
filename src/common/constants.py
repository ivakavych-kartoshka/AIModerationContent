"""Project-wide constants and path resolution for SensitiveAI.

The label mapping is FIXED by the research plan (section 2.1) and must never be
changed between the four models, otherwise the comparison is invalid:

    HATE      -> 0
    OFFENSIVE -> 1
    CLEAN     -> 2
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SENSITIVEAI_ROOT", SRC_DIR.parent.parent)).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = OUTPUTS_DIR / "reports"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
PAPER_DIR = OUTPUTS_DIR / "paper"

VIHSD_RAW_DIR = RAW_DIR / "vihsd"
VIHSD_PROCESSED_DIR = PROCESSED_DIR / "vihsd"

# --------------------------------------------------------------------------- #
# Labels  (plan section 2.1 - FIXED, do not change)
# --------------------------------------------------------------------------- #
LABELS = ("HATE", "OFFENSIVE", "CLEAN")
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

#: Native ViHSD integer ids, as published by uitnlp/vihsd.
#:   0 -> CLEAN, 1 -> OFFENSIVE, 2 -> HATE
VIHSD_ID2LABEL = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
VIHSD_LABEL2ID = {v: k for k, v in VIHSD_ID2LABEL.items()}

#: Remap table: ViHSD native id -> our canonical id.
VIHSD_TO_CANONICAL = {
    VIHSD_LABEL2ID["HATE"]: LABEL2ID["HATE"],
    VIHSD_LABEL2ID["OFFENSIVE"]: LABEL2ID["OFFENSIVE"],
    VIHSD_LABEL2ID["CLEAN"]: LABEL2ID["CLEAN"],
}

#: Canonical id -> ViHSD native id (used when writing raw files back).
CANONICAL_TO_VIHSD = {v: k for k, v in VIHSD_TO_CANONICAL.items()}

# --------------------------------------------------------------------------- #
# Dataset coordinates
# --------------------------------------------------------------------------- #
DATASET_NAME = "uitnlp/vihsd"
DATASET_HF_ID = "uitnlp/vihsd"
DATASET_REPO = "https://github.com/sonlam1102/vihsd"
DATASET_PAPER = "Luu et al., 2021 - A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts"

BASE_MODEL_DEFAULT = "microsoft/mdeberta-v3-base"

#: The four systems compared in the research plan (section 1).
MODEL_NAMES = (
    "sensitiveai-vi",
    "sensitiveai-vi-custom",
    "sensitiveai-vi-customdlr2stage",
    "sensitiveai-vi-custom-curriculum",
)

#: Column names used in every generated CSV split file.
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
LABEL_NAME_COLUMN = "label_name"
