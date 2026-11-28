"""CLI for the English data pipeline: download -> preprocess -> fixed split.

Usage
-----
    python -m src.data.prepare_data_en
    python -m src.data.prepare_data_en --stage verify summary
    python -m src.data.prepare_data_en --overwrite

Config: ``configs/data_split_en.yaml`` (paths, ratios, seed, cleaning).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.config import load_yaml
from ..common.constants import (
    HATE_SPEECH_OFFENSIVE_PROCESSED_DIR,
    HATE_SPEECH_OFFENSIVE_RAW_DIR,
    HATE_SPEECH_OFFENSIVE_SPLITS_DIR,
    ID2LABEL,
    LABEL_COLUMN,
    TEXT_COLUMN,
)
from ..common.logging_utils import get_logger
from ..common.paths import project_path
from ..preprocessing.text_cleaning import build_cleaning_config
from .download_en import DATASET_HF_ID, download_hate_speech_offensive
from .preprocess_en import preprocess_en
from .split import dataset_summary, load_manifest, split_fingerprints, verify_split_fingerprint
from .split_en import build_split_en

LOGGER = get_logger("sensitiveai.data.en")
STAGES = ("download", "preprocess", "split", "verify", "summary")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.prepare_data_en",
        description="Download, clean and split the English Davidson dataset once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/data_split_en.yaml")
    p.add_argument("--stage", nargs="*", default=["download", "preprocess", "split", "summary", "verify"], choices=STAGES)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--splits-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-dedupe", action="store_true", help="Keep duplicate texts (default: drop duplicates).")
    p.add_argument("--show-samples", type=int, default=0)
    return p


def _paths(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "raw": Path(args.raw_dir) if args.raw_dir else project_path(cfg.get("raw_dir", str(HATE_SPEECH_OFFENSIVE_RAW_DIR))),
        "processed": Path(args.processed_dir)
        if args.processed_dir
        else project_path(cfg.get("processed_dir", str(HATE_SPEECH_OFFENSIVE_PROCESSED_DIR))),
        "splits": Path(args.splits_dir)
        if args.splits_dir
        else project_path(cfg.get("splits_dir", str(HATE_SPEECH_OFFENSIVE_SPLITS_DIR))),
    }


def print_samples(path: Path, n: int) -> None:
    if n <= 0 or not path.is_file():
        return
    import pandas as pd

    df = pd.read_csv(path, dtype={TEXT_COLUMN: "string"}).head(n)
    LOGGER.info("--- cleaned English samples ---")
    for _, row in df.iterrows():
        LOGGER.info("  [%s] %s", ID2LABEL[int(row[LABEL_COLUMN])], str(row[TEXT_COLUMN])[:160])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = project_path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
    paths = _paths(cfg, args)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    ratios = cfg.get("ratios") or {}
    overwrite = bool(args.overwrite or cfg.get("overwrite", False))

    LOGGER.info("[data-en] dataset: %s", DATASET_HF_ID)
    for key, path in paths.items():
        LOGGER.info("[data-en] %-11s -> %s", key, path)

    if "download" in args.stage:
        download_hate_speech_offensive(paths["raw"], force=overwrite)
    if "preprocess" in args.stage:
        cleaning = build_cleaning_config(cfg.get("cleaning"))
        outputs = preprocess_en(
            raw_dir=paths["raw"],
            processed_dir=paths["processed"],
            cleaning=cleaning,
            dedupe=not args.no_dedupe,
        )
        LOGGER.info("[data-en] processed: %s", {k: str(v) for k, v in outputs.items()})
        print_samples(paths["processed"] / "all.csv", args.show_samples)
    if "split" in args.stage:
        build_split_en(
            processed_csv=paths["processed"] / "all.csv",
            splits_dir=paths["splits"],
            seed=seed,
            train_ratio=float(ratios.get("train", 0.8)),
            validation_ratio=float(ratios.get("validation", 0.1)),
            test_ratio=float(ratios.get("test", 0.1)),
            overwrite=overwrite,
        )
    if "verify" in args.stage:
        ok, report = verify_split_fingerprint(paths["splits"])
        LOGGER.info("[data-en] split verification: %s", json.dumps(report, ensure_ascii=False, indent=2))
        if not ok:
            return 1
    if "summary" in args.stage:
        summary = dataset_summary(paths["splits"])
        LOGGER.info("[data-en] split summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
        manifest = load_manifest(paths["splits"]) or {}
        LOGGER.info("[data-en] fingerprints: %s", {k: v[:16] for k, v in split_fingerprints(paths["splits"]).items()})
        LOGGER.info("[data-en] strategy=%s seed=%s", manifest.get("strategy"), manifest.get("seed"))

    LOGGER.info("[data-en] done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
