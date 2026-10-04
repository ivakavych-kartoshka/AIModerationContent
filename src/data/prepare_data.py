"""CLI for the data pipeline: download -> preprocess -> fixed stratified split.

Usage
-----
    python -m src.data.prepare_data                       # everything, official split
    python -m src.data.prepare_data --strategy stratified  # re-split 80/10/10
    python -m src.data.prepare_data --stage split --overwrite
    python -m src.data.prepare_data --show-samples 5

Stages
------
``download``    fetch train.csv / dev.csv / test.csv from the Hugging Face Hub
``preprocess``  clean the text and remap the labels to HATE=0 / OFFENSIVE=1 / CLEAN=2
``split``       create the ONE fixed split shared by all four models
``verify``      check the split against the recorded fingerprints
``summary``     print the row counts and label distributions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.config import load_yaml
from ..common.constants import (
    ID2LABEL,
    LABEL_COLUMN,
    TEXT_COLUMN,
    VIHSD_PROCESSED_DIR,
    VIHSD_RAW_DIR,
)
from ..common.logging_utils import get_logger
from ..common.paths import project_path
from ..preprocessing.text_cleaning import build_cleaning_config
from .download import download_vihsd
from .preprocess import preprocess
from .split import (
    build_split,
    dataset_summary,
    load_manifest,
    split_fingerprints,
    verify_split_fingerprint,
)

LOGGER = get_logger("sensitiveai.data")

STAGES = ("download", "preprocess", "split", "verify", "summary")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.prepare_data",
        description="Download, clean and split ViHSD once for all four models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/data_split.yaml", help="Data-split config file.")
    p.add_argument("--stage", nargs="*", default=["download", "preprocess", "split", "summary", "verify"],
                   choices=STAGES, help="Which stages to run.")
    p.add_argument("--strategy", default=None, choices=["official", "stratified"], help="Override the split strategy.")
    p.add_argument("--seed", type=int, default=None, help="Override the fixed seed.")
    p.add_argument("--raw-dir", default=None, help="Override the raw data directory.")
    p.add_argument("--processed-dir", default=None, help="Override the processed data directory.")
    p.add_argument("--splits-dir", default=None, help="Override the output split directory.")
    p.add_argument("--overwrite", action="store_true", help="Rebuild the split even if it already exists.")
    p.add_argument("--show-samples", type=int, default=0, help="Print N cleaned examples (0 = none).")
    return p


def _paths(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "raw": Path(args.raw_dir) if args.raw_dir else project_path(cfg.get("raw_dir", str(VIHSD_RAW_DIR))),
        "processed": Path(args.processed_dir)
        if args.processed_dir
        else project_path(cfg.get("processed_dir", str(VIHSD_PROCESSED_DIR))),
        "splits": Path(args.splits_dir)
        if args.splits_dir
        else project_path(cfg.get("splits_dir", "data/splits")),
    }


def print_samples(processed_dir: Path, n: int) -> None:
    import pandas as pd

    for split in ("train", "dev", "test"):
        path = processed_dir / f"{split}.csv"
        if not path.is_file() or n <= 0:
            continue
        df = pd.read_csv(path, dtype={TEXT_COLUMN: "string"}).head(n)
        LOGGER.info("--- cleaned %s samples ---", split)
        for _, row in df.iterrows():
            LOGGER.info("  [%s] %s", ID2LABEL[int(row[LABEL_COLUMN])], str(row[TEXT_COLUMN])[:160])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(project_path(args.config)) if (project_path(args.config)).is_file() else {}
    paths = _paths(cfg, args)
    strategy = args.strategy or str(cfg.get("strategy", "official"))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    ratios = cfg.get("ratios") or {}
    overwrite = bool(args.overwrite or cfg.get("overwrite", False))

    for key, path in paths.items():
        LOGGER.info("[data] %-11s -> %s", key, path)

    if "download" in args.stage:
        download_vihsd(paths["raw"])

    if "preprocess" in args.stage:
        cleaning = build_cleaning_config(cfg.get("cleaning"))
        outputs = preprocess(raw_dir=paths["raw"], processed_dir=paths["processed"], cleaning=cleaning)
        LOGGER.info("[data] processed files: %s", {k: str(v) for k, v in outputs.items()})
        print_samples(paths["processed"], args.show_samples)

    if "split" in args.stage:
        build_split(
            processed_dir=paths["processed"],
            splits_dir=paths["splits"],
            strategy=strategy,
            seed=seed,
            train_ratio=float(ratios.get("train", 0.8)),
            validation_ratio=float(ratios.get("validation", 0.1)),
            test_ratio=float(ratios.get("test", 0.1)),
            overwrite=overwrite,
        )

    if "verify" in args.stage:
        ok, report = verify_split_fingerprint(paths["splits"])
        LOGGER.info("[data] split verification: %s", json.dumps(report, ensure_ascii=False, indent=2))
        if not ok:
            return 1

    if "summary" in args.stage:
        summary = dataset_summary(paths["splits"])
        LOGGER.info("[data] split summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
        manifest = load_manifest(paths["splits"]) or {}
        LOGGER.info("[data] fingerprints: %s", {k: v[:16] for k, v in split_fingerprints(paths["splits"]).items()})
        LOGGER.info("[data] strategy=%s seed=%s", manifest.get("strategy"), manifest.get("seed"))

    LOGGER.info("[data] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
