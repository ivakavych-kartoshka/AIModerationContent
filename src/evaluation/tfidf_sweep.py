"""Grid sweep of the TF-IDF gate thresholds on the validation split (plan 36, 44).

The lexicon is built once and then *pruned* by TF-IDF. The two thresholds
(``min_ratio`` / ``min_score``) therefore control how many entries reach the rule
engine, and the right value is an empirical question. This module answers it
without touching the test set:

1. score every frozen lexicon entry once against the training corpus (the
   expensive step is done a single time);
2. for each ``(min_ratio, min_score)`` pair build the pruned lexicon **in
   memory**, run the rule engine over the *validation* split, optimise the
   per-label decision thresholds and record macro-F1 / per-class F1;
3. report the grid plus the best configuration.

Running on the test split is intentionally unsupported (section 44): the sweep
is a tuning tool, the winning thresholds are frozen and the test split is scored
once by ``scripts/17_evaluate_rule_engine.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common.constants import EVALUATION_DIR, LABELS, LEXICON_PROCESSED_DIR, SPLITS_DIR
from ..common.logging_utils import get_logger
from ..data.lexicon_pruner import LEXICON_FIELDS, collect_term_stats, decide, load_lexicon_rows
from ..data.lexical_labeler import load_training_rows
from ..data.tfidf_filter import TfidfConfig
from ..rule_engine import RuleEngine, load_rule_engine_config
from ..rule_engine.lexical_classifier import Lexicon
from .metrics import apply_decision_rule, compute_metrics
from .thresholds import optimize_thresholds

LOGGER = get_logger("sensitiveai.evaluation.tfidf_sweep")

CLEAN_ID = LABELS.index("CLEAN")

DEFAULT_CONFIG = Path("configs/rule_engine.yaml")
DEFAULT_OUT = EVALUATION_DIR / "rule_engine_tfidf_sweep"

DEFAULT_RATIOS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_SCORES = (0.002, 0.003, 0.004, 0.005, 0.006, 0.008)


def load_split(path: str | Path, limit: Optional[int] = None) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(int(row["label"]))
            if limit and len(texts) >= limit:
                break
    return texts, labels


def to_pseudo_proba(hate: np.ndarray, offensive: np.ndarray) -> np.ndarray:
    clean = np.clip(1.0 - np.maximum(hate, offensive), 0.0, 1.0)
    return np.stack([hate, offensive, clean], axis=1)


def run_engine(engine: RuleEngine, texts: Sequence[str]) -> np.ndarray:
    hate = np.zeros(len(texts))
    offensive = np.zeros(len(texts))
    for index, text in enumerate(texts):
        result = engine.analyze(text)
        hate[index] = float(result["rule_hate_score"])
        offensive[index] = float(result["rule_offensive_score"])
    return to_pseudo_proba(hate, offensive)


def _write_temp_lexicon(directory: Path, rows: Sequence[Dict[str, str]]) -> Path:
    path = directory / "verified_lexicon_pruned.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEXICON_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _evaluate(
    engine: RuleEngine,
    texts: Sequence[str],
    y_true: np.ndarray,
    objective: str,
    grid_size: int,
) -> Dict[str, Any]:
    proba = run_engine(engine, texts)
    payload = optimize_thresholds(y_true, proba, objective=objective, grid_size=grid_size)
    thresholds = payload["threshold_list"]
    y_pred = apply_decision_rule(proba, "thresholds", thresholds)
    metrics = compute_metrics(y_true, y_pred, proba)
    return {
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "accuracy": round(float(metrics["accuracy"]), 6),
        "HATE_f1": round(float(metrics["HATE_f1"]), 6),
        "OFFENSIVE_f1": round(float(metrics["OFFENSIVE_f1"]), 6),
        "CLEAN_f1": round(float(metrics["CLEAN_f1"]), 6),
        "thresholds": {LABELS[i]: thresholds[i] for i in range(len(LABELS))},
    }


def sweep(
    lexicon_path: str | Path = LEXICON_PROCESSED_DIR / "verified_lexicon.csv",
    split: str = "validation",
    config_path: str | Path = DEFAULT_CONFIG,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    scores: Sequence[float] = DEFAULT_SCORES,
    smoothing: float = 1.0,
    objective: str = "macro_f1",
    grid_size: int = 101,
    limit: Optional[int] = None,
    output_dir: str | Path = DEFAULT_OUT,
) -> Dict[str, Any]:
    if split != "validation":
        raise ValueError("the TF-IDF sweep must tune on the validation split (plan section 44)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(SPLITS_DIR) / f"{split}.csv"
    texts, y_list = load_split(split_path, limit)
    y_true = np.asarray(y_list, dtype=int)
    LOGGER.info("[sweep] %s: %d rows", split_path.name, len(texts))

    config = load_rule_engine_config(config_path)
    rows = load_lexicon_rows(lexicon_path)
    allowed = set(config.scoring.min_review_statuses)
    applicable = [row for row in rows if (row.get("review_status") or "").strip() in allowed]
    terms = [row["canonical_form"] or row["term"] for row in applicable]
    LOGGER.info(
        "[sweep] lexicon %s: %d rows (%d with status in %s)",
        Path(lexicon_path).name,
        len(rows),
        len(applicable),
        sorted(allowed),
    )

    train_rows = load_training_rows()
    stats, _rows_by_label = collect_term_stats(terms, train_rows, TfidfConfig(smoothing=smoothing))

    # baseline: the frozen dictionary as-is (gate disabled)
    baseline_engine = RuleEngine(Lexicon.load(lexicon_path, allowed), config)
    baseline = _evaluate(baseline_engine, texts, y_true, objective, grid_size)
    baseline.update({"min_ratio": None, "min_score": None, "n_entries": len(baseline_engine.lexicon), "n_dropped": 0})
    LOGGER.info("[sweep] baseline: macro_f1=%.4f (%d entries)", baseline["macro_f1"], baseline["n_entries"])

    results: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    total = len(ratios) * len(scores)
    step = 0
    with tempfile.TemporaryDirectory(prefix="tfidf_sweep_") as tmp:
        tmp_dir = Path(tmp)
        for ratio in ratios:
            for score in scores:
                step += 1
                cfg = TfidfConfig(enabled=True, min_ratio=ratio, min_score=score, smoothing=smoothing)
                kept: List[Dict[str, str]] = []
                for row, term in zip(applicable, terms):
                    stat = stats.get(term)
                    keep = False if stat is None else decide(stat, cfg)[0]
                    if keep:
                        kept.append(row)
                temp_path = _write_temp_lexicon(tmp_dir, kept)
                engine = RuleEngine(Lexicon.load(temp_path, allowed), config)
                record = _evaluate(engine, texts, y_true, objective, grid_size)
                record.update(
                    {
                        "min_ratio": ratio,
                        "min_score": score,
                        "n_entries": len(kept),
                        "n_dropped": len(applicable) - len(kept),
                    }
                )
                results.append(record)
                LOGGER.info(
                    "[sweep] %d/%d ratio=%.2f score=%.4f -> macro_f1=%.4f (%d entries, -%d)",
                    step,
                    total,
                    ratio,
                    score,
                    record["macro_f1"],
                    record["n_entries"],
                    record["n_dropped"],
                )
                if best is None or record["macro_f1"] > best["macro_f1"]:
                    best = record

    best = best or baseline
    payload = {
        "generated_by": "python scripts/21_sweep_tfidf.py",
        "split": split,
        "config": str(config_path),
        "lexicon": str(lexicon_path),
        "objective": objective,
        "smoothing": smoothing,
        "baseline": baseline,
        "best": best,
        "grid": results,
    }
    (output_dir / "tfidf_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "tfidf_sweep.csv", results)
    LOGGER.info("[sweep] best: %s", json.dumps(best, ensure_ascii=False))
    LOGGER.info("[sweep] artifacts -> %s", output_dir)
    return payload


def _write_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "min_ratio",
        "min_score",
        "macro_f1",
        "accuracy",
        "HATE_f1",
        "OFFENSIVE_f1",
        "CLEAN_f1",
        "n_entries",
        "n_dropped",
        "thresholds",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "thresholds": json.dumps(row["thresholds"])})


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - exotic streams
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.tfidf_sweep",
        description="Sweep the TF-IDF gate thresholds on the validation split (plan 36, 44).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lexicon", default=str(LEXICON_PROCESSED_DIR / "verified_lexicon.csv"), help="Frozen lexicon CSV.")
    p.add_argument("--split", default="validation", choices=["validation"], help="Tuning split (validation only).")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Rule-engine YAML config.")
    p.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS), help="min_ratio grid.")
    p.add_argument("--scores", type=float, nargs="+", default=list(DEFAULT_SCORES), help="min_score grid.")
    p.add_argument("--smoothing", type=float, default=1.0, help="Add-k smoothing inside the IDF.")
    p.add_argument("--objective", default="macro_f1", help="Threshold objective.")
    p.add_argument("--grid-size", type=int, default=101, help="Threshold grid size.")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of validation rows (debug).")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_console()
    args = build_parser().parse_args(argv)
    sweep(
        lexicon_path=Path(args.lexicon),
        split=args.split,
        config_path=Path(args.config),
        ratios=args.ratios,
        scores=args.scores,
        smoothing=args.smoothing,
        objective=args.objective,
        grid_size=args.grid_size,
        limit=args.limit,
        output_dir=Path(args.out),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
