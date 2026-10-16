"""Step 17 - evaluate the lexical rule engine on the held-out splits (plan 39, 44).

    python scripts/17_evaluate_rule_engine.py                      # tune on validation
    python scripts/17_evaluate_rule_engine.py --split test         # frozen thresholds
    python scripts/17_evaluate_rule_engine.py --text "tháº±ng ngu nÃ y"
    python scripts/17_evaluate_rule_engine.py --robustness         # obfuscation recall

Protocol (plan section 44): thresholds are optimised on the VALIDATION split and
frozen into ``thresholds.json``; the TEST split is scored once with those frozen
thresholds and never used for tuning. Running ``--split test`` without a frozen
``thresholds.json`` is refused.

The rule engine outputs two scores per comment (``rule_hate_score``,
``rule_offensive_score`` in ``[0, 1]``). They are turned into a 3-class decision
with the project's own decision rule by building a pseudo-probability vector
``[HATE, OFFENSIVE, CLEAN]`` with ``CLEAN = 1 - max(HATE, OFFENSIVE)`` and reusing
``src.evaluation.thresholds.optimize_thresholds``, so the rule engine is scored
exactly like the neural models.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.common.constants import (  # noqa: E402
    EVALUATION_DIR,
    ID2LABEL,
    LABELS,
    LEXICON_PROCESSED_DIR,
    NUM_LABELS,
    SPLITS_DIR,
)
from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    classification_report_text,
    compute_metrics,
    confusion_matrix_text,
)
from src.evaluation.thresholds import (  # noqa: E402
    describe,
    load_thresholds,
    optimize_thresholds,
    save_thresholds,
)
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.17_evaluate_rule_engine")

DEFAULT_CONFIG = Path("configs/rule_engine.yaml")
DEFAULT_OUT = EVALUATION_DIR / "rule_engine"
CLEAN_ID = LABELS.index("CLEAN")


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-tty
            pass


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_split(path: Path, limit: int | None = None) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
            if limit and len(texts) >= limit:
                break
    return texts, labels


def to_pseudo_proba(hate: np.ndarray, offensive: np.ndarray) -> np.ndarray:
    clean = np.clip(1.0 - np.maximum(hate, offensive), 0.0, 1.0)
    return np.stack([hate, offensive, clean], axis=1)


def run_engine(
    engine: RuleEngine, texts: Sequence[str]
) -> Tuple[np.ndarray, List[float], List[Dict[str, object]]]:
    hate = np.zeros(len(texts))
    offensive = np.zeros(len(texts))
    latencies: List[float] = []
    results: List[Dict[str, object]] = []
    for index, text in enumerate(texts):
        started = time.perf_counter()
        result = engine.analyze(text)
        latencies.append((time.perf_counter() - started) * 1000.0)
        hate[index] = float(result["rule_hate_score"])
        offensive[index] = float(result["rule_offensive_score"])
        results.append(result)
    return to_pseudo_proba(hate, offensive), latencies, results


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def per_class_rows(metrics: Dict[str, object]) -> List[List[object]]:
    rows: List[List[object]] = []
    for name in LABELS:
        rows.append(
            [
                name,
                round(float(metrics[f"{name}_precision"]), 6),
                round(float(metrics[f"{name}_recall"]), 6),
                round(float(metrics[f"{name}_f1"]), 6),
                int(metrics[f"{name}_support"]),
                _round_nan(metrics.get(f"{name}_roc_auc")),
                _round_nan(metrics.get(f"{name}_average_precision")),
            ]
        )
    rows.append(
        [
            "MACRO",
            round(float(metrics["macro_precision"]), 6),
            round(float(metrics["macro_recall"]), 6),
            round(float(metrics["macro_f1"]), 6),
            int(sum(int(metrics[f"{name}_support"]) for name in LABELS)),
            _round_nan(metrics.get("macro_roc_auc")),
            _round_nan(metrics.get("macro_average_precision")),
        ]
    )
    return rows


def _round_nan(value: object) -> object:
    if value is None:
        return ""
    number = float(value)
    return "" if np.isnan(number) else round(number, 6)


def write_confusion_csv(path: Path, metrics: Dict[str, object]) -> Path:
    matrix = metrics["confusion_matrix"]
    rows = [[f"actual_{name}"] + list(row) for name, row in zip(LABELS, matrix)]
    return write_csv(path, ["actual\\predicted"] + list(LABELS), rows)


def write_error_analysis(
    path: Path,
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    proba: np.ndarray,
    results: Sequence[Dict[str, object]],
    positive: bool,
    limit: int = 300,
) -> Path:
    header = [
        "text",
        "true_label",
        "pred_label",
        "rule_hate_score",
        "rule_offensive_score",
        "n_matches",
        "top_matches",
    ]
    selected: List[Tuple[float, List[object]]] = []
    for index, (truth, pred) in enumerate(zip(y_true, y_pred)):
        is_positive = truth != CLEAN_ID
        if positive != is_positive or truth == pred:
            continue
        matches = results[index].get("matches", [])[:5]
        rendered = " | ".join(
            f"{m['term']}({m['label']}:{m['match_type']}:{m['score']})" for m in matches
        )
        row = [
            texts[index],
            ID2LABEL[int(truth)],
            ID2LABEL[int(pred)],
            round(float(proba[index, 0]), 4),
            round(float(proba[index, 1]), 4),
            len(results[index].get("matches", [])),
            rendered,
        ]
        score = float(max(proba[index, 0], proba[index, 1]))
        selected.append((score, row))
    selected.sort(key=lambda item: item[0], reverse=True)
    return write_csv(path, header, [row for _, row in selected[:limit]])


def latency_summary(latencies: Sequence[float]) -> Dict[str, object]:
    values = np.asarray(latencies, dtype=float)
    return {
        "n": int(values.size),
        "mean_ms": round(float(values.mean()), 4),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "max_ms": round(float(values.max()), 4),
        "throughput_per_sec": round(1000.0 / float(values.mean()), 2) if values.size else 0.0,
    }


# --------------------------------------------------------------------------- #
# robustness (section 44)
# --------------------------------------------------------------------------- #
def obfuscate(text: str, mode: str) -> str:
    leet = {"o": "0", "a": "4", "e": "3", "i": "1", "s": "5"}
    if mode == "leet":
        return "".join(leet.get(ch, ch) for ch in text)
    if mode == "spaced":
        return " ".join(text)
    if mode == "no_accent":
        import unicodedata

        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    if mode == "repeat":
        return "".join(ch * 2 if ch.isalpha() else ch for ch in text)
    return text


def run_robustness(
    engine: RuleEngine,
    texts: Sequence[str],
    y_true: Sequence[int],
    thresholds: Sequence[float],
    limit: int = 400,
) -> Dict[str, object]:
    from src.evaluation.metrics import apply_decision_rule

    positives = [t for t, y in zip(texts, y_true) if y != CLEAN_ID][:limit]
    report: Dict[str, object] = {"n_positive": len(positives), "modes": {}}
    for mode in ("leet", "spaced", "no_accent", "repeat"):
        variants = [obfuscate(text, mode) for text in positives]
        proba, _, _ = run_engine(engine, variants)
        pred = apply_decision_rule(proba, "thresholds", thresholds)
        detected = int(np.sum(pred != CLEAN_ID))
        report["modes"][mode] = {
            "detection_rate": round(detected / len(variants), 4) if variants else 0.0,
            "detected": detected,
            "total": len(variants),
        }
    return report


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the lexical rule engine.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="rule engine YAML config")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--objective", default="macro_f1", help="threshold objective")
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--limit", type=int, default=None, help="cap the number of rows (debug)")
    parser.add_argument("--text", default=None, help="analyze a single text and exit")
    parser.add_argument("--robustness", action="store_true", help="run obfuscation recall")
    return parser.parse_args(list(argv))


def _single_text(engine: RuleEngine, text: str) -> int:
    result = engine.analyze(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_console()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # test results go to a sub-directory so a single frozen test run never
    # overwrites the validation artifacts that produced the thresholds.
    artifacts_dir = out_dir if args.split == "validation" else out_dir / "test"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = load_rule_engine_config(args.config)
    engine = RuleEngine.from_lexicon(config.lexicon_path, config)

    if args.text is not None:
        return _single_text(engine, args.text)

    split_path = Path(SPLITS_DIR) / f"{args.split}.csv"
    texts, y_true = load_split(split_path, args.limit)
    LOGGER.info("[17] %s: %d rows", split_path.name, len(texts))

    threshold_path = out_dir / "thresholds.json"
    frozen = None
    if args.split == "test":
        frozen = load_thresholds(threshold_path)
        if frozen is None:
            raise SystemExit(
                f"Frozen thresholds not found at {threshold_path}.\n"
                "Run 'python scripts/17_evaluate_rule_engine.py --split validation' first "
                "(plan section 44: never tune on the test split)."
            )
        LOGGER.info("[17] using frozen thresholds from %s", threshold_path)

    proba, latencies, results = run_engine(engine, texts)
    y_true_arr = np.asarray(y_true, dtype=int)

    if args.split == "validation":
        payload = optimize_thresholds(
            y_true_arr, proba, objective=args.objective, grid_size=args.grid_size
        )
        payload["split"] = "validation"
        payload["config"] = str(args.config)
        save_thresholds(payload, threshold_path)
        thresholds = payload["threshold_list"]
        print(describe(payload))
    else:
        thresholds = frozen

    from src.evaluation.metrics import apply_decision_rule

    y_pred = apply_decision_rule(proba, "thresholds", thresholds)
    metrics = compute_metrics(y_true_arr, y_pred, proba)

    payload_metrics = {
        **metrics,
        "split": args.split,
        "config": str(args.config),
        "lexicon": str(LEXICON_PROCESSED_DIR / "verified_lexicon.csv"),
        "dictionary_version": engine.lexicon.dictionary_version,
        "thresholds": {ID2LABEL[i]: thresholds[i] for i in range(NUM_LABELS)},
        "objective": args.objective if args.split == "validation" else "frozen",
        "n_lexicon_entries": len(engine.lexicon),
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(payload_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "classification_report.txt").write_text(
        classification_report_text(y_true_arr, y_pred), encoding="utf-8"
    )
    (artifacts_dir / "confusion_matrix.txt").write_text(
        confusion_matrix_text(y_true_arr, y_pred), encoding="utf-8"
    )
    write_csv(
        artifacts_dir / "per_class_metrics.csv",
        ["label", "precision", "recall", "f1", "support", "roc_auc", "average_precision"],
        per_class_rows(metrics),
    )
    write_confusion_csv(artifacts_dir / "confusion_matrix.csv", metrics)
    write_error_analysis(
        artifacts_dir / "false_positive_analysis.csv", texts, y_true, y_pred, proba, results, positive=False
    )
    write_error_analysis(
        artifacts_dir / "false_negative_analysis.csv", texts, y_true, y_pred, proba, results, positive=True
    )
    (artifacts_dir / "latency.json").write_text(
        json.dumps(latency_summary(latencies), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOGGER.info(
        "[17] %s macro_f1=%.4f accuracy=%.4f (HATE f1=%.4f, OFFENSIVE f1=%.4f)",
        args.split,
        metrics["macro_f1"],
        metrics["accuracy"],
        metrics["HATE_f1"],
        metrics["OFFENSIVE_f1"],
    )

    if args.robustness:
        report = run_robustness(engine, texts, y_true, thresholds)
        report["thresholds"] = {ID2LABEL[i]: thresholds[i] for i in range(NUM_LABELS)}
        (artifacts_dir / "robustness.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("[17] robustness: %s", json.dumps(report["modes"], ensure_ascii=False))

    LOGGER.info("[17] artifacts -> %s", artifacts_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
