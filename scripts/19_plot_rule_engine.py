"""Step 19 - visualise the lexical rule engine results (plan sections 39, 43, 44).

    python scripts/19_plot_rule_engine.py                      # validation figures
    python scripts/19_plot_rule_engine.py --split test         # test figures

This script never tunes anything: it re-scores the split with the thresholds
already frozen in ``outputs/evaluation/rule_engine/thresholds.json`` (produced by
``scripts/17_evaluate_rule_engine.py --split validation``) and only adds figures
plus a per-sample ``rule_scores.csv``.  The rule ``metrics.json`` written by step
17 is read, never overwritten.

Figures produced in ``<out>/`` (validation) or ``<out>/test/`` (test):

* ``confusion_matrix.png`` / ``confusion_matrix_normalized.png``
* ``per_class_f1.png``
* ``pr_curve.png`` / ``roc_curve.png``
* ``threshold_sweep.png``   (validation threshold-tuning trace)
* ``robustness_detection.png``   (with the missing clean baseline)
* ``latency_distribution.png``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.common.constants import (  # noqa: E402
    EVALUATION_DIR,
    ID2LABEL,
    LABELS,
    NUM_LABELS,
    SPLITS_DIR,
)
from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.curves import precision_recall_curve_data, roc_curve_data  # noqa: E402
from src.evaluation.metrics import apply_decision_rule  # noqa: E402
from src.evaluation.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_per_class_f1_bars,
    plot_pr_curve,
    plot_roc_curve,
    setup_style,
)
from src.evaluation.thresholds import load_thresholds  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.19_plot_rule_engine")

DEFAULT_CONFIG = Path("configs/rule_engine.yaml")
DEFAULT_OUT = EVALUATION_DIR / "rule_engine"
CLEAN_ID = LABELS.index("CLEAN")


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-tty
            pass


def load_split(path: Path, limit: int | None = None) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(int(row["label"]))
            if limit and len(texts) >= limit:
                break
    return texts, labels


def run_engine(engine: RuleEngine, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    hate = np.zeros(len(texts))
    offensive = np.zeros(len(texts))
    latencies: List[float] = []
    for index, text in enumerate(texts):
        started = time.perf_counter()
        result = engine.analyze(text)
        latencies.append((time.perf_counter() - started) * 1000.0)
        hate[index] = float(result["rule_hate_score"])
        offensive[index] = float(result["rule_offensive_score"])
    clean = np.clip(1.0 - np.maximum(hate, offensive), 0.0, 1.0)
    return np.stack([hate, offensive, clean], axis=1), hate, np.asarray(latencies)


def obfuscate(text: str, mode: str) -> str:
    leet = {"o": "0", "a": "4", "e": "3", "i": "1", "s": "5"}
    if mode == "leet":
        return "".join(leet.get(ch, ch) for ch in text)
    if mode == "spaced":
        return " ".join(text)
    if mode == "no_accent":
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    if mode == "repeat":
        return "".join(ch * 2 if ch.isalpha() else ch for ch in text)
    return text


def detection_rate(engine: RuleEngine, texts: Sequence[str], thresholds: Sequence[float]) -> float:
    if not texts:
        return 0.0
    proba, _, _ = run_engine(engine, texts)
    pred = apply_decision_rule(proba, "thresholds", thresholds)
    return float(np.mean(pred != CLEAN_ID))


def robustness_report(
    engine: RuleEngine,
    texts: Sequence[str],
    y_true: Sequence[int],
    thresholds: Sequence[float],
    limit: int = 400,
) -> Dict[str, object]:
    positives = [t for t, y in zip(texts, y_true) if y != CLEAN_ID][:limit]
    modes = ("clean", "leet", "spaced", "no_accent", "repeat")
    report: Dict[str, object] = {"n_positive": len(positives), "modes": {}}
    for mode in modes:
        variants = positives if mode == "clean" else [obfuscate(t, mode) for t in positives]
        rate = detection_rate(engine, variants, thresholds)
        report["modes"][mode] = {
            "detection_rate": round(rate, 4),
            "detected": int(round(rate * len(variants))),
            "total": len(variants),
        }
    clean_rate = float(report["modes"]["clean"]["detection_rate"])
    for mode in modes:
        if mode == "clean":
            continue
        report["modes"][mode]["detection_drop_vs_clean"] = round(
            clean_rate - float(report["modes"][mode]["detection_rate"]), 4
        )
    return report


def write_rule_scores(
    path: Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    hate: np.ndarray,
    offensive: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "true_label", "pred_label", "rule_hate_score", "rule_offensive_score"])
        for index in range(len(y_true)):
            writer.writerow(
                [
                    index,
                    ID2LABEL[int(y_true[index])],
                    ID2LABEL[int(y_pred[index])],
                    round(float(hate[index]), 6),
                    round(float(offensive[index]), 6),
                ]
            )
    return path


def plot_threshold_sweep(threshold_path: Path, out_path: Path) -> Path | None:
    if not threshold_path.exists():
        LOGGER.warning("[19] no thresholds.json at %s - skipping threshold sweep", threshold_path)
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    trace = payload.get("trace", [])
    if not trace:
        return None

    setup_style()
    objectives = [float(step["objective"]) for step in trace]
    positions = list(range(len(objectives)))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(positions, objectives, marker="o", linewidth=1.9, color="#1f77b4", label="macro-F1 objective")
    for position, step in zip(positions, trace):
        ax.annotate(
            f"{step['label']}={step['threshold']:.2f}",
            (position, float(step["objective"])),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(positions)
    ax.set_xlabel("Coordinate-ascent step")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Rule engine threshold tuning (validation only)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOGGER.info("[plot] %s", out_path)
    return out_path


def plot_robustness(report: Dict[str, object], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style()
    modes = report["modes"]
    order = ["clean", "leet", "spaced", "no_accent", "repeat"]
    names = [mode for mode in order if mode in modes]
    rates = [100.0 * float(modes[mode]["detection_rate"]) for mode in names]
    colors = ["#1f77b4" if mode == "clean" else "#d62728" for mode in names]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(names, rates, color=colors, width=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, min(100.0, max(rates + [0.0]) * 1.3 + 2))
    ax.set_ylabel("Detection rate (%)")
    ax.set_xlabel("Input transformation")
    ax.set_title("Rule engine robustness to obfuscation (400 positives)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOGGER.info("[plot] %s", out_path)
    return out_path


def plot_latency(latencies: np.ndarray, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style()
    values = np.asarray(latencies, dtype=float)
    mean_ms = float(values.mean())
    p50 = float(np.percentile(values, 50))
    p95 = float(np.percentile(values, 95))
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    upper = float(np.percentile(values, 99.5))
    ax.hist(values, bins=60, range=(0.0, max(upper, values.min() + 1e-6)), color="#1f77b4", alpha=0.85)
    ax.axvline(p50, color="#2ca02c", linestyle="--", linewidth=1.6, label=f"p50 = {p50:.3f} ms")
    ax.axvline(p95, color="#d62728", linestyle="--", linewidth=1.6, label=f"p95 = {p95:.3f} ms")
    ax.axvline(mean_ms, color="black", linestyle=":", linewidth=1.6, label=f"mean = {mean_ms:.3f} ms")
    ax.set_xlabel("Latency per comment (ms, clipped at p99.5)")
    ax.set_ylabel("Comments")
    ax.set_title(f"Rule engine latency (n = {values.size}, throughput = {throughput:.0f}/s)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOGGER.info("[plot] %s", out_path)
    return out_path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise the lexical rule engine results.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="rule engine YAML config")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="rule engine evaluation directory")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of rows (debug)")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_console()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.out)
    artifacts_dir = out_dir if args.split == "validation" else out_dir / "test"
    threshold_path = out_dir / "thresholds.json"

    metrics_path = artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            f"Missing {metrics_path}. Run "
            f"'python scripts/17_evaluate_rule_engine.py --split {args.split}' first."
        )
    thresholds = load_thresholds(threshold_path)
    if thresholds is None:
        raise SystemExit(
            f"Frozen thresholds not found at {threshold_path}. "
            "Run step 17 on validation first (plan section 44)."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    config = load_rule_engine_config(args.config)
    engine = RuleEngine.from_lexicon(config.lexicon_path, config)

    split_path = Path(SPLITS_DIR) / f"{args.split}.csv"
    texts, y_true = load_split(split_path, args.limit)
    LOGGER.info("[19] %s: %d rows", split_path.name, len(texts))

    proba, hate, latencies = run_engine(engine, texts)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = apply_decision_rule(proba, "thresholds", thresholds)

    write_rule_scores(artifacts_dir / "rule_scores.csv", y_true, y_pred, hate, proba[:, 1])
    LOGGER.info("[19] scores -> %s", artifacts_dir / "rule_scores.csv")

    tag = "validation" if args.split == "validation" else "test"
    plot_confusion_matrix(
        metrics["confusion_matrix"], artifacts_dir / "confusion_matrix.png",
        normalize=False, model_name=f"Rule engine ({tag})",
        title=f"Rule engine confusion matrix ({tag})",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"], artifacts_dir / "confusion_matrix_normalized.png",
        normalize=True, model_name=f"Rule engine ({tag})",
        title=f"Rule engine confusion matrix, row-normalised ({tag})",
    )
    plot_per_class_f1_bars(
        metrics, artifacts_dir / "per_class_f1.png", model_name=f"Rule engine ({tag})",
        title=f"Rule engine per-class F1 ({tag})",
    )
    plot_pr_curve(
        precision_recall_curve_data(y_true_arr, proba),
        artifacts_dir / "pr_curve.png", model_name=f"Rule engine ({tag})",
        title=f"Rule engine precision-recall curve ({tag})",
    )
    plot_roc_curve(
        roc_curve_data(y_true_arr, proba),
        artifacts_dir / "roc_curve.png", model_name=f"Rule engine ({tag})",
        title=f"Rule engine ROC curve ({tag})",
    )
    plot_threshold_sweep(threshold_path, artifacts_dir / "threshold_sweep.png")

    report = robustness_report(engine, texts, y_true, thresholds)
    report["thresholds"] = {ID2LABEL[i]: thresholds[i] for i in range(NUM_LABELS)}
    (artifacts_dir / "robustness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_robustness(report, artifacts_dir / "robustness_detection.png")
    plot_latency(latencies, artifacts_dir / "latency_distribution.png")

    LOGGER.info("[19] artefacts -> %s", artifacts_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
