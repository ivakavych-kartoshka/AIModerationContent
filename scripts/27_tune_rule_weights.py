"""Step 27 - tune the rule engine SCORING weights on the validation split.

    python scripts/27_tune_rule_weights.py                       # coordinate descent, 3 rounds
    python scripts/27_tune_rule_weights.py --rounds 1 --limit 800   # quick trial
    python scripts/27_tune_rule_weights.py --config configs/rule_engine_tuned.yaml
    python scripts/27_tune_rule_weights.py --no-write-config     # search only

Protocol (Rule_Engine_Plan.md sections 39 and 44):

* the search reads **only** ``data/splits/validation.csv`` - the test split is
  never loaded;
* for every candidate weight set the decision thresholds are re-optimised on
  validation (``optimize_thresholds``, objective macro-F1), exactly like
  ``scripts/17_evaluate_rule_engine.py`` does, so weights and thresholds are
  always scored together;
* the search is a greedy coordinate descent over ``scoring.*`` keys only
  (severity, match_factors, negation/quote/context factors, aggregation_cap);
  matcher / obfuscation / normalisation are left untouched;
* every trial is logged to ``outputs/evaluation/rule_engine_tune/`` and the
  winning weights are written to ``configs/rule_engine_tuned.yaml``.

This tunes the knobs that `match_score = severity x confidence x match_factor
x context_factor` (and the saturating aggregation) expose. Thresholds stay
frozen afterwards: run script 17 on validation to freeze ``thresholds.json``,
then score test exactly once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from src.evaluation.metrics import apply_decision_rule, compute_metrics  # noqa: E402
from src.evaluation.thresholds import optimize_thresholds  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402
from src.rule_engine.config import RuleEngineConfig, validate  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.27_tune_rule_weights")

DEFAULT_CONFIG = Path("configs/rule_engine.yaml")
DEFAULT_OUT = EVALUATION_DIR / "rule_engine_tune"
DEFAULT_BEST_CONFIG = Path("configs/rule_engine_tuned.yaml")
CLEAN_ID = LABELS.index("CLEAN")

#: ``dotted.key -> candidate values`` tried by the coordinate descent. Order is
#: the search order of one round; the current value of every key is skipped.
PARAM_GRID: Dict[str, List[Any]] = {
    "severity.HATE": [0.75, 0.85, 0.90, 0.95, 1.00],
    "severity.OFFENSIVE": [0.40, 0.50, 0.60, 0.70, 0.85, 1.00],
    "match_factors.exact": [0.90, 1.00],
    "match_factors.accentless_alias": [0.00, 0.10, 0.15, 0.30, 0.50, 0.75, 0.90],
    "match_factors.leet": [0.40, 0.60, 0.80, 1.00],
    "match_factors.repeat_collapsed": [0.40, 0.55, 0.75, 0.90, 1.00],
    "match_factors.separator_inserted": [0.40, 0.60, 0.80, 1.00],
    "match_factors.obfuscated_combined": [0.30, 0.45, 0.60, 0.80, 1.00],
    "match_factors.fuzzy": [0.00, 0.30, 0.45, 0.60, 0.80, 1.00],
    "negation_factor": [0.00, 0.25, 0.40, 0.50, 0.70, 1.00],
    "quote_factor": [0.30, 0.45, 0.60, 0.80, 1.00],
    "context_required_factor": [0.50, 0.70, 0.85, 1.00],
    "aggregation_cap": [3, 5, 10, 20, 50],
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
        reader = csv.DictReader(fh)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
            if limit and len(texts) >= limit:
                break
    return texts, labels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_param(cfg: RuleEngineConfig, key: str) -> Any:
    if key.startswith("severity."):
        return cfg.scoring.severity[key.split(".", 1)[1]]
    if key.startswith("match_factors."):
        return cfg.scoring.match_factors[key.split(".", 1)[1]]
    return getattr(cfg.scoring, key)


def set_param(cfg: RuleEngineConfig, key: str, value: Any) -> None:
    if key.startswith("severity."):
        cfg.scoring.severity[key.split(".", 1)[1]] = float(value)
    elif key.startswith("match_factors."):
        cfg.scoring.match_factors[key.split(".", 1)[1]] = float(value)
    else:
        setattr(cfg.scoring, key, value)


def snapshot(cfg: RuleEngineConfig) -> Dict[str, Any]:
    return {key: get_param(cfg, key) for key in PARAM_GRID}


def build_config(base_path: Path, values: Dict[str, Any]) -> RuleEngineConfig:
    """Fresh config from the base YAML with ``values`` applied on top."""
    cfg = load_rule_engine_config(base_path)
    for key, value in values.items():
        set_param(cfg, key, value)
    validate(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# evaluation (validation split only)
# --------------------------------------------------------------------------- #
def run_scores(engine: RuleEngine, texts: Sequence[str]) -> np.ndarray:
    hate = np.zeros(len(texts))
    offensive = np.zeros(len(texts))
    for index, text in enumerate(texts):
        result = engine.analyze(text)
        hate[index] = float(result["rule_hate_score"])
        offensive[index] = float(result["rule_offensive_score"])
    clean = np.clip(1.0 - np.maximum(hate, offensive), 0.0, 1.0)
    return np.stack([hate, offensive, clean], axis=1)


def evaluate(
    cfg: RuleEngineConfig,
    texts: Sequence[str],
    y_true: np.ndarray,
    objective: str,
    grid_size: int,
) -> Dict[str, Any]:
    """Score one candidate: build the engine, tune thresholds, return metrics."""
    engine = RuleEngine.from_lexicon(cfg.lexicon_path, cfg)
    proba = run_scores(engine, texts)
    payload = optimize_thresholds(y_true, proba, objective=objective, grid_size=grid_size)
    thresholds = payload["threshold_list"]
    y_pred = apply_decision_rule(proba, "thresholds", thresholds)
    metrics = compute_metrics(y_true, y_pred, proba)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "HATE_f1": float(metrics["HATE_f1"]),
        "OFFENSIVE_f1": float(metrics["OFFENSIVE_f1"]),
        "CLEAN_f1": float(metrics["CLEAN_f1"]),
        "thresholds": [float(t) for t in thresholds],
        "n_lexicon_entries": len(engine.lexicon),
    }


def _quiet_loggers(quiet: bool) -> None:
    """Silence the per-trial chatter (engine ready, threshold convergence)."""
    for name in (
        "sensitiveai.rule_engine",
        "sensitiveai.rule_engine.lexicon",
        "sensitiveai.lexical_classifier",
        "src.evaluation.thresholds",
    ):
        logging.getLogger(name).setLevel(logging.WARNING if quiet else logging.INFO)


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
TRIAL_FIELDS = [
    "round",
    "trial",
    "param",
    "value",
    "macro_f1",
    "accuracy",
    "HATE_f1",
    "OFFENSIVE_f1",
    "CLEAN_f1",
    "th_HATE",
    "th_OFFENSIVE",
    "th_CLEAN",
    "accepted",
    "seconds",
]


def write_trials(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_config(
    cfg: RuleEngineConfig,
    base_config: str,
    lexicon_sha: str,
    n_rows: int,
    baseline: Dict[str, Any],
    best: Dict[str, Any],
    changes: Dict[str, List[Any]],
    generated_at: str,
) -> str:
    """Serialise the winning config with a provenance header (plan section 44)."""
    scoring = cfg.scoring
    lines = [
        "# =============================================================================",
        "# Rule engine configuration - TUNED on the validation split (plan sections 39, 44)",
        "#",
        "#   python scripts/27_tune_rule_weights.py       # produced this file",
        "#   python scripts/17_evaluate_rule_engine.py --config configs/rule_engine_tuned.yaml",
        "#",
        "# Protocol: greedy coordinate descent over `scoring.*` only; every candidate",
        "# re-optimises the decision thresholds on the VALIDATION split (macro-F1).",
        "# The TEST split is never loaded during the search (section 44).",
        f"# Generated: {generated_at}",
        f"# Base config : {base_config}",
        f"# Lexicon     : {cfg.lexicon_path} (sha256={lexicon_sha[:16]}, {best['n_lexicon_entries']} entries)",
        f"# Validation  : {n_rows} rows",
        (
            f"# macro-F1    : {baseline['macro_f1']:.4f} -> {best['macro_f1']:.4f}"
            f"  (HATE {baseline['HATE_f1']:.4f} -> {best['HATE_f1']:.4f},"
            f" OFFENSIVE {baseline['OFFENSIVE_f1']:.4f} -> {best['OFFENSIVE_f1']:.4f},"
            f" CLEAN {baseline['CLEAN_f1']:.4f} -> {best['CLEAN_f1']:.4f})"
        ),
        "#",
        "# Changed weights:",
    ]
    for key, (old, new) in changes.items():
        lines.append(f"#   {key}: {old} -> {new}")
    if not changes:
        lines.append("#   (none - the search found no improvement over the base config)")
    lines += [
        "#",
        "# History: an earlier tuning pass on an older 1828-entry lexicon lowered",
        "# match_factors.accentless_alias 0.90 -> 0.15 for the same reason (section 38:",
        "# the accentless view is an auxiliary signal). This search re-derives every",
        "# weight from scratch on the current lexicon.",
        "# Every weight is still a heuristic hypothesis, not a calibrated probability.",
        "# =============================================================================",
        "",
        f"lexicon_path: {cfg.lexicon_path}",
        f"dictionary_version: {cfg.dictionary_version}",
        "",
        "# --- section 37: input normalization -------------------------------------- #",
        "normalize:",
        f"  unicode_form: {cfg.normalize.unicode_form}",
        f"  lowercase: {_yaml_scalar(cfg.normalize.lowercase)}",
        f"  remove_invisible: {_yaml_scalar(cfg.normalize.remove_invisible)}",
        f"  collapse_whitespace: {_yaml_scalar(cfg.normalize.collapse_whitespace)}",
        f"  max_repeat_char: {cfg.normalize.max_repeat_char}        # \"nguuuu\" -> \"nguu\"; 0 disables the cap",
        f"  max_text_chars: {cfg.normalize.max_text_chars}",
        "",
        "# --- section 38: anti-obfuscation matching views -------------------------- #",
        "obfuscation:",
        f"  enabled: {_yaml_scalar(cfg.obfuscation.enabled)}",
        f"  accentless: {_yaml_scalar(cfg.obfuscation.accentless)}           # \"con cho\"  (auxiliary signal only)",
        f"  leet: {_yaml_scalar(cfg.obfuscation.leet)}                 # \"c0n ch0\"",
        f"  collapse_repeats: {_yaml_scalar(cfg.obfuscation.collapse_repeats)}         # \"nguuu\" -> \"ngu\"",
        f"  separator_stripping: {_yaml_scalar(cfg.obfuscation.separator_stripping)}  # \"n g u\"",
        f"  min_compact_term_len: {cfg.obfuscation.min_compact_term_len}",
        f"  max_variant_chars: {cfg.obfuscation.max_variant_chars}",
        "",
        "# --- dictionary matching --------------------------------------------------- #",
        "matcher:",
        f"  min_term_len: {cfg.matcher.min_term_len}",
        f"  fuzzy_enabled: {_yaml_scalar(cfg.matcher.fuzzy_enabled)}",
        f"  fuzzy_max_distance: {cfg.matcher.fuzzy_max_distance}",
        f"  fuzzy_min_term_len: {cfg.matcher.fuzzy_min_term_len}",
        f"  max_matches: {cfg.matcher.max_matches}",
        "",
        "# --- section 39: scoring (TUNED, see header) ------------------------------ #",
        "scoring:",
        "  severity:                  # label severity, section 39 table",
    ]
    for label in ("HATE", "OFFENSIVE"):
        lines.append(f"    {label}: {scoring.severity[label]:.4g}")
    lines.append("  match_factors:             # confidence of each match type (TUNED)")
    for name, value in scoring.match_factors.items():
        lines.append(f"    {name}: {value:.4g}")
    lines += [
        f"  negation_factor: {scoring.negation_factor:.4g}      # slur preceded by \"không / chẳng / ...\"",
        f"  quote_factor: {scoring.quote_factor:.4g}         # slur inside quotation marks",
        f"  context_required_factor: {scoring.context_required_factor:.4g}",
        f"  aggregation_cap: {scoring.aggregation_cap}        # max matches per label in 1 - prod(1 - s_i)",
        "  min_review_statuses:       # which lexicon entries may match",
    ]
    for status in scoring.min_review_statuses:
        lines.append(f"    - {status}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune rule-engine scoring weights on the validation split (plan section 39/44)."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="base rule engine YAML config")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation"],
        help="tuning split; test is refused (plan section 44)",
    )
    parser.add_argument("--objective", default="macro_f1", help="threshold objective re-optimised per trial")
    parser.add_argument("--grid-size", type=int, default=101, help="threshold search grid")
    parser.add_argument("--rounds", type=int, default=3, help="coordinate-descent rounds (stop early when flat)")
    parser.add_argument("--min-gain", type=float, default=0.001, help="minimum macro-F1 gain to accept a change")
    parser.add_argument("--limit", type=int, default=None, help="cap validation rows (debug)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where trials.csv / tuning_report.json go")
    parser.add_argument("--best-config", default=str(DEFAULT_BEST_CONFIG), help="where the winning YAML is written")
    parser.add_argument("--no-write-config", action="store_true", help="search only; do not touch any YAML")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_console()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.split != "validation":  # argparse already restricts it; keep the plan-44 guard explicit
        raise SystemExit("plan section 44: weights may only be tuned on the validation split.")

    base_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(SPLITS_DIR) / "validation.csv"
    texts, labels = load_split(split_path, args.limit)
    y_true = np.asarray(labels, dtype=int)
    LOGGER.info("[27] validation: %d rows (objective=%s)", len(texts), args.objective)

    base_cfg = load_rule_engine_config(base_path)
    lexicon_path = Path(base_cfg.lexicon_path)
    lexicon_sha = sha256_file(lexicon_path)

    _quiet_loggers(True)
    started = time.perf_counter()

    baseline = evaluate(base_cfg, texts, y_true, args.objective, args.grid_size)
    LOGGER.info(
        "[27] baseline macro_f1=%.4f (HATE %.4f, OFFENSIVE %.4f, CLEAN %.4f) thresholds=%s",
        baseline["macro_f1"],
        baseline["HATE_f1"],
        baseline["OFFENSIVE_f1"],
        baseline["CLEAN_f1"],
        baseline["thresholds"],
    )

    best_values = snapshot(base_cfg)
    best = dict(baseline)
    trials: List[Dict[str, Any]] = []
    changes: Dict[str, List[Any]] = {}
    trial_index = 0
    rounds_run = 0

    for round_index in range(1, int(args.rounds) + 1):
        rounds_run = round_index
        improved = False
        for key, values in PARAM_GRID.items():
            for value in values:
                if value == best_values[key]:
                    continue
                candidate_values = dict(best_values)
                candidate_values[key] = value
                cfg = build_config(base_path, candidate_values)
                t0 = time.perf_counter()
                result = evaluate(cfg, texts, y_true, args.objective, args.grid_size)
                elapsed = time.perf_counter() - t0
                trial_index += 1
                gain = result["macro_f1"] - best["macro_f1"]
                accepted = gain > args.min_gain
                trial: Dict[str, Any] = {
                    "round": round_index,
                    "trial": trial_index,
                    "param": key,
                    "value": value,
                    "macro_f1": round(result["macro_f1"], 6),
                    "accuracy": round(result["accuracy"], 6),
                    "HATE_f1": round(result["HATE_f1"], 6),
                    "OFFENSIVE_f1": round(result["OFFENSIVE_f1"], 6),
                    "CLEAN_f1": round(result["CLEAN_f1"], 6),
                    "th_HATE": result["thresholds"][0],
                    "th_OFFENSIVE": result["thresholds"][1],
                    "th_CLEAN": result["thresholds"][2],
                    "accepted": accepted,
                    "seconds": round(elapsed, 2),
                }
                trials.append(trial)
                if accepted:
                    previous = best_values[key]
                    best_values[key] = value
                    best = dict(result)
                    changes[key] = [previous, value]
                    improved = True
                    LOGGER.info(
                        "[27] r%d %-28s %s -> %s  macro_f1 %.4f -> %.4f (+%.4f)",
                        round_index,
                        key,
                        previous,
                        value,
                        trial["macro_f1"] - gain,
                        trial["macro_f1"],
                        gain,
                    )
                elif trial_index % 10 == 0:
                    LOGGER.info(
                        "[27] r%d trial %d %s=%s macro_f1=%.4f (best %.4f, %s)",
                        round_index,
                        trial_index,
                        key,
                        value,
                        trial["macro_f1"],
                        best["macro_f1"],
                        "kept" if accepted else "skip",
                    )
                write_trials(out_dir / "trials.csv", trials)  # crash-safe progress
        if not improved:
            LOGGER.info("[27] round %d found no improvement; stopping", round_index)
            break

    elapsed_total = time.perf_counter() - started
    _quiet_loggers(False)

    best_cfg = build_config(base_path, best_values)
    validate(best_cfg)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    report = {
        "generated_by": "python scripts/27_tune_rule_weights.py",
        "generated_at": generated_at,
        "protocol": (
            "plan sections 39/44: coordinate descent over scoring.* on the validation split only; "
            "thresholds re-optimised per candidate (objective=macro_f1); test never loaded; "
            f"a change is accepted only when macro-F1 improves by more than {args.min_gain}"
        ),
        "base_config": str(base_path),
        "split": "validation",
        "n_rows": len(texts),
        "objective": args.objective,
        "grid_size": args.grid_size,
        "rounds_requested": int(args.rounds),
        "rounds_run": rounds_run,
        "n_trials": len(trials),
        "wall_seconds": round(elapsed_total, 1),
        "lexicon": {
            "path": str(lexicon_path),
            "sha256": lexicon_sha,
            "n_entries": best["n_lexicon_entries"],
        },
        "baseline": baseline,
        "best": best,
        "changed": changes,
        "best_scoring": best_cfg.to_dict()["scoring"],
        "best_thresholds": {ID2LABEL[i]: best["thresholds"][i] for i in range(NUM_LABELS)},
        "note": (
            "Both baseline and best numbers are validation numbers with validation-optimised "
            "thresholds; they are optimistic by construction. Freeze thresholds with script 17 "
            "and score the test split once before any claim (plan section 44)."
        ),
    }
    (out_dir / "tuning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_trials(out_dir / "trials.csv", trials)

    LOGGER.info(
        "[27] %d trials in %.1fs | validation macro_f1 %.4f -> %.4f",
        len(trials),
        elapsed_total,
        baseline["macro_f1"],
        best["macro_f1"],
    )
    for key, (old, new) in changes.items():
        LOGGER.info("[27]   %s: %s -> %s", key, old, new)
    LOGGER.info("[27] artifacts -> %s", out_dir)

    if args.no_write_config:
        LOGGER.info("[27] --no-write-config: %s left untouched", args.best_config)
        return 0

    config_path = Path(args.best_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        render_config(
            best_cfg,
            base_config=str(base_path),
            lexicon_sha=lexicon_sha,
            n_rows=len(texts),
            baseline=baseline,
            best=best,
            changes=changes,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )
    LOGGER.info("[27] winning weights -> %s", config_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
