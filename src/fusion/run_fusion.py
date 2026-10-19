"""Hybrid Rule-Engine + AI fusion experiment (Rule_Engine_Plan.md sections 40-44).

    python -m src.fusion.run_fusion --model sensitiveai-vi-customdlr2stage
    python -m src.fusion.run_fusion                 # every discovered model
    python -m src.fusion.run_fusion --model sensitiveai-vi --force-recompute

What it does, per AI model
--------------------------
1. reads the frozen per-split rule-engine scores from ``<rule_eval_dir>``
   (``rule_scores.csv`` for validation, ``test/rule_scores.csv`` for test);
2. reads the AI test probabilities from ``evaluation/per_model/<model>`` and, if
   missing, computes the AI **validation** probabilities with the trained
   checkpoint (the evaluator only saves test predictions, but fusion needs the
   validation probabilities to tune ``alpha`` without touching test);
3. aligns both sources by row ``index`` and asserts the labels agree;
4. picks ``alpha`` (and, optionally, label-specific thresholds) on validation,
   fits the meta-classifier on validation, freezes everything, then scores the
   test split **once**;
5. writes ``metrics.json`` / ``fusion_report.md`` / ``fusion_predictions.csv``
   under ``<out_dir>/<model>/`` and a cross-model summary under ``<out_dir>/``.

The test split is never used to choose ``alpha``, thresholds or meta-classifier
weights (plan section 44).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..common.config import dump_json, load_yaml
from ..common.constants import EVALUATION_DIR, ID2LABEL, LABEL2ID, LABELS, NUM_LABELS
from ..common.folder_docs import ensure_explain
from ..common.logging_utils import add_file_handler, get_logger
from ..common.paths import (
    ensure_dir,
    find_best_report_dir,
    per_model_eval_dir,
    project_path,
    project_relative,
)
from ..common.seeding import seed_everything
from ..data.dataset import build_datasets, make_dataloader
from ..evaluation.metrics import apply_decision_rule, compute_metrics, run_inference
from ..evaluation.run_evaluation import (
    PRECISION_TO_DTYPE,
    load_model_for_eval,
    load_run_config,
    resolve_checkpoint,
)
from .weighted_fusion import (
    CLEAN_ID,
    HATE_ID,
    OFFENSIVE_ID,
    MetaFusion,
    fuse_scores,
    fuse_scores_per_channel,
    rule_pseudo_proba,
    score_alpha_grid,
    score_alpha_grid_per_channel,
    tune_thresholds_at_alpha,
    tune_thresholds_at_alphas,
)

LOGGER = get_logger("sensitiveai.fusion")

DEFAULT_CONFIG = Path("configs/hybrid_fusion.yaml")
PROB_COLUMNS = ("prob_HATE", "prob_OFFENSIVE", "prob_CLEAN")


# --------------------------------------------------------------------------- #
# data handling
# --------------------------------------------------------------------------- #
@dataclass
class AlignedSplit:
    """Rule scores and AI probabilities aligned on the row ``index``."""

    name: str
    index: np.ndarray
    y: np.ndarray
    p_ai: np.ndarray
    rule_hate: np.ndarray
    rule_offensive: np.ndarray
    rule_pred: Optional[np.ndarray]
    texts: List[str]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def rule_proba(self) -> np.ndarray:
        return rule_pseudo_proba(self.rule_hate, self.rule_offensive)


def read_ai_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required = {"index", "true_label", *PROB_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if "true_label_id" not in df.columns:
        df["true_label_id"] = df["true_label"].map(LABEL2ID).astype(int)
    return df


def load_rule_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Rule-engine scores not found: {path}\n"
            "Run 'python scripts/17_evaluate_rule_engine.py --split validation' and "
            "--split test for the configured rule engine first."
        )
    df = pd.read_csv(path, encoding="utf-8")
    required = {"index", "true_label", "rule_hate_score", "rule_offensive_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df


def align(ai_df: pd.DataFrame, rule_df: pd.DataFrame, name: str) -> AlignedSplit:
    merged = ai_df.merge(rule_df, on="index", suffixes=("_ai", "_rule"), how="inner")
    if len(merged) != len(ai_df) or len(merged) != len(rule_df):
        raise ValueError(
            f"{name}: AI has {len(ai_df)} rows, rule has {len(rule_df)}, merged {len(merged)} "
            "- the two sources do not cover the same sample indices."
        )
    disagreement = int((merged["true_label_ai"] != merged["true_label_rule"]).sum())
    if disagreement:
        raise ValueError(
            f"{name}: {disagreement} rows disagree on true_label between the AI and rule "
            "scores - the index order is not aligned, refusing to fuse."
        )
    merged = merged.sort_values("index").reset_index(drop=True)
    y = merged["true_label_id"].astype(int).to_numpy()
    rule_pred = None
    for column in ("pred_label_rule", "pred_label"):
        if column in merged.columns:
            rule_pred = merged[column].map(LABEL2ID).astype(int).to_numpy()
            break
    return AlignedSplit(
        name=name,
        index=merged["index"].to_numpy(),
        y=y,
        p_ai=merged[list(PROB_COLUMNS)].to_numpy(dtype=float),
        rule_hate=merged["rule_hate_score"].to_numpy(dtype=float),
        rule_offensive=merged["rule_offensive_score"].to_numpy(dtype=float),
        rule_pred=rule_pred,
        texts=merged["text"].astype(str).tolist() if "text" in merged.columns else [""] * len(merged),
    )


def write_probability_csv(path: Path, texts, y_true, y_proba, indices) -> Path:
    data = {
        "index": list(indices),
        "text": list(texts),
        "true_label_id": [int(i) for i in y_true],
        "true_label": [ID2LABEL[int(i)] for i in y_true],
    }
    for i, name in enumerate(LABELS):
        data[f"prob_{name}"] = np.asarray(y_proba)[:, i]
    ensure_dir(path.parent)
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8")
    return path


def compute_split_probabilities(
    cfg: Dict[str, object],
    checkpoint: Path,
    split: str,
    device: torch.device,
) -> AlignedSplit:
    """Run the trained checkpoint over ``split`` and return its probabilities."""
    from transformers import AutoTokenizer

    model = load_model_for_eval(cfg, checkpoint).to(device).eval()

    dataset_cfg = cfg.get("dataset", {}) or {}
    splits_dir = project_path(dataset_cfg.get("splits_dir", "data/splits"))
    tokenizer_dir = checkpoint if (checkpoint / "tokenizer_config.json").is_file() else None
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir or cfg["base_model"]), use_fast=True
    )

    bundle = build_datasets(
        tokenizer=tokenizer,
        splits_dir=splits_dir,
        max_length=int((cfg.get("tokenizer", {}) or {}).get("max_length", 256)),
        seed=int(cfg.get("seed", 42)),
        text_column=str(dataset_cfg.get("text_column", "text")),
        label_column=str(dataset_cfg.get("label_column", "label")),
        splits=(split,),
    )
    dataset = {"validation": bundle.validation, "test": bundle.test}[split]
    raw = bundle.raw[split]
    loader = make_dataloader(
        dataset,
        int((cfg.get("training", {}) or {}).get("eval_batch_size", 32)),
        shuffle=False,
        num_workers=0,
    )
    precision = str((cfg.get("training", {}) or {}).get("precision", "fp32")).lower()
    outputs = run_inference(model, loader, device, amp_dtype=PRECISION_TO_DTYPE.get(precision))
    return AlignedSplit(
        name=split,
        index=np.asarray(dataset.indices),
        y=outputs["y_true"].astype(int),
        p_ai=outputs["y_proba"],
        rule_hate=np.full(len(dataset), np.nan),
        rule_offensive=np.full(len(dataset), np.nan),
        rule_pred=None,
        texts=raw[str(dataset_cfg.get("text_column", "text"))].astype(str).tolist(),
    )


def resolve_ai_source(
    model: str, reports_root: Path
) -> tuple[Path, Path, Dict[str, object], str]:
    """Pin the AI source to the *same run* that produced ``test_predictions.csv``.

    ``find_best_report_dir`` follows ``run_pointer.txt``, which may have moved on
    to a newer run *after* the model was evaluated.  Mixing a validation pass from
    one checkpoint with test predictions from another would make the fusion
    numbers meaningless, so the run recorded in ``per_model/<model>/metrics.json``
    wins; ``find_best_report_dir`` is only the fallback when that record is absent.
    """
    mirror = per_model_eval_dir(model)
    ref_metrics = mirror / "metrics.json"
    if ref_metrics.is_file():
        ref = json.loads(ref_metrics.read_text(encoding="utf-8"))
        checkpoint_rel = ref.get("checkpoint")
        if checkpoint_rel:
            checkpoint = project_path(checkpoint_rel)
            if checkpoint.is_dir():
                cfg_path = mirror / "config.yaml"
                cfg = load_yaml(cfg_path) if cfg_path.is_file() else load_run_config(mirror)
                cfg["_report_dir"] = str(mirror)
                return mirror, checkpoint, cfg, str(ref.get("run_id", "unknown"))
            LOGGER.warning(
                "[fusion] %s: checkpoint %s from metrics.json not found - "
                "falling back to find_best_report_dir",
                model, project_relative(checkpoint),
            )

    report_path = find_best_report_dir(model, str(reports_root))
    checkpoint = resolve_checkpoint(report_path, None)
    return report_path, checkpoint, load_run_config(report_path), report_path.name


def ensure_ai_predictions(
    model: str,
    cfg: Dict[str, object],
    checkpoint: Path,
    device: torch.device,
    force: bool,
) -> Dict[str, Path]:
    """Guarantee ``validation_predictions.csv`` and ``test_predictions.csv`` exist."""
    mirror = per_model_eval_dir(model)
    ensure_dir(mirror)
    paths = {
        "validation": mirror / "validation_predictions.csv",
        "test": mirror / "test_predictions.csv",
    }
    for split in ("validation", "test"):
        path = paths[split]
        if path.is_file() and not force:
            LOGGER.info("[fusion] reusing cached %s predictions -> %s", split, path)
            continue
        LOGGER.info("[fusion] computing %s probabilities for %s from %s", split, model, checkpoint)
        scored = compute_split_probabilities(cfg, checkpoint, split, device)
        write_probability_csv(path, scored.texts, scored.y, scored.p_ai, scored.index)
        LOGGER.info("[fusion] wrote %s", path)
    return paths


# --------------------------------------------------------------------------- #
# fusion for one model
# --------------------------------------------------------------------------- #
def _metrics_for(y_true, y_pred, scores) -> Dict[str, object]:
    return compute_metrics(y_true, y_pred, scores)


def run_model_fusion(
    model: str,
    fusion_cfg: Dict[str, object],
    device: torch.device,
    force: bool,
    out_root: Path,
) -> Dict[str, object]:
    reports_root = project_path(fusion_cfg["reports_dir"])
    report_path, checkpoint, cfg, run_id = resolve_ai_source(model, reports_root)
    LOGGER.info("=" * 78)
    LOGGER.info("[fusion] %s (%s) @ %s", model, run_id, report_path)
    LOGGER.info("[fusion] checkpoint %s", project_relative(checkpoint))
    LOGGER.info("=" * 78)

    pred_paths = ensure_ai_predictions(model, cfg, checkpoint, device, force)

    rule_eval_dir = project_path(fusion_cfg["rule_eval_dir"])
    rule_val_path = rule_eval_dir / "rule_scores.csv"
    rule_test_path = rule_eval_dir / "test" / "rule_scores.csv"

    val = align(read_ai_predictions(pred_paths["validation"]), load_rule_scores(rule_val_path), "validation")
    test = align(read_ai_predictions(pred_paths["test"]), load_rule_scores(rule_test_path), "test")
    LOGGER.info("[fusion] aligned validation n=%d, test n=%d", len(val), len(test))

    alphas = [float(a) for a in fusion_cfg.get("alpha_grid", [0.5])]
    best_alpha, grid = score_alpha_grid(val.y, val.p_ai, val.rule_hate, val.rule_offensive, alphas)
    LOGGER.info("[fusion] best alpha on validation = %.4f", best_alpha)

    th_cfg = fusion_cfg.get("thresholds", {}) or {}
    thresholds_payload: Optional[Dict[str, object]] = None
    thresholds: Optional[List[float]] = None
    if bool(th_cfg.get("enabled", True)):
        thresholds_payload = tune_thresholds_at_alpha(
            val.y,
            val.p_ai,
            val.rule_hate,
            val.rule_offensive,
            best_alpha,
            objective=str(th_cfg.get("objective", "macro_f1")),
            grid_size=int(th_cfg.get("grid_size", 101)),
            coordinate_rounds=int(th_cfg.get("coordinate_rounds", 3)),
        )
        thresholds = list(thresholds_payload["threshold_list"])
        LOGGER.info("[fusion] frozen thresholds @alpha=%.4f -> %s", best_alpha, thresholds)

    meta_cfg = fusion_cfg.get("meta_classifier", {}) or {}
    meta: Optional[MetaFusion] = None
    if bool(meta_cfg.get("enabled", True)):
        meta = MetaFusion(
            C=float(meta_cfg.get("C", 1.0)),
            max_iter=int(meta_cfg.get("max_iter", 1000)),
            class_weight=str(meta_cfg.get("class_weight", "balanced")),
            seed=int(cfg.get("seed", 42)),
        ).fit(val.p_ai, val.rule_hate, val.rule_offensive, val.y)

    # ---- per-channel alpha (rule weight freed per label) ---------------- #
    pc_cfg = fusion_cfg.get("per_channel", {}) or {}
    per_channel_alphas: Optional[List[float]] = None
    per_channel_thresholds: Optional[List[float]] = None
    per_channel_payload: Optional[Dict[str, object]] = None
    per_channel_trace: Optional[List[Dict[str, object]]] = None
    if bool(pc_cfg.get("enabled", True)):
        per_channel_alphas, per_channel_trace = score_alpha_grid_per_channel(
            val.y,
            val.p_ai,
            val.rule_hate,
            val.rule_offensive,
            alphas,
            rounds=int(pc_cfg.get("rounds", 3)),
        )
        LOGGER.info("[fusion] per-channel alphas on validation = %s", per_channel_alphas)
        if bool(th_cfg.get("enabled", True)):
            per_channel_payload = tune_thresholds_at_alphas(
                val.y,
                val.p_ai,
                val.rule_hate,
                val.rule_offensive,
                per_channel_alphas,
                objective=str(th_cfg.get("objective", "macro_f1")),
                grid_size=int(th_cfg.get("grid_size", 101)),
                coordinate_rounds=int(th_cfg.get("coordinate_rounds", 3)),
            )
            per_channel_thresholds = list(per_channel_payload["threshold_list"])
            LOGGER.info(
                "[fusion] frozen per-channel thresholds @alphas=%s -> %s",
                per_channel_alphas, per_channel_thresholds,
            )

    # ---- test systems (test is scored, never tuned on) ------------------ #
    val_scores = fuse_scores(val.p_ai, val.rule_hate, val.rule_offensive, best_alpha)
    test_scores = fuse_scores(test.p_ai, test.rule_hate, test.rule_offensive, best_alpha)

    systems: Dict[str, Dict[str, object]] = {}
    systems["ai_only_argmax"] = _metrics_for(test.y, test.p_ai.argmax(axis=1), test.p_ai)
    systems["rule_only_argmax"] = _metrics_for(
        test.y, test.rule_proba().argmax(axis=1), test.rule_proba()
    )
    if test.rule_pred is not None:
        systems["rule_only_frozen_thresholds"] = _metrics_for(
            test.y, test.rule_pred, test.rule_proba()
        )
    systems["fusion_argmax"] = _metrics_for(test.y, test_scores.argmax(axis=1), test_scores)
    if thresholds is not None:
        systems["fusion_thresholds"] = _metrics_for(
            test.y, apply_decision_rule(test_scores, "thresholds", thresholds), test_scores
        )
    if meta is not None:
        meta_proba = meta.predict_proba(test.p_ai, test.rule_hate, test.rule_offensive)
        systems["fusion_meta_classifier"] = _metrics_for(
            test.y, meta_proba.argmax(axis=1), meta_proba
        )
    pc_scores: Optional[np.ndarray] = None
    if per_channel_alphas is not None:
        pc_scores = fuse_scores_per_channel(
            test.p_ai, test.rule_hate, test.rule_offensive, per_channel_alphas
        )
        systems["fusion_per_channel_argmax"] = _metrics_for(
            test.y, pc_scores.argmax(axis=1), pc_scores
        )
        if per_channel_thresholds is not None:
            systems["fusion_per_channel_thresholds"] = _metrics_for(
                test.y,
                apply_decision_rule(pc_scores, "thresholds", per_channel_thresholds),
                pc_scores,
            )

    # validation view of the fused decision (for the report only)
    validation_view = {
        "best_alpha": round(float(best_alpha), 4),
        "alpha_grid": grid,
        "fusion_argmax": {
            k: v for k, v in _metrics_for(val.y, val_scores.argmax(axis=1), val_scores).items()
            if k != "confusion_matrix"
        },
        "thresholds": thresholds_payload,
    }
    if meta is not None:
        meta_val = meta.predict_proba(val.p_ai, val.rule_hate, val.rule_offensive)
        validation_view["fusion_meta_classifier"] = {
            k: v for k, v in _metrics_for(val.y, meta_val.argmax(axis=1), meta_val).items()
            if k != "confusion_matrix"
        }
    if per_channel_alphas is not None:
        pc_val_scores = fuse_scores_per_channel(
            val.p_ai, val.rule_hate, val.rule_offensive, per_channel_alphas
        )
        validation_view["per_channel"] = {
            "alphas": per_channel_alphas,
            "trace": per_channel_trace,
            "fusion_argmax_macro_f1": float(
                _metrics_for(val.y, pc_val_scores.argmax(axis=1), pc_val_scores)["macro_f1"]
            ),
            "thresholds": per_channel_payload,
        }

    # reference AI-only number as reported by the evaluator (frozen thresholds)
    ai_reference: Dict[str, object] = {}
    ref_path = per_model_eval_dir(model) / "metrics.json"
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ai_reference = {
            "decision_rule": ref.get("decision_rule"),
            "accuracy": ref.get("accuracy"),
            "macro_f1": ref.get("macro_f1"),
            "HATE_f1": ref.get("HATE_f1"),
            "OFFENSIVE_f1": ref.get("OFFENSIVE_f1"),
            "CLEAN_f1": ref.get("CLEAN_f1"),
        }

    out_dir = ensure_dir(out_root / model)
    ensure_explain(out_dir)

    payload = {
        "model": model,
        "run_id": run_id,
        "checkpoint": project_relative(checkpoint),
        "rule_eval_dir": project_relative(rule_eval_dir),
        "splits": {"validation": len(val), "test": len(test)},
        "protocol": (
            "alpha, thresholds and the meta-classifier are selected/fitted on the "
            "validation split and frozen before the single test pass; the test "
            "split is never used for tuning (Rule_Engine_Plan.md section 44)."
        ),
        "alpha": round(float(best_alpha), 4),
        "thresholds": None if thresholds is None else {ID2LABEL[i]: thresholds[i] for i in range(NUM_LABELS)},
        "per_channel_alphas": None
        if per_channel_alphas is None
        else {ID2LABEL[i]: per_channel_alphas[i] for i in range(NUM_LABELS)},
        "per_channel_thresholds": None
        if per_channel_thresholds is None
        else {ID2LABEL[i]: per_channel_thresholds[i] for i in range(NUM_LABELS)},
        "meta_classifier_coefficients": None if meta is None else meta.coefficients(),
        "ai_only_reference": ai_reference,
        "validation": validation_view,
        "test": systems,
    }
    dump_json(payload, out_dir / "metrics.json")

    # ---- per-row predictions for the primary frozen fusion --------------- #
    # The primary variant is chosen on *validation* macro-F1 (never on test).
    candidates: List[tuple] = []
    if thresholds is not None:
        candidates.append(
            ("fusion_thresholds", float(thresholds_payload["objective_after"]), test_scores, thresholds)
        )
    if per_channel_thresholds is not None and pc_scores is not None:
        candidates.append(
            (
                "fusion_per_channel_thresholds",
                float(per_channel_payload["objective_after"]),
                pc_scores,
                per_channel_thresholds,
            )
        )
    if not candidates:
        candidates.append(("fusion_argmax", float("nan"), test_scores, None))
    primary_variant, _, primary, primary_thresholds = max(
        candidates, key=lambda item: (item[1] if item[1] == item[1] else -1.0)
    )
    payload["primary_variant"] = primary_variant
    dump_json(payload, out_dir / "metrics.json")
    primary_pred = (
        apply_decision_rule(primary, "thresholds", primary_thresholds)
        if primary_thresholds is not None
        else primary.argmax(axis=1)
    )
    frame = pd.DataFrame(
        {
            "index": test.index,
            "text": test.texts,
            "true_label": [ID2LABEL[int(i)] for i in test.y],
            "ai_pred": [ID2LABEL[int(i)] for i in test.p_ai.argmax(axis=1)],
            "rule_pred": [ID2LABEL[int(i)] for i in test.rule_proba().argmax(axis=1)],
            "fused_pred": [ID2LABEL[int(i)] for i in primary_pred],
            "prob_HATE": test.p_ai[:, HATE_ID],
            "prob_OFFENSIVE": test.p_ai[:, OFFENSIVE_ID],
            "prob_CLEAN": test.p_ai[:, CLEAN_ID],
            "rule_hate_score": test.rule_hate,
            "rule_offensive_score": test.rule_offensive,
            "fused_hate": primary[:, HATE_ID],
            "fused_offensive": primary[:, OFFENSIVE_ID],
            "fused_clean": primary[:, CLEAN_ID],
        }
    )
    frame.to_csv(out_dir / "fusion_predictions.csv", index=False, encoding="utf-8")

    _write_model_report(out_dir, payload)
    return payload


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-" if value is None else str(value)


def _write_model_report(out_dir: Path, payload: Dict[str, object]) -> Path:
    test = payload["test"]
    lines = [
        f"# Fusion report - {payload['model']} ({payload['run_id']})",
        "",
        f"- Rule scores: `{payload['rule_eval_dir']}`",
        f"- mix weight alpha (AI) = **{payload['alpha']}**  (0 = rule only, 1 = AI only)",
        f"- per-channel alphas (AI weight per label) = {payload.get('per_channel_alphas')}",
        f"- frozen thresholds = {payload['thresholds']}",
        f"- primary variant (chosen on validation) = `{payload.get('primary_variant')}`",
        f"- validation n={payload['splits']['validation']}, test n={payload['splits']['test']}",
        "",
        "Thresholds / alpha / meta-classifier are tuned on validation only; test is scored once.",
        "",
        "## Test metrics",
        "",
        "| system | accuracy | macro-F1 | HATE F1 | OFFENSIVE F1 | CLEAN F1 |",
        "|---|---|---|---|---|---|",
    ]
    order = [
        "ai_only_argmax",
        "rule_only_frozen_thresholds",
        "rule_only_argmax",
        "fusion_argmax",
        "fusion_thresholds",
        "fusion_per_channel_thresholds",
        "fusion_meta_classifier",
    ]
    for name in order:
        metrics = test.get(name)
        if not metrics:
            continue
        lines.append(
            f"| {name} | {_fmt(metrics['accuracy'])} | **{_fmt(metrics['macro_f1'])}** | "
            f"{_fmt(metrics['HATE_f1'])} | {_fmt(metrics['OFFENSIVE_f1'])} | {_fmt(metrics['CLEAN_f1'])} |"
        )
    ref = payload.get("ai_only_reference") or {}
    if ref:
        lines += [
            "",
            f"AI-only as reported by the evaluator (rule=`{ref.get('decision_rule')}`): "
            f"macro-F1 {_fmt(ref.get('macro_f1'))}, HATE {_fmt(ref.get('HATE_f1'))}, "
            f"OFFENSIVE {_fmt(ref.get('OFFENSIVE_f1'))}.",
        ]
    lines.append("")
    path = out_dir / "fusion_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def discover_models(reports_root: Path, per_model_root: Path) -> List[str]:
    names: List[str] = []
    if not reports_root.is_dir():
        return names
    for child in sorted(reports_root.iterdir()):
        if not child.is_dir():
            continue
        if any((child / run / "best_model").is_dir() for run in child.glob("run_*")):
            names.append(child.name)
    return names


def write_summary(out_root: Path, payloads: Sequence[Dict[str, object]]) -> Path:
    header = (
        "| model | alpha | AI macro-F1 | rule macro-F1 | fusion(argmax) | "
        "fusion(thresholds) | fusion(per-channel) | fusion(meta) | primary (val-chosen) | Δ vs AI |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [
        "# Hybrid fusion - Rule Engine + AI",
        "",
        "Rule scores from the frozen rule-engine evaluation; alpha / thresholds / meta "
        "tuned on validation, test scored once (Rule_Engine_Plan.md sections 40-44).",
        "`primary` is the variant chosen on validation macro-F1; `Δ vs AI` is "
        "primary minus AI-only (argmax).",
        "",
        header,
        sep,
    ]
    rows = []
    for payload in payloads:
        test = payload["test"]

        def macro(name: str) -> object:
            return (test.get(name) or {}).get("macro_f1")

        rule_key = (
            "rule_only_frozen_thresholds"
            if "rule_only_frozen_thresholds" in test
            else "rule_only_argmax"
        )
        primary = payload.get("primary_variant")
        ai_macro = test.get("ai_only_argmax", {}).get("macro_f1")
        primary_macro = (test.get(primary) or {}).get("macro_f1") if primary else None
        delta = (
            float(primary_macro) - float(ai_macro)
            if primary_macro is not None and ai_macro is not None
            else None
        )
        delta_str = "-" if delta is None else f"{delta:+.4f}"
        rows.append(
            f"| {payload['model']} | {payload['alpha']} | {_fmt(macro('ai_only_argmax'))} | "
            f"{_fmt(macro(rule_key))} | {_fmt(macro('fusion_argmax'))} | "
            f"{_fmt(macro('fusion_thresholds'))} | {_fmt(macro('fusion_per_channel_thresholds'))} | "
            f"{_fmt(macro('fusion_meta_classifier'))} | "
            f"{_fmt(primary_macro)} (`{primary}`) | {delta_str} |"
        )
    lines += rows
    lines.append("")
    path = out_root / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.fusion.run_fusion",
        description="Fuse the lexical rule engine with a trained AI model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="fusion YAML config")
    p.add_argument("--model", action="append", default=None, help="AI model name (repeatable)")
    p.add_argument("--rule-eval-dir", default=None, help="override rule_eval_dir from the config")
    p.add_argument("--out-dir", default=None, help="override out_dir from the config")
    p.add_argument("--reports-dir", default=None, help="override reports_dir from the config")
    p.add_argument("--device", default=None, help="force 'cuda' or 'cpu'")
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="recompute the AI predictions instead of reusing the cached CSVs",
    )
    return p


def run(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    for key, value in (
        ("rule_eval_dir", args.rule_eval_dir),
        ("out_dir", args.out_dir),
        ("reports_dir", args.reports_dir),
    ):
        if value:
            cfg[key] = value

    out_root = ensure_dir(project_path(cfg["out_dir"]))
    add_file_handler(LOGGER, out_root / "fusion.log")
    ensure_explain(out_root)

    seed_everything(42, deterministic=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("[fusion] device=%s", device)

    models = args.model or list(cfg.get("models") or [])
    if not models:
        models = discover_models(
            project_path(cfg["reports_dir"]), project_path(EVALUATION_DIR / "per_model")
        )
    if not models:
        LOGGER.error("[fusion] no models with a trained best_model found - train first")
        return 1
    LOGGER.info("[fusion] models: %s", ", ".join(models))

    payloads: List[Dict[str, object]] = []
    for model in models:
        try:
            payloads.append(
                run_model_fusion(model, cfg, device, bool(args.force_recompute), out_root)
            )
        except Exception as exc:  # keep one failure from aborting the sweep
            LOGGER.error("[fusion] %s failed: %s", model, exc, exc_info=True)

    if payloads:
        summary = write_summary(out_root, payloads)
        LOGGER.info("[fusion] summary -> %s", summary)
        for payload in payloads:
            test = payload["test"]
            best = max(
                (s for s in test.values() if s),
                key=lambda m: m.get("macro_f1", -1),
                default={},
            )
            LOGGER.info(
                "[fusion] %-34s alpha=%.2f  best test macro-F1=%.4f",
                payload["model"], payload["alpha"], best.get("macro_f1", float("nan")),
            )
    return 0 if payloads else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
