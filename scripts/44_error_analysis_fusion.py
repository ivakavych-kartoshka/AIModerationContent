#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Error analysis for fusion (AI-only vs rule+AI).

Priority 1: find cases where AI-only is wrong but fusion(primary) correct,
and cases where AI-only correct but fusion wrong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.constants import ID2LABEL, LABELS
from src.common.logging_utils import get_logger
from src.common.paths import ensure_dir, project_path

LOGGER = get_logger("sensitiveai.scripts.44_error_analysis_fusion")

FUSION_DIR = project_path("outputs/evaluation/fusion")
RULE_DIR = project_path("outputs/evaluation/rule_engine")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Error analysis for fusion vs AI-only")
    parser.add_argument("--models", nargs="*", default=["sensitiveai-vi-custom"], help="Models to analyze")
    parser.add_argument("--fusion-dir", default=str(FUSION_DIR))
    parser.add_argument("--rule-dir", default=str(RULE_DIR))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    fusion_dir = Path(args.fusion_dir)
    rule_dir = Path(args.rule_dir)
    out_root = Path(args.out_dir) if args.out_dir else fusion_dir / "error_analysis"
    ensure_dir(out_root)

    rule_test_path = rule_dir / "test" / "rule_scores.csv"
    rule_df = pd.read_csv(rule_test_path, encoding="utf-8")
    rule_map = rule_df.set_index("index")

    label_to_id = {l: i for i, l in enumerate(LABELS)}

    for model in args.models:
        model_dir = fusion_dir / model
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.is_file():
            LOGGER.warning("[44] missing %s", metrics_path)
            continue
        metrics = load_json(metrics_path)
        primary = metrics.get("primary_variant", "unknown")
        LOGGER.info("[44] %s -> primary=%s", model, primary)

        pred_path = model_dir / "fusion_predictions.csv"
        if not pred_path.is_file():
            pred_path = model_dir / "predictions_test.csv"
        if not pred_path.is_file():
            LOGGER.warning("[44] missing %s", pred_path)
            continue
        df = pd.read_csv(pred_path, encoding="utf-8")
        if "index" not in df.columns:
            df = df.reset_index().rename(columns={"index": "index"})

        y_true_str = df["true_label"].astype(str)
        y_true = y_true_str.map(label_to_id).fillna(-1).astype(int).to_numpy()

        if "ai_pred" in df.columns:
            y_ai = df["ai_pred"].astype(str).map(label_to_id).fillna(-1).astype(int).to_numpy()
        elif all(c in df.columns for c in ["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]):
            p_ai = df[["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]].to_numpy(dtype=float)
            y_ai = p_ai.argmax(axis=1)
        elif all(c in df.columns for c in [f"prob_{l}" for l in LABELS]):
            p_ai = df[[f"prob_{l}" for l in LABELS]].to_numpy(dtype=float)
            y_ai = p_ai.argmax(axis=1)
        else:
            LOGGER.warning("[44] cannot find AI predictions for %s", model)
            continue

        if "fused_pred" in df.columns:
            y_fuse = df["fused_pred"].astype(str).map(label_to_id).fillna(-1).astype(int).to_numpy()
        elif all(c in df.columns for c in ["fused_hate", "fused_offensive", "fused_clean"]):
            p_fuse = df[["fused_hate", "fused_offensive", "fused_clean"]].to_numpy(dtype=float)
            y_fuse = p_fuse.argmax(axis=1)
        elif all(c in df.columns for c in [f"prob_{l}_{primary}" for l in LABELS]):
            p_fuse = df[[f"prob_{l}_{primary}" for l in LABELS]].to_numpy(dtype=float)
            y_fuse = p_fuse.argmax(axis=1)
        else:
            LOGGER.warning("[44] cannot find fusion predictions for %s", model)
            continue

        rows = []
        for i, idx in enumerate(df["index"]):
            r = rule_map.loc[idx] if idx in rule_map.index else None
            text = df["text"].iloc[i] if "text" in df.columns else ""
            ai_pred_str = df["ai_pred"].iloc[i] if "ai_pred" in df.columns else ""
            fused_pred_str = df["fused_pred"].iloc[i] if "fused_pred" in df.columns else ""
            rows.append({
                "index": int(idx),
                "text": text,
                "true_label": ID2LABEL[int(y_true[i])] if int(y_true[i]) >= 0 else y_true_str.iloc[i],
                "ai_pred": ID2LABEL[int(y_ai[i])] if int(y_ai[i]) >= 0 else str(ai_pred_str),
                "fuse_pred": ID2LABEL[int(y_fuse[i])] if int(y_fuse[i]) >= 0 else str(fused_pred_str),
                "ai_correct": bool(y_ai[i] == y_true[i] and y_true[i] >= 0),
                "fuse_correct": bool(y_fuse[i] == y_true[i] and y_true[i] >= 0),
                "rule_hate": float(r["rule_hate_score"]) if r is not None and "rule_hate_score" in r else (float(df["rule_hate_score"].iloc[i]) if "rule_hate_score" in df.columns else np.nan),
                "rule_offensive": float(r["rule_offensive_score"]) if r is not None and "rule_offensive_score" in r else (float(df["rule_offensive_score"].iloc[i]) if "rule_offensive_score" in df.columns else np.nan),
                "rule_clean": float(r["rule_clean_score"]) if r is not None and "rule_clean_score" in r else np.nan,
            })
        res = pd.DataFrame(rows)

        ai_wrong_fuse_right = res[(~res["ai_correct"]) & (res["fuse_correct"])].copy()
        ai_right_fuse_wrong = res[(res["ai_correct"]) & (~res["fuse_correct"])].copy()
        both_wrong = res[(~res["ai_correct"]) & (~res["fuse_correct"])].copy()
        both_right = res[(res["ai_correct"]) & (res["fuse_correct"])].copy()

        out_dir = out_root / model
        ensure_dir(out_dir)
        ai_wrong_fuse_right.to_csv(out_dir / "ai_wrong_fuse_right.csv", index=False, encoding="utf-8")
        ai_right_fuse_wrong.to_csv(out_dir / "ai_right_fuse_wrong.csv", index=False, encoding="utf-8")
        both_wrong.to_csv(out_dir / "both_wrong.csv", index=False, encoding="utf-8")
        both_right.to_csv(out_dir / "both_right.csv", index=False, encoding="utf-8")

        summary = {
            "model": model,
            "primary": primary,
            "n": int(len(res)),
            "ai_correct": int(res["ai_correct"].sum()),
            "fuse_correct": int(res["fuse_correct"].sum()),
            "ai_wrong_fuse_right": int(len(ai_wrong_fuse_right)),
            "ai_right_fuse_wrong": int(len(ai_right_fuse_wrong)),
            "net_gain": int(len(ai_wrong_fuse_right) - len(ai_right_fuse_wrong)),
            "both_wrong": int(len(both_wrong)),
            "both_right": int(len(both_right)),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        LOGGER.info("[44] %s: net_gain=%d (fixed=%d, broken=%d)", model, summary["net_gain"], summary["ai_wrong_fuse_right"], summary["ai_right_fuse_wrong"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
