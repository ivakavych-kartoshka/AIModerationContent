#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Blind-spot analysis: 504 mẫu bị cả 6 AI model bỏ sót.

Mục tiêu: Kiểm tra xem Rule Engine tuned có phủ được phần nào
trong 504 mẫu mà cả 6 mô hình AI đều miss không, và phân tích
sự bất đồng giữa AI vs RE trên toàn test set.

Dữ liệu đầu vào:

  • outputs/evaluation/analysis/supplementary_summary.json
    → có "error_overlap" với danh sách 504 mẫu all_miss (bị cả 6 miss)
  • outputs/evaluation/fusion/sensitiveai-vi-custom/fusion_predictions.csv
    → có AI predictions + rule scores
  • outputs/evaluation/rule_engine_tuned/test/rule_scores.csv
    → rule engine scores (tuned variant)

Đầu ra:

  • outputs/evaluation/fusion/blind_spot_analysis_summary.json
    → tổng hợp kết quả phân tích
  • outputs/evaluation/fusion/blind_spot_504.csv
    → chi tiết từng mẫu trong 504 blind-spot
  • outputs/evaluation/fusion/blind_spot_disagreement.csv
    → phân tích AI vs RE disagreement patterns
  • outputs/evaluation/fusion/blind_spot_report.tex
    → LaTeX table tóm tắt

Chạy:

    python scripts/58_blind_spot_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
FUSION_DIR = EVAL_DIR / "fusion"
ANALYSIS_DIR = EVAL_DIR / "analysis"
RULE_DIR = EVAL_DIR / "rule_engine_tuned"

# --------------------------------------------------------------------------- #
# Label constants
# --------------------------------------------------------------------------- #
LABELS = ("HATE", "OFFENSIVE", "CLEAN")
LABEL2ID = {lbl: idx for idx, lbl in enumerate(LABELS)}
ID2LABEL = {idx: lbl for lbl, idx in LABEL2ID.items()}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blind-spot analysis: 504 mẫu bị cả 6 AI miss")
    parser.add_argument(
        "--supplementary",
        default=str(ANALYSIS_DIR / "supplementary_summary.json"),
        help="Supplementary summary JSON (contains error_overlap with all_miss indices)",
    )
    parser.add_argument(
        "--fusion-pred",
        default=str(FUSION_DIR / "sensitiveai-vi-custom" / "fusion_predictions.csv"),
        help="Fusion predictions CSV (has AI predictions + rule scores)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(FUSION_DIR),
        help="Directory for output files",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Label mapping (needed before loading any predictions)
    # ------------------------------------------------------------------ #
    lab_map = {"HATE": 0, "OFFENSIVE": 1, "CLEAN": 2}

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print("[58] Loading supplementary summary...")
    with open(args.supplementary, "r", encoding="utf-8") as f:
        supp = json.load(f)

    error_overlap = supp.get("error_overlap", {})
    # Try different keys for backward compatibility
    all_miss_indices = set(error_overlap.get("all_miss_indices", []))
    if not all_miss_indices:
        # Fallback: reconstruct from k_distribution
        k_dist = error_overlap.get("k_distribution", {})
        n_models = error_overlap.get("n_models", 6)
        if str(n_models) in k_dist and int(k_dist[str(n_models)]) > 0:
            print(f"[58] Reconstructing all_miss from k_distribution (k={n_models})")
            # We need to read the per-model predictions to find which rows are missed by all
            # For now, we'll compute from the fusion predictions
            all_miss_indices = None  # will compute below
        else:
            all_miss_indices = None
    
    if all_miss_indices is None:
        # Compute all_miss_indices from per-model predictions
        # Read predictions from all 6 models
        per_model_dir = EVAL_DIR / "per_model"
        model_names = [
            "sensitiveai-vi",
            "sensitiveai-vi-custom", 
            "sensitiveai-vi-customdlr2stage",
            "sensitiveai-vi-custom-curriculum",
            "sensitiveai-vi-customdlr2stage-ext",
            "sensitiveai-vi-stage2",
        ]
        
        print("[58] Loading predictions from all 6 models to find blind-spot...")
        all_preds = []
        y_true_all = None
        
        for i, model_name in enumerate(model_names):
            pred_path = per_model_dir / model_name / "test_predictions.csv"
            if not pred_path.is_file():
                print(f"  ⚠️  Missing {pred_path}")
                continue
            df_model = pd.read_csv(pred_path, encoding="utf-8")
            if y_true_all is None:
                y_true_all = df_model["true_label"].map(lab_map).to_numpy(dtype=int)
            preds = df_model["pred_label"].map(lab_map).to_numpy(dtype=int)
            all_preds.append(preds)
        
        if len(all_preds) == 0:
            print("⚠️  No model predictions found. Skipping blind-spot analysis.")
            return 0
        
        # Stack and find rows where all models are wrong
        all_preds_array = np.stack(all_preds, axis=0)  # shape: (n_models, n_items)
        all_wrong = np.all(all_preds_array != y_true_all[None, :], axis=0)
        all_miss_indices = set(np.where(all_wrong)[0].tolist())
        print(f"  Found {len(all_miss_indices)} all-miss samples from per-model predictions")
    
    if not all_miss_indices:
        print("⚠️  No all_miss_indices found in supplementary summary. Skipping blind-spot analysis.")
        return 0

    print(f"[58] Found {len(all_miss_indices)} all-miss (504-expected) indices")

    print("[58] Loading fusion predictions...")
    fusion_df = pd.read_csv(args.fusion_pred, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Extract truth and predictions
    # ------------------------------------------------------------------ #
    lab_map = {"HATE": 0, "OFFENSIVE": 1, "CLEAN": 2}
    y_true = fusion_df["true_label"].map(lab_map).to_numpy(dtype=int)

    # AI predictions
    y_ai = fusion_df["ai_pred"].map(lab_map).to_numpy(dtype=int)

    # Rule predictions
    y_rule = fusion_df["rule_pred"].map(lab_map).to_numpy(dtype=int)

    # Hybrid predictions
    y_hybrid = fusion_df["fused_pred"].map(lab_map).to_numpy(dtype=int)

    # ------------------------------------------------------------------ #
    # PART 1: Analyze 504 blind-spot
    # ------------------------------------------------------------------ #
    print("[58] Analyzing 504 blind-spot...")

    blind_spot_rows: List[Dict[str, Any]] = []
    blind_spot_stats = {
        "total_in_504": len(all_miss_indices),
        "rule_correct": 0,
        "rule_correct_by_class": {"HATE": 0, "OFFENSIVE": 0, "CLEAN": 0},
        "class_distribution": {"HATE": 0, "OFFENSIVE": 0, "CLEAN": 0},
    }

    for idx in sorted(all_miss_indices):
        if idx >= len(fusion_df):
            continue
        row = fusion_df.iloc[idx]
        true_lbl = row["true_label"]
        ai_pred = row["ai_pred"]
        rule_pred = row["rule_pred"]
        hybrid_pred = row["fused_pred"]

        rule_correct = bool(rule_pred == true_lbl)
        if rule_correct:
            blind_spot_stats["rule_correct"] += 1
            blind_spot_stats["rule_correct_by_class"][true_lbl] += 1
        blind_spot_stats["class_distribution"][true_lbl] += 1

        blind_spot_rows.append({
            "index": int(idx),
            "text": row["text"][:80] if len(row["text"]) > 80 else row["text"],
            "true_label": true_lbl,
            "ai_pred": ai_pred,
            "rule_pred": rule_pred,
            "hybrid_pred": hybrid_pred,
            "rule_correct": int(rule_correct),
            "prob_HATE": float(row["prob_HATE"]),
            "prob_OFFENSIVE": float(row["prob_OFFENSIVE"]),
            "prob_CLEAN": float(row["prob_CLEAN"]),
            "rule_hate_score": float(row["rule_hate_score"]),
            "rule_offensive_score": float(row["rule_offensive_score"]),
        })

    blind_spot_df = pd.DataFrame(blind_spot_rows)
    blind_spot_path = out_dir / "blind_spot_504.csv"
    blind_spot_df.to_csv(blind_spot_path, index=False, encoding="utf-8")
    print(f"  Wrote {len(blind_spot_df)} rows to {blind_spot_path}")

    # ------------------------------------------------------------------ #
    # PART 2: AI vs Rule disagreement analysis (full test set)
    # ------------------------------------------------------------------ #
    print("[58] Analyzing AI vs Rule disagreement patterns (full test set)...")

    ai_correct = y_ai == y_true
    rule_correct = y_rule == y_true

    # Four patterns
    ai_right_rule_wrong = ai_correct & ~rule_correct
    ai_wrong_rule_right = ~ai_correct & rule_correct
    both_wrong = ~ai_correct & ~rule_correct
    both_right = ai_correct & rule_correct

    disagreement_rows: List[Dict[str, Any]] = []

    for idx in range(len(fusion_df)):
        row = fusion_df.iloc[idx]
        true_lbl = row["true_label"]
        ai_p = row["ai_pred"]
        rule_p = row["rule_pred"]

        pattern = None
        if ai_correct[idx] and rule_correct[idx]:
            pattern = "both_right"
        elif ai_correct[idx] and not rule_correct[idx]:
            pattern = "ai_right_rule_wrong"
        elif not ai_correct[idx] and rule_correct[idx]:
            pattern = "ai_wrong_rule_right"
        else:
            pattern = "both_wrong"

        in_504 = int(idx) in all_miss_indices
        disagreement_rows.append({
            "index": int(idx),
            "true_label": true_lbl,
            "ai_pred": ai_p,
            "rule_pred": rule_p,
            "pattern": pattern,
            "in_504_blind_spot": in_504,
        })

    disagreement_df = pd.DataFrame(disagreement_rows)
    disagreement_path = out_dir / "blind_spot_disagreement.csv"
    disagreement_df.to_csv(disagreement_path, index=False, encoding="utf-8")
    print(f"  Wrote {len(disagreement_df)} rows to {disagreement_path}")

    # Summarize disagreement patterns
    disagreement_summary = {}
    for pattern in ["both_right", "ai_right_rule_wrong", "ai_wrong_rule_right", "both_wrong"]:
        subset = disagreement_df[disagreement_df["pattern"] == pattern]
        disagreement_summary[pattern] = {
            "total": int(len(subset)),
            "in_504": int(subset["in_504_blind_spot"].sum()),
            "by_class": subset["true_label"].value_counts().to_dict(),
        }

    # ------------------------------------------------------------------ #
    # PART 3: Write JSON summary
    # ------------------------------------------------------------------ #
    summary = {
        "blind_spot_504": blind_spot_stats,
        "rule_engine_coverage_in_504": {
            "correct": blind_spot_stats["rule_correct"],
            "percentage": float(
                100.0 * blind_spot_stats["rule_correct"] / blind_spot_stats["total_in_504"]
                if blind_spot_stats["total_in_504"] > 0 else 0.0
            ),
        },
        "disagreement_patterns": disagreement_summary,
    }

    summary_path = out_dir / "blind_spot_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Wrote summary to {summary_path}")

    # ------------------------------------------------------------------ #
    # PART 4: Write LaTeX table
    # ------------------------------------------------------------------ #
    tex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Blind-spot analysis: 504 comments missed by all 6 AI models. Rule engine coverage and disagreement patterns.}",
        r"\label{tab:blind-spot-504}",
        r"\footnotesize",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Pattern & Count & Pct. in 504 \\",
        r"\midrule",
        f"Both AI and RE correct & {disagreement_summary['both_right']['total']} & {disagreement_summary['both_right']['in_504']} \\\\",
        f"AI correct, RE wrong & {disagreement_summary['ai_right_rule_wrong']['total']} & {disagreement_summary['ai_right_rule_wrong']['in_504']} \\\\",
        f"AI wrong, RE correct & {disagreement_summary['ai_wrong_rule_right']['total']} & {disagreement_summary['ai_wrong_rule_right']['in_504']} \\\\",
        f"Both wrong (hard cases) & {disagreement_summary['both_wrong']['total']} & {disagreement_summary['both_wrong']['in_504']} \\\\",
        r"\midrule",
        f"Total (all in 504) & {blind_spot_stats['total_in_504']} & --- \\\\",
        f"Rule engine correct in 504 & {blind_spot_stats['rule_correct']} & {summary['rule_engine_coverage_in_504']['percentage']:.1f}\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = out_dir / "blind_spot_report.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"  Wrote LaTeX table to {tex_path}")

    # ------------------------------------------------------------------ #
    # PART 5: Print human-readable summary
    # ------------------------------------------------------------------ #
    print("\n=== Blind-Spot Analysis Summary ===\n")
    print(f"Total 504 mẫu bị cả 6 AI miss: {blind_spot_stats['total_in_504']}")
    print(f"  • HATE: {blind_spot_stats['class_distribution']['HATE']}")
    print(f"  • OFFENSIVE: {blind_spot_stats['class_distribution']['OFFENSIVE']}")
    print(f"  • CLEAN: {blind_spot_stats['class_distribution']['CLEAN']}")
    print()
    print(f"Rule Engine coverage trong 504 mẫu:")
    print(f"  • Correct: {blind_spot_stats['rule_correct']} ({summary['rule_engine_coverage_in_504']['percentage']:.1f}%)")
    print(f"    - HATE: {blind_spot_stats['rule_correct_by_class']['HATE']}")
    print(f"    - OFFENSIVE: {blind_spot_stats['rule_correct_by_class']['OFFENSIVE']}")
    print(f"    - CLEAN: {blind_spot_stats['rule_correct_by_class']['CLEAN']}")
    print()
    print("AI vs RE Disagreement Patterns (full test set):")
    for pattern in ["both_right", "ai_right_rule_wrong", "ai_wrong_rule_right", "both_wrong"]:
        d = disagreement_summary[pattern]
        print(f"  • {pattern:30s}: {d['total']:5d} total, {d['in_504']:5d} in 504 blind-spot")

    print("\n✅ Blind-spot analysis complete!")
    print(f"   Outputs written to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
