#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibration diagnostics for SensitiveAI hybrid vs AI-only on bilingual test sets."""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
FUSION_VI = EVAL_DIR / "fusion" / "sensitiveai-vi-custom" / "fusion_predictions.csv"
FUSION_EN = EVAL_DIR / "fusion_en" / "sensitiveai-en-stage2" / "fusion_predictions.csv"
OUTPUT_DIR = EVAL_DIR / "fusion_bilingual"

def load_data(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8")
    lab = {"HATE": 0, "OFFENSIVE": 1, "CLEAN": 2}
    y = df["true_label"].map(lab).to_numpy(dtype=int)
    p_ai = df[["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]].to_numpy(dtype=float)
    p_fuse = df[["fused_hate", "fused_offensive", "fused_clean"]].to_numpy(dtype=float)
    return {"y": y, "p_ai": p_ai, "p_fuse": p_fuse, "n": len(y)}

def compute_metrics(probs: np.ndarray, y: np.ndarray):
    # ECE
    n_bins = 10
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i+1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y)) * abs(acc[mask].mean() - conf[mask].mean())
    
    # Brier
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((probs - onehot)**2, axis=1)))
    
    # NLL
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))
    
    return {"ece": float(ece), "brier": brier, "nll": nll, "mean_conf": float(conf.mean()), "macro_f1": float(pd.Series(preds).value_counts().sum() and (lambda: __import__('sklearn.metrics').metrics.f1_score(y, preds, average='macro'))())}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {"ViHSD": load_data(FUSION_VI), "EN (Davidson)": load_data(FUSION_EN)}
    results = {}
    rows = []

    for name, data in datasets.items():
        results[name] = {}
        for variant, probs in [("AI-only", data["p_ai"]), ("Hybrid", data["p_fuse"])]:
            m = compute_metrics(probs, data["y"])
            m["n"] = data["n"]
            results[name][variant] = m
            rows.append({"Model": "SensitiveAI", "Variant": variant, "Dataset": name, **m})

    # Save CSV & JSON
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "hybrid_calibration.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    
    # Generate Plot
    import matplotlib.pyplot as plt
    from src.evaluation.plots import setup_style
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ["ViHSD", "EN (Davidson)"]):
        for variant, probs in [("AI-only", datasets[name]["p_ai"]), ("Hybrid", datasets[name]["p_fuse"])]:
            # Simplified binning for plot
            conf = probs.max(axis=1)
            preds = probs.argmax(axis=1)
            acc = (preds == datasets[name]["y"]).astype(float)
            bins = pd.cut(conf, np.linspace(0, 1, 11))
            xs, ys = [], []
            for b in bins.unique().sort_values():
                mask = (bins == b)
                if mask.sum() > 10:
                    xs.append(conf[mask].mean())
                    ys.append(acc[mask].mean())
            ax.plot(xs, ys, marker='o', label=variant)
        ax.plot([0,1],[0,1], 'k--')
        ax.set_title(name)
        ax.legend()
    fig.savefig(OUTPUT_DIR / "hybrid_reliability.png", dpi=200)

if __name__ == "__main__":
    main()
