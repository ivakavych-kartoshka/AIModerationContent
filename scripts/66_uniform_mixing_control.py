#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Uniform-mixing control for the hybrid calibration claim.

The hybrid fuses AI probabilities with rule pseudo-probabilities, which both
moves decisions and shrinks confidence toward the uniform distribution
(1/3, 1/3, 1/3).  A lower ECE could therefore be an artefact of shrinkage
rather than better evidence.  This script isolates the effect by replacing the
rule channel with a constant uniform distribution at the same mixing weight

    p_uniform = alpha * p_ai + (1 - alpha) * (1/3, 1/3, 1/3)

and comparing ECE / Brier / NLL against AI-only and the real hybrid.

Run::

    python scripts/66_uniform_mixing_control.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "fusion"
PRED_PATH = FUSION_DIR / "sensitiveai-vi-custom" / "fusion_predictions.csv"
OUT_DIR = FUSION_DIR

LABELS = ("HATE", "OFFENSIVE", "CLEAN")
LABEL2ID = {lbl: idx for idx, lbl in enumerate(LABELS)}
ALPHA = 0.70


def brier(proba: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    picked = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(picked)))


def ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count:
            out += (count / len(y)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(out)


def metrics(proba: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = proba.argmax(axis=1)
    return {
        "ece": ece(proba, y),
        "brier": brier(proba, y),
        "nll": nll(proba, y),
        "mean_conf": float(proba.max(axis=1).mean()),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "n": int(len(y)),
    }


def main() -> int:
    df = pd.read_csv(PRED_PATH, encoding="utf-8")
    y = df["true_label"].map(LABEL2ID).to_numpy(dtype=int)
    p_ai = df[["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]].to_numpy(dtype=float)
    p_hybrid = df[["fused_hate", "fused_offensive", "fused_clean"]].to_numpy(dtype=float)

    uniform = np.full_like(p_ai, 1.0 / 3.0)
    p_control = ALPHA * p_ai + (1.0 - ALPHA) * uniform

    results: Dict[str, Dict[str, float]] = {
        "ai_only": metrics(p_ai, y),
        "hybrid": metrics(p_hybrid, y),
        "uniform_mix_alpha_0.70": metrics(p_control, y),
    }

    # Alpha sweep: separate the shrinkage trend from the rule contribution.
    sweep: List[Dict[str, float]] = []
    for a in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        p = a * p_ai + (1.0 - a) * uniform
        m = metrics(p, y)
        sweep.append({"alpha": float(a), **m})
    results["uniform_sweep"] = sweep  # type: ignore[assignment]

    out = {
        "dataset": "ViHSD test",
        "n": int(len(y)),
        "alpha": ALPHA,
        "variants": {k: v for k, v in results.items() if k != "uniform_sweep"},
        "uniform_sweep": sweep,
    }
    path = OUT_DIR / "uniform_mixing_control.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {path}")

    for name in ("ai_only", "hybrid", "uniform_mix_alpha_0.70"):
        r = results[name]
        print(
            f"{name:26s} ECE={r['ece']:.4f} Brier={r['brier']:.4f} "
            f"NLL={r['nll']:.4f} conf={r['mean_conf']:.4f} "
            f"acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
