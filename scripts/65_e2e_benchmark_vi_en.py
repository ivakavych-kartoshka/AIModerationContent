"""Step 65 - End-to-end benchmark: VI (ViHSD) + EN (Davidson) backbones.

Runs inference on test splits, computes accuracy/F1/calibration/latency.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import CheckpointPredictor  # noqa: E402
from src.common.constants import ID2LABEL  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.fusion.lang_aware_fusion import LangAwareFusion  # noqa: E402
from src.fusion.weighted_fusion import fuse_scores  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUT_DIR = EVAL_DIR / "e2e_vi_en_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EN_TEST = PROJECT_ROOT / "data" / "splits_en" / "test.csv"
VI_TEST = PROJECT_ROOT / "data" / "splits" / "test.csv"

EN_CKPT = PROJECT_ROOT / "outputs" / "reports" / "sensitiveai-en-stage2" / "run_001" / "best_model"
VI_CKPT = PROJECT_ROOT / "outputs" / "reports" / "sensitiveai-vi-custom-curriculum" / "run_002" / "best_model"


def compute_ece(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = (preds[mask] == labels[mask]).mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error between predicted probabilities and one-hot true labels."""
    n_classes = probs.shape[1]
    one_hot = np.zeros((len(labels), n_classes))
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood: -mean(log(p_true_class))."""
    epsilon = 1e-15
    p_true = probs[np.arange(len(labels)), labels]
    p_true = np.clip(p_true, epsilon, 1.0 - epsilon)
    return float(-np.mean(np.log(p_true)))


def compute_mean_confidence(probs: np.ndarray) -> float:
    """Mean confidence: mean of max class probability for each sample."""
    return float(np.mean(np.max(probs, axis=1)))



def benchmark_backbone(
    test_csv: Path,
    ckpt_path: Path,
    head: str,
    max_length: int,
    batch_size: int,
    lang: str,
) -> dict:
    """Run inference + metrics on test split."""
    print(f"  Loading {test_csv.name}...")
    df = pd.read_csv(test_csv)
    texts = df["text"].tolist()
    true_labels = df["label"].values
    print(f"    {len(df)} samples")

    print(f"  Loading checkpoint {ckpt_path.name}...")
    predictor = CheckpointPredictor(str(ckpt_path), head=head, max_length=max_length)

    print(f"  Running inference (batch_size={batch_size})...")
    start = time.time()
    probs = predictor.predict_proba(texts)
    elapsed = time.time() - start

    pred_labels = np.argmax(probs, axis=1)

    # Metrics
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    ece = compute_ece(probs, pred_labels, true_labels, n_bins=10)

    # Latency
    latency_ms = (elapsed / len(texts)) * 1000
    throughput = len(texts) / elapsed

    # Per-class F1
    report = classification_report(true_labels, pred_labels, target_names=["HATE", "OFFENSIVE", "CLEAN"], output_dict=True)

    return {
        "lang": lang,
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "head": head,
        "max_length": max_length,
        "batch_size": batch_size,
        "n_samples": int(len(df)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "ece": round(float(ece), 4),
        "latency_ms_per_text": round(float(latency_ms), 2),
        "throughput_texts_per_sec": round(float(throughput), 2),
        "total_time_sec": round(float(elapsed), 2),
        "f1_hate": round(float(report["HATE"]["f1-score"]), 4),
        "f1_offensive": round(float(report["OFFENSIVE"]["f1-score"]), 4),
        "f1_clean": round(float(report["CLEAN"]["f1-score"]), 4),
        "label_distribution": {ID2LABEL[i]: int((true_labels == i).sum()) for i in [0, 1, 2]},
    }


def run_mixed_pipeline(vi_ckpt: Path, en_ckpt: Path) -> dict:
    """Full production chain: router (langid + rule engine) -> backbone -> fusion."""
    print("  Loading VI + EN test splits...")
    vi_df = pd.read_csv(VI_TEST).assign(true_lang="vi")
    en_df = pd.read_csv(EN_TEST).assign(true_lang="en")
    mixed = pd.concat([en_df, vi_df], ignore_index=True)

    texts = [str(t) for t in mixed["text"].tolist()]
    y_true = mixed["label"].astype(int).to_numpy()
    true_langs = mixed["true_lang"].tolist()

    print(f"    {len(texts)} mixed samples (VI {len(vi_df)} / EN {len(en_df)})")

    fusion = LangAwareFusion.from_config("configs/language_router.yaml")
    fusion.alphas = {k: 0.70 for k in ("vi", "en", "mix", "multilingual", "default")}
    vi_pred = CheckpointPredictor(str(vi_ckpt), head="custom", max_length=256)
    en_pred = CheckpointPredictor(str(en_ckpt), head="standard", max_length=128)
    fusion.set_model("vi", vi_pred)
    fusion.set_model("en", en_pred)

    # pass 1: routing (langid + rule engine, per text as in production)
    print("  Pass 1/3: routing...")
    t0 = time.perf_counter()
    routed_all = []
    routing_times = []
    for t in texts:
        t_start = time.perf_counter()
        routed_all.append(fusion.router.route(t))
        routing_times.append((time.perf_counter() - t_start) * 1000.0)  # ms
    routing_sec = time.perf_counter() - t0
    routing_ms_per_text = (routing_sec / len(texts)) * 1000.0
    routing_p50 = float(np.percentile(routing_times, 50))
    routing_p95 = float(np.percentile(routing_times, 95))
    routing_p99 = float(np.percentile(routing_times, 99))

    # pass 2: batch backbone inference per routed language
    print("  Pass 2/3: backbone inference...")
    buckets: dict[str, list[int]] = {}
    for i, r in enumerate(routed_all):
        ml = r.lang if r.lang in ("vi", "en") else "multilingual"
        buckets.setdefault(ml, []).append(i)

    p_ai_all: dict[int, np.ndarray] = {}
    ai_sec = 0.0
    ai_times = []
    for lang_key, idxs in buckets.items():
        if lang_key not in ("vi", "en"):
            continue
        pred = vi_pred if lang_key == "vi" else en_pred
        batch_texts = [texts[i] for i in idxs]
        t0 = time.perf_counter()
        probs = pred.predict_proba(batch_texts)
        batch_sec = time.perf_counter() - t0
        ai_sec += batch_sec
        # Track per-text AI inference time (batch overhead amortized)
        per_text_ms = (batch_sec / len(batch_texts)) * 1000.0
        ai_times.extend([per_text_ms] * len(batch_texts))
        for row, proba in zip(idxs, probs):
            p_ai_all[row] = proba
    ai_ms_per_text = (ai_sec / len(texts)) * 1000.0
    ai_p50 = float(np.percentile(ai_times, 50))
    ai_p95 = float(np.percentile(ai_times, 95))
    ai_p99 = float(np.percentile(ai_times, 99))
    total_p50 = routing_p50 + ai_p50
    total_p95 = routing_p95 + ai_p95
    total_p99 = routing_p99 + ai_p99

    # pass 3: fusion + decision
    print("  Pass 3/3: fusion + decision...")
    y_pred = np.zeros(len(texts), dtype=int)
    fused_scores = np.zeros((len(texts), 3), dtype=float)
    for i, r in enumerate(routed_all):
        proba = p_ai_all.get(i)
        rule_hate = np.array([r.rule_hate_score], dtype=float)
        rule_off = np.array([r.rule_offensive_score], dtype=float)
        if proba is not None:
            fused = fuse_scores(proba[None, :], rule_hate, rule_off, 0.70)
            y_pred[i] = fusion._decide_label(fused[0])
        else:
            from src.fusion.weighted_fusion import rule_pseudo_proba
            fused = rule_pseudo_proba(rule_hate, rule_off)
            y_pred[i] = int(np.argmax(fused[0]))
        fused_scores[i] = fused[0]

    ece = compute_ece(fused_scores, y_pred, y_true, n_bins=10)
    brier = compute_brier(fused_scores, y_true)
    nll = compute_nll(fused_scores, y_true)
    mean_conf = compute_mean_confidence(fused_scores)
    overall = compute_metrics(y_true, y_pred, fused_scores)


    per_lang = {}
    for lang in ("vi", "en"):
        mask = np.array([tl == lang for tl in true_langs])
        m = compute_metrics(y_true[mask], y_pred[mask], fused_scores[mask])
        per_lang[lang] = {
            "n": int(mask.sum()),
            "accuracy": round(float(m.get("accuracy", float("nan"))), 4),
            "macro_f1": round(float(m.get("macro_f1", float("nan"))), 4),
            "HATE_f1": round(float(m.get("HATE_f1", float("nan"))), 4),
            "OFFENSIVE_f1": round(float(m.get("OFFENSIVE_f1", float("nan"))), 4),
            "CLEAN_f1": round(float(m.get("CLEAN_f1", float("nan"))), 4),
        }

    detected = np.array([r.lang for r in routed_all])
    lang_correct = sum(1 for tl, dl in zip(true_langs, detected) if tl == dl)

    return {
        "n_samples": int(len(texts)),
        "accuracy": round(float(overall.get("accuracy", float("nan"))), 4),
        "macro_f1": round(float(overall.get("macro_f1", float("nan"))), 4),
        "ece": round(float(ece), 4),
        "brier": round(float(brier), 4),
        "nll": round(float(nll), 4),
        "mean_conf": round(float(mean_conf), 4),
        "routing_ms_per_text": round(float(routing_ms_per_text), 3),
        "routing_p50_ms": round(float(routing_p50), 3),
        "routing_p95_ms": round(float(routing_p95), 3),
        "routing_p99_ms": round(float(routing_p99), 3),
        "routing_texts_per_sec": round(float(len(texts) / routing_sec), 1),
        "ai_total_sec": round(float(ai_sec), 2),
        "ai_ms_per_text": round(float(ai_ms_per_text), 3),
        "ai_p50_ms": round(float(ai_p50), 3),
        "ai_p95_ms": round(float(ai_p95), 3),
        "ai_p99_ms": round(float(ai_p99), 3),
        "ai_texts_per_sec": round(float(len(texts) / ai_sec), 1) if ai_sec > 0 else None,
        "total_p50_ms": round(float(total_p50), 3),
        "total_p95_ms": round(float(total_p95), 3),
        "total_p99_ms": round(float(total_p99), 3),
        "langdetect_accuracy": round(lang_correct / len(texts), 4),
        "per_language": per_lang,
    }


def main() -> int:
    print("[1/3] Benchmarking Vietnamese backbone (ViHSD test)...")
    vi_result = benchmark_backbone(
        test_csv=VI_TEST,
        ckpt_path=VI_CKPT,
        head="custom",
        max_length=256,
        batch_size=32,
        lang="vi",
    )

    print("\n[2/3] Benchmarking English backbone (Davidson test)...")
    en_result = benchmark_backbone(
        test_csv=EN_TEST,
        ckpt_path=EN_CKPT,
        head="standard",
        max_length=128,
        batch_size=64,
        lang="en",
    )

    print("\n[3/4] Running full e2e mixed pipeline (router -> backbones -> fusion)...")
    mixed_result = run_mixed_pipeline(VI_CKPT, EN_CKPT)

    print("\n[4/4] Writing summary...")
    summary = {
        "vietnamese": vi_result,
        "english": en_result,
        "mixed_e2e": mixed_result,
        "comparison": {
            "accuracy_vi_vs_en": f"{vi_result['accuracy']:.4f} vs {en_result['accuracy']:.4f}",
            "macro_f1_vi_vs_en": f"{vi_result['macro_f1']:.4f} vs {en_result['macro_f1']:.4f}",
            "ece_vi_vs_en": f"{vi_result['ece']:.4f} vs {en_result['ece']:.4f}",
            "throughput_vi_vs_en": f"{vi_result['throughput_texts_per_sec']:.1f} vs {en_result['throughput_texts_per_sec']:.1f} texts/s",
        },
    }

    out_json = OUT_DIR / "e2e_vi_en_benchmark.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved {out_json}")

    # LaTeX table
    latex = rf"""%% Auto-generated by scripts/65_e2e_benchmark_vi_en.py
\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{lcccccc}}
\toprule
\textbf{{Language}} & \textbf{{Dataset}} & \textbf{{$N$}} & \textbf{{Accuracy}} & \textbf{{Macro-F1}} & \textbf{{ECE}} & \textbf{{Throughput}} \\
\midrule
Vietnamese & ViHSD & {vi_result['n_samples']:,} & {vi_result['accuracy']:.4f} & {vi_result['macro_f1']:.4f} & {vi_result['ece']:.4f} & {vi_result['throughput_texts_per_sec']:.1f} texts/s \\
English & Davidson & {en_result['n_samples']:,} & {en_result['accuracy']:.4f} & {en_result['macro_f1']:.4f} & {en_result['ece']:.4f} & {en_result['throughput_texts_per_sec']:.1f} texts/s \\
\bottomrule
\end{{tabular}}
\caption{{End-to-end backbone evaluation on Vietnamese (ViHSD) and English (Davidson) hate-speech test splits.}}
\label{{tab:e2e_vi_en_benchmark}}
\end{{table}}
"""
    latex_path = OUT_DIR / "e2e_vi_en_benchmark.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"  saved {latex_path}")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
