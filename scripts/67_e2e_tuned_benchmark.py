"""Step 67 - Canonical end-to-end benchmark with frozen tuned configuration.

This script supersedes the ad-hoc settings in ``scripts/65_e2e_benchmark_vi_en.py``.
It evaluates, on exactly the same frozen checkpoints and the same frozen
validation-selected decision parameters, five systems:

    A. backbone-only argmax (detected language model)
    B. AI-only argmax (routing-aware)
    C. AI-only + validation thresholds (routing-aware)
    D. hybrid fusion (routing-aware; VI per-channel alpha + thresholds,
       EN AI-only + thresholds)
    E. hybrid fusion with oracle language routing (upper bound)

The per-sample table is written so every downstream table (calibration,
provenance, per-class metrics, paired tests) can be recomputed from one
artifact instead of mixing several historical runs.

Configuration (all values frozen before the test pass)
------------------------------------------------------
VI backbone : sensitiveai-vi-custom-curriculum, run_002 (custom head, max_len 256)
VI fusion   : per-channel alpha (H 0.20, O 0.80, C 1.00)
              thresholds      (H 0.17, O 0.67, C 0.16)
EN backbone : sensitiveai-en-stage2, run_001 (standard head, max_len 128)
EN decision : AI-only + thresholds (H 0.29, O 0.48, C 0.50)
              (validation alpha converged to 1.0, i.e. no rule mixing)
Router      : configs/language_router_tuned.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.constants import ID2LABEL, LABEL2ID  # noqa: E402
from src.evaluation.analysis import mcnemar, paired_bootstrap_delta  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.fusion.weighted_fusion import fuse_scores_per_channel  # noqa: E402
from src.inference.predictor import CheckpointPredictor  # noqa: E402
from src.routing.language_router import LanguageRouter  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUT_DIR = EVAL_DIR / "e2e_tuned_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EN_TEST = PROJECT_ROOT / "data" / "splits_en" / "test.csv"
VI_TEST = PROJECT_ROOT / "data" / "splits" / "test.csv"

EN_CKPT = PROJECT_ROOT / "outputs" / "reports" / "sensitiveai-en-stage2" / "run_001" / "best_model"
VI_CKPT = PROJECT_ROOT / "outputs" / "reports" / "sensitiveai-vi-custom-curriculum" / "run_002" / "best_model"
ROUTER_CONFIG = "configs/language_router_tuned.yaml"

# Frozen validation-selected parameters (outputs/evaluation/fusion/sensitiveai-vi-custom-curriculum-run002/numbers.json).
VI_ALPHAS = [0.20, 0.80, 1.00]          # HATE, OFFENSIVE, CLEAN
VI_THRESHOLDS = {"HATE": 0.17, "OFFENSIVE": 0.67, "CLEAN": 0.16}
EN_THRESHOLDS = {"HATE": 0.29, "OFFENSIVE": 0.48, "CLEAN": 0.50}
LABELS = ["HATE", "OFFENSIVE", "CLEAN"]


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def compute_ece(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > boundaries[i]) & (confidences <= boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = (preds[mask] == labels[mask]).mean()
            ece += mask.sum() / len(labels) * abs(bin_acc - confidences[mask].mean())
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p_true = np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1 - 1e-15)
    return float(-np.mean(np.log(p_true)))


def decide(scores: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Vectorised threshold rule identical to LangAwareFusion._decide_label."""
    tau = np.array([thresholds[label] for label in LABELS], dtype=float)
    above = scores >= tau[None, :]
    out = np.argmax(np.where(above, scores, -np.inf), axis=1)
    no_hit = ~above.any(axis=1)
    out[no_hit] = np.argmax(scores[no_hit], axis=1)
    return out.astype(int)


def metric_block(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> dict:
    m = compute_metrics(y_true, y_pred, proba)
    out = {
        "n": int(len(y_true)),
        "accuracy": round(float(m["accuracy"]), 4),
        "macro_f1": round(float(m["macro_f1"]), 4),
    }
    for label in LABELS:
        out[f"{label}_precision"] = round(float(m[f"{label}_precision"]), 4)
        out[f"{label}_recall"] = round(float(m[f"{label}_recall"]), 4)
        out[f"{label}_f1"] = round(float(m[f"{label}_f1"]), 4)
        out[f"{label}_support"] = int(m[f"{label}_support"])
    out["confusion_matrix"] = np.asarray(m["confusion_matrix"], dtype=int).tolist()
    if proba is not None:
        out["ece"] = round(compute_ece(proba, y_pred, y_true), 4)
        out["brier"] = round(compute_brier(proba, y_true), 4)
        out["nll"] = round(compute_nll(proba, y_true), 4)
        out["mean_confidence"] = round(float(np.mean(np.max(proba, axis=1))), 4)
    return out


def compare(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, label_a: str, label_b: str) -> dict:
    boot = paired_bootstrap_delta(y_true, pred_a, pred_b)
    mc = mcnemar(y_true, pred_a, pred_b)
    return {
        "a": label_a,
        "b": label_b,
        "delta_macro_f1": round(float(boot["delta"]), 4),
        "ci_low": round(float(boot["ci_low"]), 4),
        "ci_high": round(float(boot["ci_high"]), 4),
        "bootstrap_p": round(float(boot["p_value"]), 4),
        "mcnemar_a_right_b_wrong": int(mc["a_correct_b_wrong"]),
        "mcnemar_a_wrong_b_right": int(mc["a_wrong_b_correct"]),
        "mcnemar_p": round(float(mc["p_value"]), 4),
    }


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def infer_cached(predictor: CheckpointPredictor, texts: list[str], indices: list[int], cache: dict[int, np.ndarray]) -> float:
    """Run the predictor on the texts not yet cached; return elapsed seconds."""
    missing = [i for i in indices if i not in cache]
    if not missing:
        return 0.0
    start = time.perf_counter()
    probs = predictor.predict_proba([texts[i] for i in missing])
    elapsed = time.perf_counter() - start
    for i, proba in zip(missing, probs):
        cache[i] = np.asarray(proba, dtype=float)
    return elapsed


def main() -> int:
    print("[1/6] Loading test splits and router...")
    vi_df = pd.read_csv(VI_TEST)
    en_df = pd.read_csv(EN_TEST)
    vi_df["true_lang"] = "vi"
    en_df["true_lang"] = "en"
    mixed = pd.concat([vi_df, en_df], ignore_index=True)
    texts = [str(t) for t in mixed["text"].tolist()]
    y_true = mixed["label"].astype(int).to_numpy()
    true_langs = mixed["true_lang"].to_numpy()
    n = len(texts)
    print(f"    {n} samples (VI {len(vi_df)} / EN {len(en_df)})")

    router = LanguageRouter.from_config_files(router_config_path=ROUTER_CONFIG)
    print(f"    router config: {ROUTER_CONFIG}")

    print("[2/6] Loading frozen checkpoints...")
    vi_pred = CheckpointPredictor(str(VI_CKPT), head="custom", max_length=256, batch_size=32)
    en_pred = CheckpointPredictor(str(EN_CKPT), head="standard", max_length=128, batch_size=64)

    # --- routing pass (production order, one route per comment) ----------- #
    print("[3/6] Routing pass...")
    routed = []
    routing_times = []
    for text in texts:
        t0 = time.perf_counter()
        routed.append(router.route(text))
        routing_times.append((time.perf_counter() - t0) * 1000.0)
    detected = np.array([r.lang for r in routed])
    routing_total = sum(routing_times) / 1000.0

    # Model actually used: detected vi -> vi, detected en -> en, else fallback vi
    # (production fallback order in LangAwareFusion._model_for).
    model_lang = np.where(detected == "en", "en", "vi")
    # Oracle routing uses the ground-truth language.
    oracle_model_lang = true_langs.copy()

    print("[4/6] Backbone inference (routed + oracle, cached per model)...")
    cache_vi: dict[int, np.ndarray] = {}
    cache_en: dict[int, np.ndarray] = {}
    routed_vi = np.where(model_lang == "vi")[0].tolist()
    routed_en = np.where(model_lang == "en")[0].tolist()
    oracle_vi = np.where(true_langs == "vi")[0].tolist()
    oracle_en = np.where(true_langs == "en")[0].tolist()

    vi_sec = infer_cached(vi_pred, texts, sorted(set(routed_vi + oracle_vi)), cache_vi)
    en_sec = infer_cached(en_pred, texts, sorted(set(routed_en + oracle_en)), cache_en)

    p_routed = np.zeros((n, 3))
    p_oracle = np.zeros((n, 3))
    for i in range(n):
        p_routed[i] = (cache_en if model_lang[i] == "en" else cache_vi)[i]
        p_oracle[i] = (cache_en if oracle_model_lang[i] == "en" else cache_vi)[i]

    rule_h = np.array([r.rule_hate_score for r in routed], dtype=float)
    rule_o = np.array([r.rule_offensive_score for r in routed], dtype=float)

    # --- decisions -------------------------------------------------------- #
    print("[5/6] Decisions and metrics...")
    fused_e2e = np.zeros(n, dtype=int)
    fused_e2e_scores = np.zeros((n, 3))
    ai_thr_e2e = np.zeros(n, dtype=int)
    fused_oracle = np.zeros(n, dtype=int)
    fused_oracle_scores = np.zeros((n, 3))
    ai_thr_oracle = np.zeros(n, dtype=int)

    for i in range(n):
        # routing-aware
        if model_lang[i] == "vi":
            scores = fuse_scores_per_channel(p_routed[i : i + 1], rule_h[i : i + 1], rule_o[i : i + 1], VI_ALPHAS)[0]
            fused_e2e[i] = decide(scores[None, :], VI_THRESHOLDS)[0]
            ai_thr_e2e[i] = decide(p_routed[i : i + 1], VI_THRESHOLDS)[0]
        else:
            scores = p_routed[i].copy()
            fused_e2e[i] = decide(p_routed[i : i + 1], EN_THRESHOLDS)[0]
            ai_thr_e2e[i] = fused_e2e[i]
        fused_e2e_scores[i] = scores

        # oracle
        if oracle_model_lang[i] == "vi":
            o_scores = fuse_scores_per_channel(
                p_oracle[i : i + 1], rule_h[i : i + 1], rule_o[i : i + 1], VI_ALPHAS
            )[0]
            fused_oracle[i] = decide(o_scores[None, :], VI_THRESHOLDS)[0]
            ai_thr_oracle[i] = decide(p_oracle[i : i + 1], VI_THRESHOLDS)[0]
        else:
            o_scores = p_oracle[i].copy()
            fused_oracle[i] = decide(p_oracle[i : i + 1], EN_THRESHOLDS)[0]
            ai_thr_oracle[i] = fused_oracle[i]
        fused_oracle_scores[i] = o_scores

    ai_argmax_e2e = np.argmax(p_routed, axis=1)
    ai_argmax_oracle = np.argmax(p_oracle, axis=1)

    systems = {
        "ai_argmax_e2e": ai_argmax_e2e,
        "ai_thresholds_e2e": ai_thr_e2e,
        "hybrid_e2e": fused_e2e,
        "ai_argmax_oracle": ai_argmax_oracle,
        "ai_thresholds_oracle": ai_thr_oracle,
        "hybrid_oracle": fused_oracle,
    }

    results: dict[str, object] = {
        "protocol": {
            "router_config": ROUTER_CONFIG,
            "vi_checkpoint": str(VI_CKPT.relative_to(PROJECT_ROOT)),
            "en_checkpoint": str(EN_CKPT.relative_to(PROJECT_ROOT)),
            "vi_alphas": dict(zip(LABELS, VI_ALPHAS)),
            "vi_thresholds": VI_THRESHOLDS,
            "en_thresholds": EN_THRESHOLDS,
            "en_fusion": "alpha=1.0 on validation; AI-only + thresholds",
            "fallback": "detected mix/other -> vi backbone (production fallback order)",
            "test_size": n,
        },
        "routing": {
            "accuracy": round(float(np.mean(detected == true_langs)), 4),
            "detected_distribution": {k: int((detected == k).sum()) for k in sorted(set(detected.tolist()))},
            "confusion": {
                f"{tl}->{dl}": int(((true_langs == tl) & (detected == dl)).sum())
                for tl in ("vi", "en")
                for dl in sorted(set(detected.tolist()))
            },
        },
        "systems": {},
        "per_language": {},
        "significance": {},
    }

    for name, pred in systems.items():
        proba = None
        if name.startswith("hybrid"):
            proba = fused_e2e_scores if name.endswith("e2e") else fused_oracle_scores
        elif name.startswith("ai"):
            proba = p_routed if name.endswith("e2e") else p_oracle
        results["systems"][name] = metric_block(y_true, pred, proba)
        results["per_language"][name] = {
            lang: metric_block(y_true[true_langs == lang], pred[true_langs == lang],
                               None if proba is None else proba[true_langs == lang])
            for lang in ("vi", "en")
        }

    # changed decisions: hybrid vs AI-only argmax (same routing)
    corrections = int(np.sum((ai_argmax_e2e != y_true) & (fused_e2e == y_true)))
    regressions = int(np.sum((ai_argmax_e2e == y_true) & (fused_e2e != y_true)))
    results["change_analysis"] = {
        "accuracy_tolerant": "matched on all labels (CLEAN included)",
        "corrections": corrections,
        "regressions": regressions,
        "net": corrections - regressions,
    }

    # paired significance
    results["significance"]["hybrid_vs_ai_argmax_e2e"] = compare(
        y_true, fused_e2e, ai_argmax_e2e, "hybrid_e2e", "ai_argmax_e2e"
    )
    results["significance"]["hybrid_vs_ai_argmax_oracle"] = compare(
        y_true, fused_oracle, ai_argmax_oracle, "hybrid_oracle", "ai_argmax_oracle"
    )

    # latency (serialised, measured)
    routing_total_measured = sum(routing_times) / 1000.0
    results["latency"] = {
        "routing_mean_ms": round(float(np.mean(routing_times)), 3),
        "routing_p50_ms": round(float(np.percentile(routing_times, 50)), 3),
        "routing_p95_ms": round(float(np.percentile(routing_times, 95)), 3),
        "routing_p99_ms": round(float(np.percentile(routing_times, 99)), 3),
        "routing_texts_per_sec": round(n / routing_total_measured, 1),
        "vi_backbone_total_sec": round(vi_sec, 2),
        "en_backbone_total_sec": round(en_sec, 2),
        "ai_total_sec": round(vi_sec + en_sec, 2),
        "ai_ms_per_text": round((vi_sec + en_sec) / n * 1000.0, 3),
        "total_mean_ms": round(float(np.mean(routing_times)) + (vi_sec + en_sec) / n * 1000.0, 3),
        "total_p95_conservative_ms": round(
            float(np.percentile(routing_times, 95)) + ((vi_sec + en_sec) / n * 1000.0), 3
        ),
    }

    # --- per-sample artifact --------------------------------------------- #
    frame = pd.DataFrame(
        {
            "row": np.arange(n),
            "true_lang": true_langs,
            "detected_lang": detected,
            "model_lang": model_lang,
            "y_true": y_true,
            "y_true_name": [ID2LABEL[i] for i in y_true],
            "rule_hate": np.round(rule_h, 6),
            "rule_offensive": np.round(rule_o, 6),
            "ai_hate": np.round(p_routed[:, LABEL2ID["HATE"]], 6),
            "ai_offensive": np.round(p_routed[:, LABEL2ID["OFFENSIVE"]], 6),
            "ai_clean": np.round(p_routed[:, LABEL2ID["CLEAN"]], 6),
            "fused_hate": np.round(fused_e2e_scores[:, LABEL2ID["HATE"]], 6),
            "fused_offensive": np.round(fused_e2e_scores[:, LABEL2ID["OFFENSIVE"]], 6),
            "fused_clean": np.round(fused_e2e_scores[:, LABEL2ID["CLEAN"]], 6),
            "pred_ai_argmax": ai_argmax_e2e,
            "pred_ai_thresholds": ai_thr_e2e,
            "pred_hybrid": fused_e2e,
            "pred_oracle_hybrid": fused_oracle,
            "correct_ai": (ai_argmax_e2e == y_true).astype(int),
            "correct_hybrid": (fused_e2e == y_true).astype(int),
        }
    )
    frame.to_csv(OUT_DIR / "per_sample_predictions.csv", index=False, encoding="utf-8")

    # --- write outputs ---------------------------------------------------- #
    results["label_distribution"] = {ID2LABEL[i]: int((y_true == i).sum()) for i in range(3)}
    out_json = OUT_DIR / "e2e_tuned_benchmark.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown report
    lines = ["# End-to-end tuned benchmark", ""]
    for name, m in results["systems"].items():
        lines.append(
            f"- **{name}**: acc {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}, "
            f"ECE {m.get('ece', float('nan')):.4f}"
        )
    lines.append("")
    lines.append("## Per language")
    for name, per in results["per_language"].items():
        for lang, m in per.items():
            lines.append(
                f"- {name} / {lang}: acc {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}, "
                f"H {m['HATE_f1']:.4f}, O {m['OFFENSIVE_f1']:.4f}, C {m['CLEAN_f1']:.4f}"
            )
    lines.append("")
    lines.append("## Significance")
    for key, s in results["significance"].items():
        lines.append(
            f"- {key}: delta {s['delta_macro_f1']:+.4f} "
            f"[{s['ci_low']:+.4f},{s['ci_high']:+.4f}], p={s['bootstrap_p']:.4f}; "
            f"McNemar {s['mcnemar_a_right_b_wrong']}/{s['mcnemar_a_wrong_b_right']}, p={s['mcnemar_p']:.4f}"
        )
    (OUT_DIR / "e2e_tuned_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # LaTeX table: full-split comparison
    def row(name: str, m: dict, lang: str, dataset: str) -> str:
        return (
            f"{name} & {lang} & {dataset} & {m['n']:,} & {m['accuracy']:.4f} & "
            f"{m['macro_f1']:.4f} & {m['HATE_f1']:.4f} & {m['OFFENSIVE_f1']:.4f} & {m['CLEAN_f1']:.4f} \\\\"
        )

    tex = [
        "% Auto-generated by scripts/67_e2e_tuned_benchmark.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{End-to-end results on the full test splits under the frozen tuned configuration. "
        "\\emph{Hybrid} applies the VI per-channel fusion and the EN AI-only threshold rule; "
        "\\emph{Auto} uses the router output, \\emph{Oracle} the ground-truth language.}",
        "\\label{tab:e2e_tuned}",
        "\\begin{tabular}{llcrcccc}",
        "\\toprule",
        "System & Lang. & Dataset & $N$ & Acc. & Macro-F1 & \\hate & \\off & \\clean \\\\",
        "\\midrule",
        "\\multicolumn{9}{l}{\\emph{Automatic routing (production)}} \\\\",
        row("AI-only (argmax)", results["per_language"]["ai_argmax_e2e"]["vi"], "VI", "ViHSD"),
        row("Hybrid", results["per_language"]["hybrid_e2e"]["vi"], "VI", "ViHSD"),
        row("AI-only (argmax)", results["per_language"]["ai_argmax_e2e"]["en"], "EN", "Davidson"),
        row("AI-only (thr.)", results["per_language"]["ai_thresholds_e2e"]["en"], "EN", "Davidson"),
        "\\midrule",
        "\\multicolumn{9}{l}{\\emph{Mixed stream ($N=9{,}158$)}} \\\\",
        row("AI-only (argmax)", results["systems"]["ai_argmax_e2e"], "VI+EN", "Both"),
        row("Hybrid", results["systems"]["hybrid_e2e"], "VI+EN", "Both"),
        "\\midrule",
        "\\multicolumn{9}{l}{\\emph{Oracle language routing (upper bound)}} \\\\",
        row("Hybrid", results["per_language"]["hybrid_oracle"]["vi"], "VI", "ViHSD"),
        row("Hybrid", results["per_language"]["hybrid_oracle"]["en"], "EN", "Davidson"),
        row("Hybrid", results["systems"]["hybrid_oracle"], "VI+EN", "Both"),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    (OUT_DIR / "e2e_tuned_benchmark.tex").write_text("\n".join(tex), encoding="utf-8")

    print("\n=== Systems ===")
    for name, m in results["systems"].items():
        print(f"  {name:24s} acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} ece={m.get('ece')}")
    print("\n=== Per language ===")
    for name, per in results["per_language"].items():
        for lang, m in per.items():
            print(f"  {name:24s} {lang} acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
                  f"H={m['HATE_f1']:.4f} O={m['OFFENSIVE_f1']:.4f} C={m['CLEAN_f1']:.4f}")
    print("\n=== Significance ===")
    for key, s in results["significance"].items():
        print(f"  {key}: {s}")
    print(f"\nSaved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
