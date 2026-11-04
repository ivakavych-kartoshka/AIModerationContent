"""Step 54 - end-to-end evaluation of the language-aware pipeline on a mixed sample.

    python scripts/54_end_to_end_eval.py
    python scripts/54_end_to_end_eval.py --samples-per-lang 1000 --seed 42

Runs the full production chain over a mixed sample and writes a
per-sample CSV + aggregate metrics / JSON / markdown summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.constants import ID2LABEL, LABELS  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.fusion.lang_aware_fusion import LangAwareFusion, _PredictorAdapter  # noqa: E402
from src.inference.predictor import CheckpointPredictor  # noqa: E402
from src.fusion.lang_aware_fusion import _as_proba  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/evaluation/mixed_1000_test.csv", help="mixed sample CSV (text,label,true_lang)")
    parser.add_argument("--out-dir", default="outputs/evaluation/end_to_end_1000")
    parser.add_argument("--en-checkpoint", default="outputs/reports/sensitiveai-en-stage2/run_001/best_model")
    parser.add_argument("--vi-checkpoint", default="outputs/reports/sensitiveai-vi-custom-curriculum/run_002/best_model")
    parser.add_argument("--vi-head", default="custom")
    parser.add_argument("--en-head", default="standard")
    parser.add_argument("--vi-max-length", type=int, default=256)
    parser.add_argument("--en-max-length", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--samples-per-lang", type=int, default=1000, help="how many per language to draw (with seed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    args = parser.parse_args(argv)

    wall_start = time.perf_counter()

    from src.common.logging_utils import get_logger

    LOGGER = get_logger("sensitiveai.scripts.54_end_to_end_eval")

    # ------------------------------------------------------------------ #
    # load / draw sample
    # ------------------------------------------------------------------ #
    import pandas as pd

    df = pd.read_csv(args.input)
    per_lang = int(args.samples_per_lang)
    rng = np.random.default_rng(int(args.seed))
    frames = []
    for lang, group in df.groupby("true_lang", sort=False):
        n = min(per_lang, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        frames.append(group.loc[sorted(idx)])
    sample = pd.concat(frames, ignore_index=True)

    texts = [str(t) for t in sample["text"].tolist()]
    true_labels = [int(y) for y in sample["label"].tolist()]
    true_langs = [str(t) for t in sample["true_lang"].tolist()]

    # ------------------------------------------------------------------ #
    # build fusion & predictors
    # ------------------------------------------------------------------ #
    fusion = LangAwareFusion.from_config("configs/language_router.yaml")
    fusion.alphas = {str(k): float(v) for k, v in ({"vi": args.alpha, "en": args.alpha, "mix": args.alpha, "multilingual": args.alpha, "default": args.alpha}).items()}

    vi_pred = CheckpointPredictor(args.vi_checkpoint, head=args.vi_head, max_length=args.vi_max_length) if args.vi_checkpoint else None
    en_pred = CheckpointPredictor(args.en_checkpoint, head=args.en_head, max_length=args.en_max_length) if args.en_checkpoint else None
    fusion.set_model("vi", vi_pred)
    fusion.set_model("en", en_pred)

    # ------------------------------------------------------------------ #
    # pass 1: route all texts (language detection + rule engine)
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    routed_all = []
    per_text_routing_ms: List[float] = []
    for t in texts:
        tt = time.perf_counter()
        routed_all.append(fusion.router.route(t))
        per_text_routing_ms.append((time.perf_counter() - tt) * 1000.0)
    routing_ms = (time.perf_counter() - t0) * 1000.0
    LOGGER.info("[e2e] routed %d texts in %.1f ms (%.2f ms/text)", len(texts), routing_ms, routing_ms / len(texts))

    # ------------------------------------------------------------------ #
    # pass 2: batch AI inference per model_language
    # ------------------------------------------------------------------ #
    buckets: Dict[str, List[int]] = {}
    for i, routed in enumerate(routed_all):
        ml = routed.lang if routed.lang in ("vi", "en") else "multilingual"
        buckets.setdefault(ml, []).append(i)

    p_ai_all: Dict[int, np.ndarray] = {}
    total_ai_ms = 0.0
    ai_batch_count = 0
    for lang_key, idxs in buckets.items():
        if lang_key not in ("vi", "en"):
            continue
        batch_texts = [texts[i] for i in idxs]
        predictor = vi_pred if lang_key == "vi" else en_pred
        t0 = time.perf_counter()
        probs = predictor.predict_proba(batch_texts)
        dt = (time.perf_counter() - t0) * 1000.0
        total_ai_ms += dt
        ai_batch_count += 1
        LOGGER.info("[e2e] AI %s batch %d texts -> %.1f ms (%.2f ms/text)", lang_key, len(batch_texts), dt, dt / len(batch_texts))
        for row, proba in zip(idxs, probs):
            p_ai_all[row] = proba

    ai_latency_ms = total_ai_ms / len(texts) if texts else 0.0

    # ------------------------------------------------------------------ #
    # pass 3: fuse per text & decide
    # ------------------------------------------------------------------ #
    rows: List[Dict[str, Any]] = []
    for i, (text, true_label, true_lang) in enumerate(zip(texts, true_labels, true_langs)):
        t_text = time.perf_counter()
        routed = routed_all[i]
        lang = routed.lang
        ml = lang if lang in ("vi", "en") else "multilingual"
        alpha = float(fusion._alpha_for(lang if lang in fusion.alphas else ml))

        proba = p_ai_all.get(i)
        ai_used = proba is not None
        p = _as_proba(proba, 1) if ai_used else None

        rule_hate = np.array([routed.rule_hate_score], dtype=float)
        rule_off = np.array([routed.rule_offensive_score], dtype=float)

        if ai_used:
            from src.fusion.weighted_fusion import fuse_scores
            fused = fuse_scores(p, rule_hate, rule_off, alpha)
        else:
            from src.fusion.weighted_fusion import rule_pseudo_proba
            fused = rule_pseudo_proba(rule_hate, rule_off)

        label_id = fusion._decide_label(fused[0])

        rows.append(
            {
                "text": text,
                "true_lang": true_lang,
                "true_label": int(true_label),
                "true_label_name": LABELS[true_label],
                "detected_lang": lang,
                "model_language": ml,
                "alpha": float(alpha),
                "ai_used": bool(ai_used),
                "rule_hate": float(routed.rule_hate_score),
                "rule_offensive": float(routed.rule_offensive_score),
                "rule_clean": float(routed.rule_clean_score),
                "p_ai_hate": float(p[0][0]) if ai_used else 0.0,
                "p_ai_offensive": float(p[0][1]) if ai_used else 0.0,
                "p_ai_clean": float(p[0][2]) if ai_used else 0.0,
                "fused_hate": float(fused[0][0]),
                "fused_offensive": float(fused[0][1]),
                "fused_clean": float(fused[0][2]),
                "pred_label_id": int(label_id),
                "pred_label": ID2LABEL[int(label_id)],
                "correct": int(label_id == true_label),
                "n_matches": len(routed.matches),
                "latency_routing_ms": float(per_text_routing_ms[i]),
                "latency_ai_ms": float(ai_latency_ms),
                "latency_total_ms": float((time.perf_counter() - t_text) * 1000.0),  # fusion-only per text
            }
        )

    # ------------------------------------------------------------------ #
    # write per-sample CSV
    # ------------------------------------------------------------------ #
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "e2e_results.csv"
    result_df = pd.DataFrame(rows)
    result_df.to_csv(csv_path, index=False, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # aggregate metrics
    # ------------------------------------------------------------------ #
    y_true = np.asarray([r["true_label"] for r in rows], dtype=int)
    y_pred = np.asarray([r["pred_label_id"] for r in rows], dtype=int)
    scores = np.asarray([[r["fused_hate"], r["fused_offensive"], r["fused_clean"]] for r in rows], dtype=float)
    metrics = compute_metrics(y_true, y_pred, scores)

    report_lines = [
        "# End-to-end language-aware pipeline - mixed sample",
        "",
        f"- input sample: {len(rows)} texts (EN {sum(1 for r in rows if r['true_lang'] == 'en')} / VI {sum(1 for r in rows if r['true_lang'] == 'vi')})",
        f"- fusion alpha: {args.alpha}",
        f"- decision: argmax (no extra thresholds)",
        f"- routing time: {routing_ms:.1f} ms total ({routing_ms/len(rows):.2f} ms/text)",
        f"- AI inference time: {total_ai_ms:.1f} ms total ({total_ai_ms/len(rows):.2f} ms/text, {ai_batch_count} batches)",
        "",
        "## Aggregate (fused scores -> final decision)",
        "",
        f"- accuracy : {metrics.get('accuracy', float('nan')):.4f}",
        f"- macro_f1: {metrics.get('macro_f1', float('nan')):.4f}",
        f"- weighted_f1: {metrics.get('weighted_f1', float('nan')):.4f}",
        "",
        "## Per-language breakdown",
        "",
    ]
    for lang in ("vi", "en"):
        mask = np.asarray([r["true_lang"] == lang for r in rows])
        if mask.any():
            m = compute_metrics(y_true[mask], y_pred[mask], scores[mask])
            report_lines.append(
                f"- {lang}: n={int(mask.sum())} acc={m.get('accuracy', float('nan')):.4f} macro_f1={m.get('macro_f1', float('nan')):.4f}"
            )

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps({"metrics": metrics, "n": len(rows), "alpha": args.alpha}, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(report_lines), encoding="utf-8")

    LOGGER.info("[e2e] wrote %s", csv_path)
    LOGGER.info("[e2e] wrote %s", json_path)
    LOGGER.info("[e2e] wrote %s", md_path)

    # ------------------------------------------------------------------ #
    # supplementary CSVs: per-language metrics + latency summary
    # ------------------------------------------------------------------ #
    import pandas as pd

    per_lang_rows = []
    for lang in ("vi", "en"):
        mask = np.asarray([r["true_lang"] == lang for r in rows])
        if not mask.any():
            continue
        m = compute_metrics(y_true[mask], y_pred[mask], scores[mask])
        per_lang_rows.append(
            {
                "language": lang,
                "n": int(mask.sum()),
                "accuracy": float(m.get("accuracy", float("nan"))),
                "macro_f1": float(m.get("macro_f1", float("nan"))),
                "weighted_f1": float(m.get("weighted_f1", float("nan"))),
                "HATE_f1": float(m.get("HATE_f1", float("nan"))),
                "OFFENSIVE_f1": float(m.get("OFFENSIVE_f1", float("nan"))),
                "CLEAN_f1": float(m.get("CLEAN_f1", float("nan"))),
                "HATE_precision": float(m.get("HATE_precision", float("nan"))),
                "HATE_recall": float(m.get("HATE_recall", float("nan"))),
                "OFFENSIVE_precision": float(m.get("OFFENSIVE_precision", float("nan"))),
                "OFFENSIVE_recall": float(m.get("OFFENSIVE_recall", float("nan"))),
                "CLEAN_precision": float(m.get("CLEAN_precision", float("nan"))),
                "CLEAN_recall": float(m.get("CLEAN_recall", float("nan"))),
            }
        )
    per_lang_df = pd.DataFrame(per_lang_rows)
    per_lang_df.to_csv(out_dir / "per_language_metrics.csv", index=False, encoding="utf-8")

    lat_rows = []
    for key, values in (
        ("routing", np.asarray(per_text_routing_ms, dtype=float)),
        ("ai", np.asarray([ai_latency_ms] * len(rows), dtype=float)),
        ("fusion", np.asarray([r["latency_total_ms"] for r in rows], dtype=float)),
    ):
        lat_rows.append(
            {
                "stage": key,
                "n": len(values),
                "mean_ms": float(values.mean()),
                "p50_ms": float(np.median(values)),
                "p95_ms": float(np.quantile(values, 0.95)),
                "max_ms": float(values.max()),
                "throughput_per_sec": float(len(values) / (values.mean() / 1000.0)) if values.mean() > 0 else float("nan"),
            }
        )
    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(out_dir / "latency_summary.csv", index=False, encoding="utf-8")

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    report_lines.append(f"- total wall time (incl. load): {wall_ms:.0f} ms")

    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            import seaborn  # noqa: F401

            sns_cm = ax
        except Exception:  # pragma: no cover
            pass

    LOGGER.info("[e2e] wrote %s", csv_path)
    LOGGER.info("[e2e] wrote %s", json_path)
    LOGGER.info("[e2e] wrote %s", md_path)
    LOGGER.info("[e2e] wrote %s", out_dir / "per_language_metrics.csv")
    LOGGER.info("[e2e] wrote %s", out_dir / "latency_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())