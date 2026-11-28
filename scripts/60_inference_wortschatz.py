"""Step 60 - run the frozen English backbone over the Wortschatz test split.

    python scripts/60_inference_wortschatz.py

Reuses ``CheckpointPredictor`` (same path as the production pipeline) so the
latency numbers are directly comparable to the mixed end-to-end evaluation.
Writes a per-sentence CSV + a JSON summary with timing and prediction stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.constants import ID2LABEL  # noqa: E402
from src.inference.predictor import CheckpointPredictor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/evaluation/wortschatz/wortschatz_split_test.csv",
        help="Wortschatz test CSV (id,sentence,...)",
    )
    parser.add_argument(
        "--en-checkpoint",
        default="outputs/reports/sensitiveai-en-stage2/run_001/best_model",
    )
    parser.add_argument("--head", default="standard", choices=["standard", "custom"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="outputs/evaluation/wortschatz")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap rows (0 = all)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(int(args.limit)).copy()
    texts = [str(t) for t in df["sentence"].tolist()]
    ids = [str(i) for i in df["id"].tolist()]
    print(f"[1/3] loaded {len(texts)} sentences from {args.input}")

    predictor = CheckpointPredictor(
        args.en_checkpoint,
        head=args.head,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(f"[2/3] running inference (batch_size={args.batch_size}, max_length={args.max_length})")

    t0 = time.perf_counter()
    probs = predictor.predict_proba(texts)
    elapsed = time.perf_counter() - t0

    probs = np.asarray(probs, dtype=float)
    preds = np.argmax(probs, axis=1)
    labels = [ID2LABEL[int(p)] for p in preds]

    out_df = df.copy()
    out_df["pred_class"] = labels
    out_df["prob_HATE"] = probs[:, 0]
    out_df["prob_OFFENSIVE"] = probs[:, 1]
    out_df["prob_CLEAN"] = probs[:, 2]
    out_df["confidence"] = probs.max(axis=1)

    out_path = out_dir / "predictions_sensitiveai-en-stage2.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  saved {out_path} ({len(out_df)} rows)")

    counts = Counter(labels)
    n = len(texts)
    summary = {
        "checkpoint": args.en_checkpoint,
        "head": args.head,
        "input": args.input,
        "n_samples": n,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "total_sec": round(elapsed, 3),
        "ms_per_text": round(1000.0 * elapsed / n, 3) if n else None,
        "texts_per_sec": round(n / elapsed, 2) if elapsed > 0 else None,
        "prediction_distribution": dict(counts),
        "prediction_fraction": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "mean_confidence": round(float(out_df["confidence"].mean()), 4) if n else None,
        "mean_prob_HATE": round(float(out_df["prob_HATE"].mean()), 4) if n else None,
        "mean_prob_OFFENSIVE": round(float(out_df["prob_OFFENSIVE"].mean()), 4) if n else None,
        "mean_prob_CLEAN": round(float(out_df["prob_CLEAN"].mean()), 4) if n else None,
        "output_file": str(out_path),
    }
    summary_path = out_dir / "inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[3/3] summary -> {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
