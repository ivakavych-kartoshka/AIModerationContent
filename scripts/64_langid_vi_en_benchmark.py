"""Step 64 - Language detection benchmark: ViNLI (VI) vs Wortschatz (EN).

Runs LanguageDetector.detect_batch on matched samples and reports accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.langid.detector import LanguageDetector  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
LANGID_DIR = EVAL_DIR / "langid"
LANGID_DIR.mkdir(parents=True, exist_ok=True)

# Target sample size — match Wortschatz total (29,934)
TARGET_N = 29_934


def load_vi_sentences() -> list[str]:
    """Load ViNLI unique sentences, return up to TARGET_N."""
    path = PROJECT_ROOT / "data" / "vinli" / "vinli_sentences.csv"
    df = pd.read_csv(path)
    texts = df["text"].tolist()
    return texts[:TARGET_N]


def load_en_sentences() -> list[str]:
    """Load Wortschatz all sentences (train+val+test combined)."""
    base = EVAL_DIR / "wortschatz"
    parts = []
    for fn in ("wortschatz_split_train.csv", "wortschatz_split_val.csv", "wortschatz_split_test.csv"):
        df = pd.read_csv(base / fn)
        parts.append(df["sentence"].tolist())
    texts = [t for part in parts for t in part]
    return texts[:TARGET_N]


def evaluate_batch(texts: list[str], true_lang: str, detector: LanguageDetector) -> tuple[int, int]:
    """Run detect_batch and count correct predictions."""
    results = detector.detect_batch(texts)
    correct = sum(1 for r in results if r.lang == true_lang)
    return correct, len(texts)


def main() -> int:
    print("[1/3] Loading sentences...")
    vi_texts = load_vi_sentences()
    en_texts = load_en_sentences()
    print(f"  VI: {len(vi_texts)}")
    print(f"  EN: {len(en_texts)}")

    print("[2/3] Running LanguageDetector.detect_batch...")
    detector = LanguageDetector()

    print("  Vietnamese batch...")
    vi_correct, vi_total = evaluate_batch(vi_texts, "vi", detector)

    print("  English batch...")
    en_correct, en_total = evaluate_batch(en_texts, "en", detector)

    # Combined metrics
    total_correct = vi_correct + en_correct
    total_samples = vi_total + en_total
    overall_acc = total_correct / total_samples
    vi_acc = vi_correct / vi_total
    en_acc = en_correct / en_total

    summary = {
        "dataset_vi": "ViNLI (30,376 pairs, 34,021 unique sents)",
        "dataset_en": "Wortschatz Leipzig 2015 English News (29,934 sentences)",
        "sample_size_per_lang": TARGET_N,
        "vi_correct": int(vi_correct),
        "vi_total": int(vi_total),
        "vi_accuracy": round(vi_acc, 4),
        "en_correct": int(en_correct),
        "en_total": int(en_total),
        "en_accuracy": round(en_acc, 4),
        "overall_correct": int(total_correct),
        "overall_total": int(total_samples),
        "overall_accuracy": round(overall_acc, 4),
    }

    out_json = LANGID_DIR / "langid_vi_en_benchmark.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[3/3] Saved {out_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())