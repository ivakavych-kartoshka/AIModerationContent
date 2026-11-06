"""Step 62b - Quick test: build Vietnamese+English bilingual language detection test set.

Optimized: use batch detection + sample 5k VI + 2k EN for speed validation.
Once validated, can scale to full 30k+9.9k.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.langid.detector import LanguageDetector  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"

# Quick test sizes (scale down for validation)
VNTC_SAMPLE = 5_000   # test 5k VI (not 30k)
EN_SAMPLE = 2_000     # test 2k EN (not 9.9k)


def main(argv: list[str] | None = None) -> int:
    # 1. Load Vietnamese: VNTC test split (sample 5k)
    print("[1/4] Loading VNTC Vietnamese test split (5k sample)...")
    from datasets import load_dataset as _ld
    vntc = _ld("kornwtp/VNTC-10Topics", split="test", trust_remote_code=True)
    vntc_sample = vntc.shuffle(seed=42).select(range(VNTC_SAMPLE))
    vi_texts = [str(t) for t in vntc_sample["text"]]
    print(f"  ✓ {len(vi_texts)} Vietnamese articles")

    # 2. Load English: Wortschatz test split (sample 2k)
    print("[2/4] Loading Wortschatz English test split (2k sample)...")
    en_path = EVAL_DIR / "wortschatz" / "wortschatz_split_test.csv"
    df_en = pd.read_csv(en_path)
    en_texts = df_en["sentence"].tolist()[:EN_SAMPLE]
    print(f"  ✓ {len(en_texts)} English sentences")

    # 3. Build combined test set
    print("[3/4] Building bilingual test set...")
    records = []
    for i, t in enumerate(vi_texts):
        records.append({"id": f"vi_{i:05d}", "text": t, "lang_ground_truth": "vi"})
    for i, t in enumerate(en_texts):
        records.append({"id": f"en_{i:04d}", "text": t, "lang_ground_truth": "en"})
    df_test = pd.DataFrame(records)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  ✓ Total {len(df_test)} samples: {VNTC_SAMPLE} VI + {EN_SAMPLE} EN")

    # 4. Run LanguageDetector batch mode
    print("[4/4] Running LanguageDetector (batch mode)...")
    detector = LanguageDetector()
    texts_list = df_test["text"].tolist()
    results_batch = detector.detect_batch(texts_list)
    
    detected_langs = [r.lang for r in results_batch]
    confidences = [r.confidence for r in results_batch]

    df_test["detected_lang"] = detected_langs
    df_test["confidence"] = confidences

    # 5. Compute metrics
    correct = sum(1 for _, r in df_test.iterrows() if r["lang_ground_truth"] == r["detected_lang"])
    
    vi_rows = df_test[df_test["lang_ground_truth"] == "vi"]
    en_rows = df_test[df_test["lang_ground_truth"] == "en"]
    
    vi_correct = sum(1 for _, r in vi_rows.iterrows() if r["detected_lang"] == "vi")
    en_correct = sum(1 for _, r in en_rows.iterrows() if r["detected_lang"] == "en")
    
    accuracy = correct / len(df_test)
    vi_accuracy = vi_correct / len(vi_rows) if len(vi_rows) > 0 else 0.0
    en_accuracy = en_correct / len(en_rows) if len(en_rows) > 0 else 0.0

    summary = {
        "test_type": "quick_validation",
        "vi_samples": VNTC_SAMPLE,
        "en_samples": EN_SAMPLE,
        "total_samples": len(df_test),
        "accuracy_overall": round(float(accuracy), 4),
        "accuracy_vi": round(float(vi_accuracy), 4),
        "accuracy_en": round(float(en_accuracy), 4),
        "vi_correct": int(vi_correct),
        "en_correct": int(en_correct),
        "vntc_split": "test",
        "wortschatz_split": "test",
    }

    # 6. Save outputs
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "langid").mkdir(parents=True, exist_ok=True)
    
    csv_path = EVAL_DIR / "langid" / "vietnamese_english_bilingual_test_5k2k.csv"
    df_test.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  saved {csv_path}")

    json_path = EVAL_DIR / "langid" / "language_detection_summary_5k2k.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {json_path}")

    # Print report
    print("\n=== Language Detection Summary (Quick Test) ===")
    print(f"Overall accuracy: {summary['accuracy_overall']:.4f} ({correct}/{len(df_test)})")
    print(f"Vietnamese accuracy: {summary['accuracy_vi']:.4f} ({summary['vi_correct']}/{VNTC_SAMPLE})")
    print(f"English accuracy:    {summary['accuracy_en']:.4f} ({summary['en_correct']}/{EN_SAMPLE})")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
