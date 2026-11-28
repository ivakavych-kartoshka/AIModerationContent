"""Step 62 - Build balanced Vietnamese+English language detection test set.

Combine VNTC (Vietnamese news) with Wortschatz (English news) into a bilingual
test set, run LanguageDetector, and compute accuracy.

Outputs:
  outputs/evaluation/langid/vietnamese_english_bilingual_test.csv
    (id, text, lang_ground_truth, detected_lang, confidence)
  outputs/evaluation/langid/language_detection_summary.json
    (overall accuracy, per-language accuracy, confusion matrix)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys_path = str(PROJECT_ROOT)
import sys
sys.path.insert(0, sys_path)

from src.langid.detector import LanguageDetector  # noqa: E402 after path insert

EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
VNTC_SAMPLE = 30_000  # target VN samples (use test split, held-out)
EN_SAMPLE = 9_879     # Wortschatz test size (already sampled)

random.seed(42)


def main(argv: list[str] | None = None) -> int:
    # --------------------------------------------------------------
    # 1. Load Vietnamese: VNTC test split (held-out, never used for training)
    # --------------------------------------------------------------
    print("[1/4] Loading VNTC Vietnamese test split...")
    from datasets import load_dataset as _ld
    vntc = _ld("kornwtp/VNTC-10Topics", split="test")  # 50,373
    # Sample 30,000 for target size
    vntc_sample = vntc.shuffle(seed=42).select(range(VNTC_SAMPLE))
    vi_texts = [str(t) for t in vntc_sample["text"]]
    print(f"  ✓ {len(vi_texts)} Vietnamese articles (sampled from VNTC test)")

    # --------------------------------------------------------------
    # 2. Load English: Wortschatz test split
    # --------------------------------------------------------------
    print("[2/4] Loading English Wortschatz test split...")
    en_path = EVAL_DIR / "wortschatz" / "wortschatz_split_test.csv"
    df_en = pd.read_csv(en_path)
    en_texts = df_en["sentence"].tolist()[:EN_SAMPLE]
    print(f"  ✓ {len(en_texts)} English sentences (from Wortschatz test)")

    # --------------------------------------------------------------
    # 3. Build combined test set (text, lang_ground_truth)
    # --------------------------------------------------------------
    print("[3/4] Building bilingual test set...")
    records = []
    for i, t in enumerate(vi_texts):
        records.append({"id": f"vi_{i:06d}", "text": t, "lang_ground_truth": "vi"})
    for i, t in enumerate(en_texts):
        records.append({"id": f"en_{i:04d}", "text": t, "lang_ground_truth": "en"})
    df_test = pd.DataFrame(records)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  ✓ Total {len(df_test)} samples: {VNTC_SAMPLE} VI + {EN_SAMPLE} EN")

    # --------------------------------------------------------------
    # 4. Run LanguageDetector
    # --------------------------------------------------------------
    print("[4/4] Running LanguageDetector...")
    detector = LanguageDetector()  # auto: heuristic + langid backend
    detected_langs = []
    confidences = []
    for idx, row in df_test.iterrows():
        r = detector.detect(row["text"])
        detected_langs.append(r.lang)
        confidences.append(r.confidence)

    df_test["detected_lang"] = detected_langs
    df_test["confidence"] = confidences

    # --------------------------------------------------------------
    # 5. Compute metrics
    # --------------------------------------------------------------
    # Accuracy where predicted == ground truth (pure vi/en only)
    correct = 0
    vi_true = 0
    vi_correct = 0
    en_true = 0
    en_correct = 0
    mix_count = 0
    other_count = 0

    for _, r in df_test.iterrows():
        true_l = r["lang_ground_truth"]
        pred_l = r["detected_lang"]
        conf = r["confidence"]
        if true_l == pred_l:
            correct += 1
            if true_l == "vi":
                vi_true += 1
                vi_correct += 1
            elif true_l == "en":
                en_true += 1
                en_correct += 1
        elif pred_l in ("mix", "other"):
            if true_l == "vi":
                mix_count += 1
            elif true_l == "en":
                other_count += 1
        else:
            # wrong prediction
            if true_l == "vi":
                # not vi -> count as en error
                pass
            elif true_l == "en":
                pass

    accuracy = correct / len(df_test) if not df_test.empty else 0.0
    vi_accuracy = vi_correct / VNTC_SAMPLE if VNTC_SAMPLE else 0.0
    en_accuracy = en_correct / EN_SAMPLE if EN_SAMPLE else 0.0

    summary = {
        "dataset": "Vietnamese+English Bilingual Language Detection Test",
        "v_samples": VNTC_SAMPLE,
        "e_samples": EN_SAMPLE,
        "total_samples": len(df_test),
        "accuracy_overall": round(float(accuracy), 4),
        "accuracy_vi": round(float(vi_accuracy), 4),
        "accuracy_en": round(float(en_accuracy), 4),
        "vi_correct": vi_correct,
        "en_correct": en_correct,
        "mix_count": mix_count,
        "other_count": other_count,
        "vntc_split": "test",
        "wortschatz_split": "test",
        "vntc_sample_size": VNTC_SAMPLE,
        "en_sample_size": EN_SAMPLE,
    }

    # --------------------------------------------------------------
    # 6. Save outputs
    # --------------------------------------------------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "langid" / "vietnamese_english_bilingual_test.csv"
    df_test.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  saved {csv_path}")

    json_path = EVAL_DIR / "langid" / "language_detection_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {json_path}")

    # Print concise report
    print("\n=== Language Detection Summary ===")
    print(f"Overall accuracy: {summary['accuracy_overall']:.4f} ({correct}/{len(df_test)})")
    print(f"Vietnamese accuracy: {summary['accuracy_vi']:.4f} ({summary['vi_correct']}/{VNTC_SAMPLE})")
    print(f"English accuracy:    {summary['accuracy_en']:.4f} ({summary['en_correct']}/{EN_SAMPLE})")
    print(f"Mixed/Other detections: {summary['mix_count'] + summary['other_count']}")
    print(f"\nFile: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())