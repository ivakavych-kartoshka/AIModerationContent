"""Step 63 - Download ViNLI (Vietnamese NLI, COLING 2022) for language-detection evaluation.

ViNLI: Huynh et al., "ViNLI: A Vietnamese Corpus for Studies on Open-Domain
Natural Language Inference", COLING 2022 (ACL Anthology 2022.coling-1.339).
HF mirror used here: presencesw/vinli_4_label (30,376 premise-hypothesis pairs).

Outputs:
  data/vinli/vinli_pairs.csv        - all 30,376 pairs (split, gold_label, sentence1, sentence2)
  data/vinli/vinli_sentences.csv    - deduplicated Vietnamese sentence pool
  data/vinli/vinli_summary.json     - size/stat summary vs Wortschatz
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_HOME", "C:/Users/Lenovo/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "vinli"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WORTSCHATZ_SUMMARY = PROJECT_ROOT / "outputs" / "evaluation" / "wortschatz" / "wortschatz_benchmark_summary.json"


def main() -> int:
    print("[1/4] Loading ViNLI (presencesw/vinli_4_label)...")
    from datasets import load_dataset

    ds = load_dataset("presencesw/vinli_4_label")
    rows = []
    for split in ("train", "dev", "test"):
        part = ds[split]
        for label, s1, s2 in zip(part["gold_label"], part["sentence1"], part["sentence2"]):
            rows.append(
                {
                    "split": split,
                    "gold_label": str(label),
                    "sentence1": str(s1).strip(),
                    "sentence2": str(s2).strip(),
                }
            )
    df_pairs = pd.DataFrame(rows)
    print(f"  {len(df_pairs)} pairs: " + ", ".join(f"{k}={v}" for k, v in df_pairs["split"].value_counts().items()))

    print("[2/4] Building deduplicated Vietnamese sentence pool...")
    pool = list(df_pairs["sentence1"]) + list(df_pairs["sentence2"])
    # Keep order-stable dedupe
    seen = set()
    sents = []
    for s in pool:
        if s and s not in seen:
            seen.add(s)
            sents.append(s)
    df_sents = pd.DataFrame({"text": sents, "lang": "vi", "source": "vinli"})
    print(f"  {len(df_pairs) * 2} sentence slots -> {len(df_sents)} unique sentences")

    print("[3/4] Writing CSVs...")
    pairs_path = DATA_DIR / "vinli_pairs.csv"
    sents_path = DATA_DIR / "vinli_sentences.csv"
    df_pairs.to_csv(pairs_path, index=False, encoding="utf-8")
    df_sents.to_csv(sents_path, index=False, encoding="utf-8")
    print(f"  saved {pairs_path}")
    print(f"  saved {sents_path}")

    print("[4/4] Writing summary...")
    wortschatz_n = None
    if WORTSCHATZ_SUMMARY.exists():
        ws = json.loads(WORTSCHATZ_SUMMARY.read_text(encoding="utf-8"))
        wortschatz_n = ws.get("total_rows")

    summary = {
        "dataset": "ViNLI (Vietnamese Natural Language Inference)",
        "paper": "Huynh, Van Nguyen, Nguyen. ViNLI. COLING 2022 (2022.coling-1.339)",
        "hf_source": "presencesw/vinli_4_label",
        "license_note": "Research use, per ViNLI paper",
        "lang": "vi",
        "domain": "online news (13 topics)",
        "num_pairs": int(len(df_pairs)),
        "num_unique_sentences": int(len(df_sents)),
        "pairs_by_split": {k: int(v) for k, v in df_pairs["split"].value_counts().items()},
        "labels": {k: int(v) for k, v in df_pairs["gold_label"].value_counts().items()},
        "mean_sentence_chars": round(float(df_sents["text"].str.len().mean()), 3),
        "median_sentence_chars": float(df_sents["text"].str.len().median()),
        "wortschatz_total_sentences": wortschatz_n,
        "size_ratio_vs_wortschatz": round(len(df_sents) / wortschatz_n, 4) if wortschatz_n else None,
    }
    summary_path = DATA_DIR / "vinli_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
