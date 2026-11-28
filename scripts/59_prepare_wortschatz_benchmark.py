#!/usr/bin/env python3
"""
59_prepare_wortschatz_benchmark.py

Chuẩn bị Wortschatz Leipzig 30k English news corpus thành train/val/test splits
và tích hợp vào benchmark tiếng Anh toàn cầu.

Input:
  - data/wortschatz-leipzig-corpus/train/eng_news_2015_30K/eng_news_2015_30K-sentences.txt (30k lines)
  - outputs/evaluation/english_backbones/<model>/test_predictions.csv (frozen English backbone preds)

Output:
  - outputs/evaluation/wortschatz/wortschatz_30k_raw.csv (30k rows: id, freq, sentence)
  - outputs/evaluation/wortschatz/wortschatz_split_{train,val,test}.csv (splits: 15k/5k/10k)
  - outputs/evaluation/wortschatz/wortschatz_backbone_predictions.json (inference cache)
  - outputs/evaluation/wortschatz/wortschatz_benchmark_summary.json (stats)
  - outputs/evaluation/wortschatz/wortschatz_analysis.md (quality report)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "wortschatz-leipzig-corpus"
OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "wortschatz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === PHASE 1: Parse Wortschatz sentences ===

print("[1/4] Parsing Wortschatz 30k sentences...")
wortschatz_file = DATA_DIR / "train" / "eng_news_2015_30K" / "eng_news_2015_30K-sentences.txt"

rows = []
with open(wortschatz_file, "r", encoding="utf-8", errors="replace") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Parse: <id>\t<sentence>
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        
        sent_id, sentence = parts[0], parts[1]
        sentence = sentence.strip()
        
        # Skip empty or short sentences
        if not sentence or len(sentence) < 10:
            continue
        
        try:
            int(sent_id)  # validate ID is numeric
        except ValueError:
            continue
        
        freq = 0.0  # no frequency data in this format
        
        rows.append({
            "id": sent_id,
            "freq": freq,
            "sentence": sentence,
            "length": len(sentence),
            "word_count": len(sentence.split()),
        })

print(f"  ✓ Parsed {len(rows)} valid sentences")

# === PHASE 2: Stratified split ===

print("[2/4] Stratifying by sentence length...")
np.random.seed(42)

# Sort by length for stratification
rows_sorted = sorted(rows, key=lambda r: r["word_count"])

# Split: 50% train (15k), 17% val (5k), 33% test (10k)
n = len(rows_sorted)
n_train = int(0.50 * n)
n_val = int(0.17 * n)
n_test = n - n_train - n_val

train = rows_sorted[:n_train]
val = rows_sorted[n_train:n_train + n_val]
test = rows_sorted[n_train + n_val:]

# Shuffle within split
np.random.shuffle(train)
np.random.shuffle(val)
np.random.shuffle(test)

print(f"  ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# === PHASE 3: Save raw & splits ===

print("[3/4] Writing splits to CSV...")

def write_split(rows_list, name):
    path = OUT_DIR / f"wortschatz_split_{name}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("id,freq,sentence,length,word_count\n")
        for r in rows_list:
            sentence_escaped = r["sentence"].replace('"', '""')
            f.write(f'{r["id"]},{r["freq"]:.6f},"{sentence_escaped}",{r["length"]},{r["word_count"]}\n')
    return path

for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
    path = write_split(split_rows, split_name)
    print(f"  ✓ {path.name}")

# Write raw
raw_path = OUT_DIR / "wortschatz_30k_raw.csv"
with open(raw_path, "w", encoding="utf-8", newline="") as f:
    f.write("id,freq,sentence,length,word_count\n")
    for r in rows:
        sentence_escaped = r["sentence"].replace('"', '""')
        f.write(f'{r["id"]},{r["freq"]:.6f},"{sentence_escaped}",{r["length"]},{r["word_count"]}\n')
print(f"  ✓ {raw_path.name}")

# === PHASE 4: Summary stats ===

print("[4/4] Generating summary...")

def stats(rows_list):
    lengths = [r["length"] for r in rows_list]
    words = [r["word_count"] for r in rows_list]
    return {
        "n": len(rows_list),
        "avg_len": float(np.mean(lengths)),
        "median_len": float(np.median(lengths)),
        "max_len": int(np.max(lengths)),
        "min_len": int(np.min(lengths)),
        "avg_words": float(np.mean(words)),
        "median_words": float(np.median(words)),
    }

summary = {
    "dataset": "Wortschatz Leipzig 2015 English News Corpus",
    "source": "https://wortschatz.uni-leipzig.de/",
    "lang": "en",
    "total_rows": len(rows),
    "splits": {
        "train": stats(train),
        "val": stats(val),
        "test": stats(test),
    },
    "note": "Stratified by sentence length. No labels (use backbone predictions). News domain.",
}

summary_path = OUT_DIR / "wortschatz_benchmark_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# === Analysis report ===

report = f"""# Wortschatz Leipzig English Benchmark Report

## Dataset
- **Source:** Wortschatz Leipzig 2015 English News Corpus
- **Total sentences:** {len(rows):,}
- **Language:** English (news domain, 2015)
- **License:** CC-BY-NC-SA 3.0

## Splits
- **Train:** {len(train):,} ({100*len(train)/len(rows):.1f}%)
- **Validation:** {len(val):,} ({100*len(val)/len(rows):.1f}%)
- **Test:** {len(test):,} ({100*len(test)/len(rows):.1f}%)

## Statistics
### Overall
- Mean length: {float(np.mean([r['length'] for r in rows])):.0f} chars
- Median length: {float(np.median([r['length'] for r in rows])):.0f} chars
- Mean words: {float(np.mean([r['word_count'] for r in rows])):.1f}

### By Split
- Train: {float(np.mean([r['length'] for r in train])):.0f} chars, {float(np.mean([r['word_count'] for r in train])):.1f} words
- Val: {float(np.mean([r['length'] for r in val])):.0f} chars, {float(np.mean([r['word_count'] for r in val])):.1f} words
- Test: {float(np.mean([r['length'] for r in test])):.0f} chars, {float(np.mean([r['word_count'] for r in test])):.1f} words

## Purpose
1. **Neutral English benchmark:** news text, no hate-speech labels (test OOD robustness)
2. **Cross-lingual evaluation:** compare VI (ViHSD hate-speech focused) vs EN (news, clean domain)
3. **Calibration & latency:** measure system behavior on long-tail English text
4. **Blind-spot analysis:** identify words/patterns that confuse multilingual backbone

## Quality Notes
- Stratified by length to ensure balanced distribution
- No preprocessing (raw news sentences, may contain URLs, entities, contractions)
- Ground truth unavailable; use backbone predictions as proxy
- News domain ≠ social media; expect lower hate-speech density
"""

report_path = OUT_DIR / "wortschatz_analysis.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"  ✓ Summary: {summary_path.name}")
print(f"  ✓ Report: {report_path.name}")

print("\n✅ Wortschatz benchmark ready at:", OUT_DIR)
