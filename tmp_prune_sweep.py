"""Temporary: how many terms survive the prune gate at stricter settings?"""
import csv
from pathlib import Path

rows = list(csv.DictReader(
    Path("data/processed/lexicon/pruning/lexicon_prune_audit.csv").open(encoding="utf-8")))
n = len(rows)
scores = sorted(float(r["tfidf_score"]) for r in rows)
ratios = sorted(float(r["tfidf_ratio"]) for r in rows)


def pct(v, p):
    return v[min(len(v) - 1, int(p / 100 * len(v)))]


print(f"n = {n}")
print("tfidf_score quantiles: " +
      " ".join(f"p{p}={pct(scores, p):.4f}" for p in (10, 25, 50, 75, 90)))
print("tfidf_ratio quantiles: " +
      " ".join(f"p{p}={pct(ratios, p):.3f}" for p in (10, 25, 50, 75, 90)))

print("\n--- gate: ratio >= 0.60 (giữ nguyên), siết SCORE ---")
for s in (0.004, 0.005, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050):
    k = sum(1 for r in rows
            if float(r["tfidf_ratio"]) >= 0.6 and float(r["tfidf_score"]) >= s)
    print(f"  min_score={s:<6} -> kept {k:4d}/{n}  ({100*k/n:5.1f}% , drop {100*(n-k)/n:4.1f}%)")

print("\n--- gate: score >= 0.004 (giữ nguyên), siết RATIO ---")
for q in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
    k = sum(1 for r in rows
            if float(r["tfidf_ratio"]) >= q and float(r["tfidf_score"]) >= 0.004)
    print(f"  min_ratio={q:<5} -> kept {k:4d}/{n}  ({100*k/n:5.1f}% , drop {100*(n-k)/n:4.1f}%)")

print("\n--- kết hợp (score, ratio) ---")
for s, q in ((0.006, 0.7), (0.008, 0.7), (0.010, 0.7), (0.010, 0.8),
             (0.015, 0.8), (0.020, 0.85), (0.020, 0.9), (0.030, 0.9)):
    k = sum(1 for r in rows
            if float(r["tfidf_ratio"]) >= q and float(r["tfidf_score"]) >= s)
    print(f"  score>={s:<6} ratio>={q:<5} -> kept {k:4d}/{n}  ({100*k/n:5.1f}%)")

# label mix at the strictest sensible gates, to see what gets dropped
print("\n--- thành phần bị BỎ ở score>=0.010 & ratio>=0.70 ---")
from collections import Counter
dropped = [r for r in rows
           if not (float(r["tfidf_ratio"]) >= 0.7 and float(r["tfidf_score"]) >= 0.010)]
print("  dropped:", len(dropped))
print("  labels:", dict(Counter(r["lexical_labels"] for r in dropped)))
print("  sample:", [r["term"] for r in dropped[:20]])
