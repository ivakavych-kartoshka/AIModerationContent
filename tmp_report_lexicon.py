"""Temporary summary of the regenerated lexicon (tokenize + TF-IDF + labeling)."""
import collections
import csv
from pathlib import Path

p = Path("data/processed/lexicon/verified_lexicon.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
print("Tong entry:", len(rows))
print("Cot:", list(rows[0].keys()))

lc = collections.Counter(r["lexical_labels"] for r in rows)
print("--- Phan bo nhan ---")
for k, v in lc.most_common():
    print("  %s: %d" % (k or "(trong)", v))

rs = collections.Counter(r["review_status"] for r in rows)
print("--- review_status ---")
for k, v in rs.most_common():
    print("  %s: %d" % (k, v))

# severity breakdown for pure labels
sev = collections.Counter((r["lexical_labels"], r["severity"]) for r in rows)
print("--- nhan x severity ---")
for k, v in sev.most_common(8):
    print("  %s: %d" % (k, v))

print("--- 15 vi du dau tien ---")
for r in rows[:15]:
    print("  %-24s %-16s sev=%s conf=%s %s" % (
        r["term"][:24], r["lexical_labels"], r["severity"],
        r["confidence"][:5], r["review_status"]))
