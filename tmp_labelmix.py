"""Temporary: label/review mix per lexicon variant."""
import csv
import collections

for n, p in [
    ("922", "data/processed/lexicon/verified_lexicon.csv"),
    ("691", "data/processed/lexicon/sweeps/pruned_score008/verified_lexicon_pruned.csv"),
    ("576", "data/processed/lexicon/sweeps/pruned_score010/verified_lexicon_pruned.csv"),
    ("324", "data/processed/lexicon/sweeps/pruned_score015/verified_lexicon_pruned.csv"),
]:
    r = list(csv.DictReader(open(p, encoding="utf-8")))
    c = collections.Counter(x["lexical_labels"] for x in r)
    rs = collections.Counter(x["review_status"] for x in r)
    print(f"{n:>4} | total={len(r):4d} | HATE={c.get('HATE',0):4d} "
          f"H;O={c.get('HATE;OFFENSIVE',0):4d} OFF={c.get('OFFENSIVE',0):4d} "
          f"| verified={rs.get('verified',0):4d} auto={rs.get('auto',0):4d}")
