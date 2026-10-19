"""Temporary: collect hashes + latency for the report (plan section 44)."""
import hashlib
import json
from pathlib import Path

FILES = [
    "data/processed/lexicon/verified_lexicon.csv",
    "data/processed/lexicon/sweeps/pruned_score008/verified_lexicon_pruned.csv",
    "data/processed/lexicon/sweeps/pruned_score010/verified_lexicon_pruned.csv",
    "data/processed/lexicon/sweeps/pruned_score015/verified_lexicon_pruned.csv",
    "configs/rule_engine_tuned.yaml",
    "configs/rule_engine_tuned_p010.yaml",
    "configs/rule_engine.yaml",
    "configs/rule_engine_pruned.yaml",
]
print("--- SHA256 (16 đầu) ---")
for f in FILES:
    p = Path(f)
    if not p.exists():
        print(f"  MISSING {f}")
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    n = ""
    if p.suffix == ".csv":
        n = f"  ({sum(1 for _ in p.open(encoding='utf-8')) - 1} rows)"
    print(f"  {h}  {f}{n}")

# split hashes if present
print("--- split files ---")
for f in sorted(Path("data/processed").glob("*/test.csv")) + sorted(Path("data/processed").glob("*.csv")):
    if f.name in ("test.csv", "validation.csv", "train.csv"):
        print(f"  {hashlib.sha256(f.read_bytes()).hexdigest()[:16]}  {f}")

print("\n--- latency (ms/comment) ---")
for name, d in [
    ("full 922", "outputs/evaluation/rule_engine_tuned"),
    ("pruned 576", "outputs/evaluation/rule_engine_tuned_p010"),
    ("pruned 691", "outputs/evaluation/rule_engine_tuned_p008"),
    ("pruned 324", "outputs/evaluation/rule_engine_tuned_p015"),
]:
    for sub in ("test", ""):
        p = Path(d) / sub / "latency.json"
        if p.exists():
            v = json.loads(p.read_text(encoding="utf-8"))
            print(f"  {name:11s} [{sub or 'val':4s}] mean={v['mean_ms']:.3f} p95={v['p95_ms']:.3f} "
                  f"throughput={v['throughput_per_sec']:.0f}/s")
            break
