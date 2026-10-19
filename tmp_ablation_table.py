"""Temporary: validation-side ablation table (lexicon size vs macro-F1)."""
import json
from pathlib import Path

BASE = Path("outputs/evaluation")
ROWS = [
    ("full 922 (tuned)", 922, BASE / "rule_engine_tuned"),
    ("pruned 691", 691, BASE / "rule_engine_tuned_p008"),
    ("pruned 576", 576, BASE / "rule_engine_tuned_p010"),
    ("pruned 324", 324, BASE / "rule_engine_tuned_p015"),
]

print(f"{'variant':18s} {'terms':>5s} {'valF1':>7s} {'acc':>7s} "
      f"{'HATE':>6s} {'OFFEN':>6s} {'CLEAN':>6s}  thresholds")
data = []
for name, terms, d in ROWS:
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    th = m["thresholds"]
    print(f"{name:18s} {terms:5d} {m['macro_f1']:7.4f} {m['accuracy']:7.4f} "
          f"{m['HATE_f1']:6.4f} {m['OFFENSIVE_f1']:6.4f} {m['CLEAN_f1']:6.4f}  "
          f"H={th['HATE']} O={th['OFFENSIVE']} C={th['CLEAN']}")
    data.append({"variant": name, "terms": terms, "val_macro_f1": m["macro_f1"],
                 "val_accuracy": m["accuracy"], "HATE_f1": m["HATE_f1"],
                 "OFFENSIVE_f1": m["OFFENSIVE_f1"], "CLEAN_f1": m["CLEAN_f1"],
                 "thresholds": th, "n_lexicon_entries": m["n_lexicon_entries"]})

out = Path("outputs/evaluation/rule_engine_tuned_pruning_ablation_val.json")
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n-> wrote {out}")
