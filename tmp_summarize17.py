"""Temporary: summarize rule-engine eval (script 17) across configs + splits."""
import json
from pathlib import Path

BASE = Path("outputs/evaluation")
RUNS = [
    ("baseline", BASE / "rule_engine"),
    ("tuned", BASE / "rule_engine_tuned"),
    ("pruned", BASE / "rule_engine_pruned"),
]

rows = []
for name, d in RUNS:
    print("=" * 72)
    print(f"{name}   [{d}]")
    for label, path in (("validation", d / "metrics.json"),
                        ("test", d / "test" / "metrics.json")):
        if not path.exists():
            print(f"  {label:11s} MISSING {path}")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        print(f"  {label:11s} macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
              f"weighted={m['weighted_f1']:.4f}  macroAUC={m.get('macro_roc_auc', 0):.4f}")
        for cls in ("HATE", "OFFENSIVE", "CLEAN"):
            print(f"              {cls:9s} P={m[f'{cls}_precision']:.4f} R={m[f'{cls}_recall']:.4f} "
                  f"F1={m[f'{cls}_f1']:.4f} sup={m[f'{cls}_support']}")
        print(f"              thresholds={m['thresholds']}  obj={m['objective']}  "
              f"lexicon={m['n_lexicon_entries']} ({m['lexicon']})")
        print(f"              cm={m['confusion_matrix']}")
        if label == "test":
            rows.append((name, m["macro_f1"], m["accuracy"],
                         m["HATE_f1"], m["OFFENSIVE_f1"], m["CLEAN_f1"]))

# latency
print("=" * 72)
print("LATENCY (ms/comment) — test")
for name, d in RUNS:
    p = d / "test" / "latency.json"
    if p.exists():
        lat = json.loads(p.read_text(encoding="utf-8"))
        print(f"  {name:10s} {lat}")

print("=" * 72)
print(f"{'config':10s} {'macro-F1':>9s} {'acc':>7s} {'HATE':>7s} {'OFFEN':>7s} {'CLEAN':>7s}")
for r in rows:
    print(f"{r[0]:10s} {r[1]:9.4f} {r[2]:7.4f} {r[3]:7.4f} {r[4]:7.4f} {r[5]:7.4f}")
