"""Temporary: false positives in quoted / negated / context_required contexts.

Runs the rule engine on the validation split for two configs and splits the
CLEAN false positives by the context signals the scorer attached
(`quoted`, `negated`, `context_required`). Plan section 44 asks for exactly
this measurement ("do false positives o ngu canh trich dan, phu dinh, giao duc").
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.metrics import apply_decision_rule  # noqa: E402
from src.evaluation.thresholds import load_thresholds  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402

SPLIT = Path("data/splits/validation.csv")
CONFIGS = [
    ("baseline", "configs/rule_engine.yaml", Path("outputs/evaluation/rule_engine/thresholds.json")),
    ("tuned", "configs/rule_engine_tuned.yaml", Path("outputs/evaluation/rule_engine_tuned/thresholds.json")),
]

rows = list(csv.DictReader(SPLIT.open(encoding="utf-8")))
texts = [r["text"] for r in rows]
y_true = [int(r["label"]) for r in rows]
CLEAN = 2

for name, cfg_path, th_path in CONFIGS:
    cfg = load_rule_engine_config(cfg_path)
    engine = RuleEngine.from_lexicon(cfg.lexicon_path, cfg)
    thresholds = load_thresholds(th_path)
    hate, off = [], []
    for t in texts:
        r = engine.analyze(t)
        hate.append(float(r["rule_hate_score"]))
        off.append(float(r["rule_offensive_score"]))
    import numpy as np

    proba = np.stack([hate, off, np.clip(1.0 - np.maximum(hate, off), 0, 1)], axis=1)
    pred = apply_decision_rule(proba, "thresholds", thresholds)

    fp_total = 0
    by_signal = Counter()
    fp_texts = {s: [] for s in ("negated", "quoted", "context_required")}
    for i, (p, y) in enumerate(zip(pred, y_true)):
        if y != CLEAN or p == CLEAN:
            continue
        fp_total += 1
        result = engine.analyze(texts[i])
        signals = set()
        for m in result["matches"]:
            signals.update(m.get("signals") or [])
        hit = signals & {"negated", "quoted", "context_required"}
        if hit:
            for s in hit:
                by_signal[s] += 1
                if len(fp_texts[s]) < 3:
                    fp_texts[s].append(texts[i][:90])
        else:
            by_signal["none"] += 1

    n_clean = sum(1 for y in y_true if y == CLEAN)
    print(f"\n=== {name} ({cfg_path}) ===")
    print(f"CLEAN FPs: {fp_total}/{n_clean} ({100 * fp_total / n_clean:.2f}%)")
    for k, v in by_signal.most_common():
        print(f"  {k:20s} {v}")
    for s, examples in fp_texts.items():
        if examples:
            print(f"  sample [{s}]: {examples[0]!r}")
