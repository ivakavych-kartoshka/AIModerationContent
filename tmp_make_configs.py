"""Temporary: derive tuned+pruned configs from configs/rule_engine_tuned.yaml."""
from pathlib import Path

src = Path("configs/rule_engine_tuned.yaml").read_text(encoding="utf-8")

VARIANTS = {
    "p008": ("data/processed/lexicon/sweeps/pruned_score008/verified_lexicon_pruned.csv", 691, 0.008),
    "p010": ("data/processed/lexicon/sweeps/pruned_score010/verified_lexicon_pruned.csv", 576, 0.010),
    "p015": ("data/processed/lexicon/sweeps/pruned_score015/verified_lexicon_pruned.csv", 324, 0.015),
}

OLD = "lexicon_path: data/processed/lexicon/verified_lexicon.csv"
for name, (path, size, score) in VARIANTS.items():
    new = src.replace(
        "# Rule engine configuration - TUNED (Rule_Engine_Plan.md sections 37-39)",
        ("# Rule engine configuration - TUNED + PRUNED variant (lexicon-size ablation)\n"
         "#   identical to rule_engine_tuned.yaml EXCEPT lexicon_path (isolate lexicon size)\n"
         f"#   prune gate: tfidf-min-score={score} -> {size} terms (full = 922)"),
        1,
    ).replace(OLD, f"lexicon_path: {path}", 1)
    assert new != src and path in new, name
    out = Path(f"configs/rule_engine_tuned_{name}.yaml")
    out.write_text(new, encoding="utf-8")
    print(f"wrote {out} -> {path} ({size} terms)")
