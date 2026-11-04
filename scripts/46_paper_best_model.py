#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Use selected model + rule engine for ICCIES paper"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.common.paths import ensure_dir, project_path

FUSION_DIR = project_path("outputs/evaluation/fusion")
RULE_DIR = project_path("outputs/evaluation/rule_engine")
OUT_ROOT = FUSION_DIR / "paper_selected_model"
ensure_dir(OUT_ROOT)

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sensitiveai-vi-custom", help="Model to focus on for paper")
    args = parser.parse_args(argv)

    BEST_MODEL = args.model
    print(f"[PAPER] Using selected model: {BEST_MODEL}")
    (OUT_ROOT / "selected_model.txt").write_text(BEST_MODEL, encoding="utf-8")

    print("[1/5] Running error analysis...")
    subprocess.run([sys.executable, "scripts/44_error_analysis_fusion.py", "--models", BEST_MODEL], check=False)

    base_err = FUSION_DIR / "error_analysis" / BEST_MODEL
    ai_wrong_fuse_right = pd.read_csv(base_err / "ai_wrong_fuse_right.csv")
    summary = load_json(base_err / "summary.json")

    print("[2/5] Analyzing patterns...")
    patterns = defaultdict(list)
    obf_dot = re.compile(r"[a-zA-Z\u00C0-\u1EF9].*[.\*\-_][a-zA-Z\u00C0-\u1EF9]")
    genz_list = ["vl", "vcl", "vkl", "vlon", "cl", "cc", "dm", "đm", "lol", "wibu", "ngu", "óc", "não", "nhục"]
    neg_list = ["không", "ko", "chẳng", "chả", "chưa", "đâu", "mày nghĩ", "tao sợ"]

    for i, row in ai_wrong_fuse_right.iterrows():
        t = str(row["text"]).lower()
        tagged = False
        if obf_dot.search(t) or re.search(r"[a-zA-Z\u00C0-\u1EF9]\s+[a-zA-Z\u00C0-\u1EF9]{1,2}\b", t):
            patterns["obfuscation"].append(i); tagged = True
        if not tagged:
            for g in genz_list:
                if g in t:
                    patterns["genz_slang"].append(i); tagged = True; break
        if not tagged:
            for n in neg_list:
                if n in t:
                    patterns["negation_irony"].append(i); tagged = True; break
        if not tagged and re.search(r"(.)\1{2,}", t):
            patterns["repetition"].append(i); tagged = True
        if not tagged:
            patterns["unclear_other"].append(i)

    pat_rows = []
    for k, v in patterns.items():
        pat_rows.append({"pattern": k, "count": len(v), "percent": round(len(v)/len(ai_wrong_fuse_right)*100,2) if len(ai_wrong_fuse_right)>0 else 0})
    pd.DataFrame(pat_rows).sort_values("count", ascending=False).to_csv(OUT_ROOT / f"pattern_analysis_{BEST_MODEL}.csv", index=False, encoding="utf-8-sig")
    with open(OUT_ROOT / f"pattern_analysis_{BEST_MODEL}.md", "w", encoding="utf-8") as f:
        f.write(f"# Pattern Analysis - {BEST_MODEL}\n\n")
        f.write(f"Total AI_wrong->Fuse_right: {len(ai_wrong_fuse_right)}\n\n")
        f.write("| Pattern | Count | % |\n|---|---|---|\n")
        for r in pat_rows:
            f.write(f"| {r['pattern']} | {r['count']} | {r['percent']}% |\n")

    print("[3/5] Computing rule coverage...")
    rule_df = pd.read_csv(RULE_DIR / "test" / "rule_scores.csv")
    total_test = len(rule_df)
    matched = ((rule_df["rule_hate_score"]>0.01) | (rule_df["rule_offensive_score"]>0.01)).sum()
    rule_map = rule_df.set_index("index")
    fixed_with_support = 0
    for _, row in ai_wrong_fuse_right.iterrows():
        idx = row["index"]
        if idx in rule_map.index:
            r = rule_map.loc[idx]
            if r["rule_hate_score"]>0.01 or r["rule_offensive_score"]>0.01:
                fixed_with_support += 1
    cov = {
        "model": BEST_MODEL,
        "total_test": int(total_test),
        "samples_with_rule_match": int(matched),
        "rule_coverage_percent": round(float(matched)/total_test*100,2),
        "ai_wrong_fixed_total": int(len(ai_wrong_fuse_right)),
        "ai_wrong_fixed_with_rule_support": int(fixed_with_support),
        "ai_wrong_fixed_with_rule_support_percent": round(float(fixed_with_support)/len(ai_wrong_fuse_right)*100,2) if len(ai_wrong_fuse_right)>0 else 0
    }
    json.dump(cov, open(OUT_ROOT / f"rule_coverage_{BEST_MODEL}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("[4/5] Generating case studies...")
    top5 = ai_wrong_fuse_right.head(5).copy()
    top5.to_csv(OUT_ROOT / f"case_studies_top5_{BEST_MODEL}.csv", index=False, encoding="utf-8-sig")
    with open(OUT_ROOT / f"case_studies_top5_{BEST_MODEL}.md", "w", encoding="utf-8") as f:
        f.write(f"# Case Studies (AI-only wrong -> Fusion correct) - {BEST_MODEL}\n\n")
        for i,(_,row) in enumerate(top5.iterrows(),1):
            f.write(f"## Case {i}\n\n")
            f.write(f"**Text:** {row['text']}\n\n")
            f.write(f"| Item | Prediction |\n|---|---|\n")
            f.write(f"| Ground Truth | {row['true_label']} |\n")
            f.write(f"| AI-only | {row['ai_pred']} (WRONG) |\n")
            f.write(f"| Fusion (hybrid) | {row['fuse_pred']} (CORRECT) |\n\n")
            f.write(f"**Rule evidence:** rule_hate={row['rule_hate']:.4f}, rule_offensive={row['rule_offensive']:.4f}\n\n---\n\n")

    print("[5/5] Writing summary...")
    with open(OUT_ROOT / f"PAPER_READY_SUMMARY_{BEST_MODEL}.md", "w", encoding="utf-8") as f:
        f.write("# Paper-ready summary for ICCIES\n\n")
        f.write(f"## Focus model\n- Model: **{BEST_MODEL}**\n- Primary fusion variant: **{summary.get('primary')}**\n\n")
        f.write(f"## Quantitative\n- Test size: {summary['n']}\n- AI-only correct: {summary['ai_correct']} ({summary['ai_correct']/summary['n']*100:.2f}%)\n- Fusion correct: {summary['fuse_correct']} ({summary['fuse_correct']/summary['n']*100:.2f}%)\n- AI_wrong -> Fuse_right: {summary['ai_wrong_fuse_right']}\n- AI_right -> Fuse_wrong: {summary['ai_right_fuse_wrong']}\n- Net gain: {summary['net_gain']}\n\n")
        f.write(f"## Explainability-oriented metrics\n- Rule coverage: {cov['rule_coverage_percent']}% of test samples\n- Among AI misses corrected by fusion: {cov['ai_wrong_fixed_with_rule_support_percent']}% had explicit rule support\n\n")
        f.write("## Suggested framing\nTitle suggestion:\n> *\"An Explainable Hybrid Approach for Vietnamese Hate Speech Detection: Combining Lexicon-based Rules with Transformer Models\"*\n\n")

    print(f"\nDone! Check {OUT_ROOT}")

if __name__ == "__main__":
    main()
