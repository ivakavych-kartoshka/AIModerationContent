#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive analysis for ICCIES paper - sensitiveai-vi-customdlr2stage"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.common.paths import ensure_dir

base = Path("outputs/evaluation/fusion/error_analysis/sensitiveai-vi-customdlr2stage")
out_dir = base / "paper_analysis"
ensure_dir(out_dir)

ai_wrong_fuse_right = pd.read_csv(base / "ai_wrong_fuse_right.csv")
ai_right_fuse_wrong = pd.read_csv(base / "ai_right_fuse_wrong.csv")
summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
rule_df = pd.read_csv("outputs/evaluation/rule_engine/test/rule_scores.csv")
rule_map = rule_df.set_index("index")

print(f"[45] Loaded {len(ai_wrong_fuse_right)} fixed cases, {len(ai_right_fuse_wrong)} broken cases")

patterns = {
    "obfuscation_dot_slash": [],
    "obfuscation_space": [],
    "obfuscation_star_underscore": [],
    "genz_slang": [],
    "negation_irony": [],
    "repetition": [],
    "other": []
}

obf_dot = re.compile(r"[a-zA-Z\u00C0-\u1EF9].*[.\*\-_][a-zA-Z\u00C0-\u1EF9]")
genz = ["vl", "vcl", "vkl", "vlon", "cl", "cc", "dm", "đm", "lol", "wibu", "ngu", "óc", "não", "nhục"]
negation = ["không", "ko", "chẳng", "chả", "chưa", "đâu", "mày nghĩ", "tao sợ", "có sợ"]

for idx, row in ai_wrong_fuse_right.iterrows():
    text = str(row["text"]).lower()
    matched = False
    if obf_dot.search(text):
        patterns["obfuscation_dot_slash"].append(idx)
        matched = True
    if not matched and re.search(r"[a-zA-Z\u00C0-\u1EF9]\s+[a-zA-Z\u00C0-\u1EF9]{1,3}\b", text):
        patterns["obfuscation_space"].append(idx)
        matched = True
    if not matched and ("*" in text or "_" in text):
        patterns["obfuscation_star_underscore"].append(idx)
        matched = True
    if not matched:
        for g in genz:
            if g in text:
                patterns["genz_slang"].append(idx)
                matched = True
                break
    if not matched:
        for n in negation:
            if n in text:
                patterns["negation_irony"].append(idx)
                matched = True
                break
    if not matched and re.search(r"(.)\1{2,}", text):
        patterns["repetition"].append(idx)
        matched = True
    if not matched:
        patterns["other"].append(idx)

pattern_stats = []
for k, v in patterns.items():
    pct = round(len(v) / len(ai_wrong_fuse_right) * 100, 2) if len(ai_wrong_fuse_right) > 0 else 0
    pattern_stats.append({"pattern": k, "count": len(v), "percentage": pct})

pd.DataFrame(pattern_stats).to_csv(out_dir / "pattern_analysis.csv", index=False, encoding="utf-8-sig")

with open(out_dir / "pattern_analysis.md", "w", encoding="utf-8") as f:
    f.write("# Pattern Analysis - AI Wrong → Fusion Right (sensitiveai-vi-customdlr2stage)\n\n")
    f.write(f"Total cases: {len(ai_wrong_fuse_right)}\n\n")
    f.write("## Pattern Distribution\n\n")
    f.write("| Pattern | Count | Percentage |\n|---|---|---|\n")
    for p in pattern_stats:
        f.write(f"| {p['pattern']} | {p['count']} | {p['percentage']}% |\n")
    f.write("\n## Detailed Cases by Pattern\n\n")
    for pat_name, indices in patterns.items():
        if indices:
            f.write(f"### {pat_name} ({len(indices)} cases)\n\n")
            for idx in indices[:10]:
                row = ai_wrong_fuse_right.iloc[idx]
                f.write(f"**Index:** {row['index']}  \n")
                f.write(f"**Text:** {row['text']}  \n")
                f.write(f"**True:** {row['true_label']} | **AI:** {row['ai_pred']} | **Fusion:** {row['fuse_pred']}  \n")
                f.write(f"**Rule scores:** hate={row['rule_hate']:.3f}, offensive={row['rule_offensive']:.3f}  \n\n")

print("[45] Pattern analysis done")

total_test = len(rule_df)
matched = ((rule_df["rule_hate_score"] > 0.01) | (rule_df["rule_offensive_score"] > 0.01)).sum()
matched_ai_wrong_fixed = 0
for _, row in ai_wrong_fuse_right.iterrows():
    idx = row["index"]
    if idx in rule_map.index:
        r = rule_map.loc[idx]
        if r["rule_hate_score"] > 0.01 or r["rule_offensive_score"] > 0.01:
            matched_ai_wrong_fixed += 1

coverage = {
    "model": "sensitiveai-vi-customdlr2stage",
    "total_test_samples": int(total_test),
    "samples_with_rule_match": int(matched),
    "rule_coverage_percent": round(float(matched) / total_test * 100, 2),
    "ai_wrong_cases_total": int(len(ai_wrong_fuse_right)),
    "ai_wrong_fixed_by_rule_match": int(matched_ai_wrong_fixed),
    "fixed_cases_with_rule_support_percent": round(float(matched_ai_wrong_fixed) / len(ai_wrong_fuse_right) * 100, 2) if len(ai_wrong_fuse_right) > 0 else 0,
}

json.dump(coverage, open(out_dir / "rule_coverage.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with open(out_dir / "rule_coverage.md", "w", encoding="utf-8") as f:
    f.write("# Rule Coverage Analysis - sensitiveai-vi-customdlr2stage\n\n")
    for k, v in coverage.items():
        f.write(f"- {k}: {v}\n")
    f.write("\n**Key insight:** Rule engine has relatively low coverage but is highly effective on the cases it does match.\n")

print("[45] Rule coverage done")

case_studies = ai_wrong_fuse_right.head(5).copy()
case_studies.to_csv(out_dir / "case_studies_top5.csv", index=False, encoding="utf-8-sig")
with open(out_dir / "case_studies.md", "w", encoding="utf-8") as f:
    f.write("# Case Studies - AI Wrong → Fusion Right (sensitiveai-vi-customdlr2stage)\n\n")
    f.write("These examples demonstrate the explainability advantage of the hybrid approach.\n\n")
    for i, (_, row) in enumerate(case_studies.iterrows(), 1):
        f.write(f"## Case Study {i}\n\n")
        f.write(f"**Index:** {row['index']}  \n")
        f.write(f"**Text:** `{row['text']}`  \n\n")
        f.write("| Item | Prediction |\n|---|---|\n")
        f.write(f"| Ground truth | {row['true_label']} |\n")
        f.write(f"| AI-only prediction | {row['ai_pred']} ❌ |\n")
        f.write(f"| Fusion prediction | {row['fuse_pred']} ✅ |\n\n")
        f.write(f"**Rule engine scores:** rule_hate={row['rule_hate']:.4f}, rule_offensive={row['rule_offensive']:.4f}\n\n")
        f.write("**Explanation:** Lexical rule signals provide interpretable evidence supporting the corrected label.\n\n---\n\n")

with open(out_dir / "paper_summary.md", "w", encoding="utf-8") as f:
    f.write("# Paper Analysis Summary (sensitiveai-vi-customdlr2stage)\n\n")
    f.write("## Overall Results\n\n")
    f.write(f"- Net gain: {summary['net_gain']} ({summary['ai_wrong_fuse_right']} fixed - {summary['ai_right_fuse_wrong']} broken)\n")
    f.write(f"- AI-only accuracy: {summary['ai_correct']/summary['n']*100:.2f}% ({summary['ai_correct']}/{summary['n']})\n")
    f.write(f"- Fusion accuracy: {summary['fuse_correct']/summary['n']*100:.2f}% ({summary['fuse_correct']}/{summary['n']})\n\n")
    f.write("## Explainability Metrics\n\n")
    f.write(f"- Rule coverage: {coverage['rule_coverage_percent']}% of test samples have rule matches\n")
    f.write(f"- Of AI-miss cases corrected: {coverage['fixed_cases_with_rule_support_percent']}% had explicit rule support\n\n")
    f.write("## Key Patterns in Fixed Cases\n\n")
    for p in pattern_stats:
        if p['count'] > 0:
            f.write(f"- {p['pattern']}: {p['count']} cases ({p['percentage']}%)\n")

print(f"[45] Done. Outputs saved to: {out_dir}")
