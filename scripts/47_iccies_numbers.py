#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ICCIES 2027 paper numbers.

Recomputes every quantitative claim used in ``papers/iccies2027`` from the
frozen artefacts and writes a single machine-readable bundle plus a Markdown
report, so the paper can be regenerated instead of trusted.

Important: fusion consumed the *tuned* rule engine
(``outputs/evaluation/rule_engine_tuned``), not the untuned
``outputs/evaluation/rule_engine``.  Earlier ad-hoc analyses used the untuned
scores and therefore over-reported rule coverage (51.45% instead of 15.75%).
This script is the single source of truth for the paper.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUSION = ROOT / "outputs/evaluation/fusion"
RULE_TUNED = ROOT / "outputs/evaluation/rule_engine_tuned"
OUT = FUSION / "iccies2027"
OUT.mkdir(parents=True, exist_ok=True)

FOCUS = "sensitiveai-vi-custom"
MODELS = [
    "sensitiveai-vi",
    "sensitiveai-vi-custom",
    "sensitiveai-vi-customdlr2stage",
    "sensitiveai-vi-custom-curriculum",
]
EPS = 0.01


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def rule_path() -> Path:
    p = RULE_TUNED / "test" / "rule_scores.csv"
    return p if p.exists() else RULE_TUNED / "rule_scores.csv"


def main() -> int:
    numbers: dict = {"focus_model": FOCUS, "epsilon": EPS, "models": MODELS}

    # ---------------- component metrics ----------------
    fusion_metrics = {m: jload(FUSION / m / "metrics.json") for m in MODELS}
    for m in MODELS:
        j = fusion_metrics[m]
        prim = j["primary_variant"]
        numbers.setdefault("backbones", {})[m] = {
            "primary_variant": prim,
            "alpha": j.get("alpha"),
            "ai_macro_f1": round(j["ai_only_reference"]["macro_f1"], 4),
            "ai_accuracy": round(j["ai_only_reference"]["accuracy"], 4),
            "fusion_macro_f1": round(j["test"][prim]["macro_f1"], 4),
            "fusion_accuracy": round(j["test"][prim]["accuracy"], 4),
            "delta_macro_f1": round(
                j["test"][prim]["macro_f1"] - j["ai_only_reference"]["macro_f1"], 4
            ),
        }

    focus = fusion_metrics[FOCUS]
    numbers["fusion_variants"] = {
        k: {
            "macro_f1": round(v["macro_f1"], 4),
            "accuracy": round(v["accuracy"], 4),
            "HATE_f1": round(v["HATE_f1"], 4),
            "OFFENSIVE_f1": round(v["OFFENSIVE_f1"], 4),
            "CLEAN_f1": round(v["CLEAN_f1"], 4),
        }
        for k, v in focus["test"].items()
    }
    numbers["hyperparameters"] = {
        "alpha": focus.get("alpha"),
        "per_channel_alphas": focus.get("per_channel_alphas"),
        "per_channel_thresholds": focus.get("per_channel_thresholds"),
    }

    rule_eval = jload(RULE_TUNED / "test" / "metrics.json")
    numbers["rule_engine_standalone"] = {
        "accuracy": round(rule_eval["accuracy"], 4),
        "macro_f1": round(rule_eval["macro_f1"], 4),
        "HATE_f1": round(rule_eval["HATE_f1"], 4),
        "OFFENSIVE_f1": round(rule_eval["OFFENSIVE_f1"], 4),
        "CLEAN_f1": round(rule_eval["CLEAN_f1"], 4),
    }

    # ---------------- per-sample error analysis ----------------
    preds = pd.read_csv(FUSION / FOCUS / "fusion_predictions.csv", encoding="utf-8")
    tuned = pd.read_csv(rule_path(), encoding="utf-8")[["index", "rule_hate_score", "rule_offensive_score"]]
    df = preds.merge(tuned, on="index", suffixes=("", "_rt"))
    # the rule columns shipped with fusion_predictions must equal the tuned run
    assert np.allclose(df["rule_hate_score"], df["rule_hate_score_rt"]), "rule source mismatch"

    df["rule_fire"] = (df["rule_hate_score"] > EPS) | (df["rule_offensive_score"] > EPS)
    df["ai_ok"] = df["ai_pred"] == df["true_label"]
    df["fu_ok"] = df["fused_pred"] == df["true_label"]

    numbers["test_distribution"] = {k: int(v) for k, v in df["true_label"].value_counts().items()}
    numbers["confusion"] = {
        "ai_correct": int(df["ai_ok"].sum()),
        "fusion_correct": int(df["fu_ok"].sum()),
        "fixed": int((~df["ai_ok"] & df["fu_ok"]).sum()),
        "broken": int((df["ai_ok"] & ~df["fu_ok"]).sum()),
        "both_wrong": int((~df["ai_ok"] & ~df["fu_ok"]).sum()),
    }

    fixed = df[~df["ai_ok"] & df["fu_ok"]]
    broken = df[df["ai_ok"] & ~df["fu_ok"]]
    numbers["fixed_transitions"] = {
        f"{r.true_label}<-{r.ai_pred}": int(g)
        for (r_true, r_ai), g in fixed.groupby(["true_label", "ai_pred"]).size().items()
        for r in [type("X", (), {"true_label": r_true, "ai_pred": r_ai})]
    }
    numbers["broken_transitions"] = {
        f"{k[0]}->{k[2]}": int(v)
        for k, v in broken.groupby(["true_label", "ai_pred", "fused_pred"]).size().items()
    }

    # rule channel behaviour
    n = len(df)
    nonclean = df[df["true_label"] != "CLEAN"]
    clean = df[df["true_label"] == "CLEAN"]
    numbers["rule_channel"] = {
        "coverage_count": int(df["rule_fire"].sum()),
        "coverage_pct": round(df["rule_fire"].mean() * 100, 2),
        "precision_on_firing_pct": round((df[df["rule_fire"]]["true_label"] != "CLEAN").mean() * 100, 2),
        "recall_nonclean_pct": round(nonclean["rule_fire"].mean() * 100, 2),
        "false_alarm_clean_pct": round(clean["rule_fire"].mean() * 100, 2),
    }

    # correction provenance
    fixed_hate = fixed[fixed["true_label"] == "HATE"]
    fixed_clean = fixed[fixed["true_label"] == "CLEAN"]
    numbers["correction_provenance"] = {
        "fixed_hate": int(len(fixed_hate)),
        "fixed_hate_with_rule_fire": int(fixed_hate["rule_fire"].sum()),
        "fixed_hate_with_rule_direction_ok": int(
            ((fixed_hate["rule_hate_score"] > fixed_hate["rule_offensive_score"])).sum()
        ),
        "fixed_hate_rule_hate_median": round(float(fixed_hate["rule_hate_score"].median()), 4),
        "fixed_hate_rule_hate_mean": round(float(fixed_hate["rule_hate_score"].mean()), 4),
        "fixed_clean": int(len(fixed_clean)),
        "fixed_clean_with_rule_silent": int((~fixed_clean["rule_fire"]).sum()),
        "fixed_clean_with_rule_fire_wrong_dir": int(fixed_clean["rule_fire"].sum()),
        "rule_consistent_total": int(len(fixed_hate) + (~fixed_clean["rule_fire"]).sum()),
    }

    # missed opportunities & regressions caused by the rule channel
    hate_dir = (df["true_label"] == "HATE") & (df["rule_hate_score"] > df["rule_offensive_score"]) & (df["rule_hate_score"] > EPS)
    off_dir = (df["true_label"] == "OFFENSIVE") & (df["rule_offensive_score"] >= df["rule_hate_score"]) & (df["rule_offensive_score"] > EPS)
    correct_dir = hate_dir | off_dir
    ai_wrong = ~df["ai_ok"]
    numbers["opportunities"] = {
        "ai_wrong_total": int(ai_wrong.sum()),
        "ai_wrong_nonclean_correct_rule_evidence": int((ai_wrong & correct_dir).sum()),
        "ai_wrong_nonclean_correct_rule_evidence_fixed": int((ai_wrong & correct_dir & df["fu_ok"]).sum()),
        "ai_wrong_clean_rule_silent": int((ai_wrong & (df["true_label"] == "CLEAN") & ~df["rule_fire"]).sum()),
        "ai_wrong_clean_rule_silent_fixed": int((ai_wrong & (df["true_label"] == "CLEAN") & ~df["rule_fire"] & df["fu_ok"]).sum()),
        "missed_corrections": int((ai_wrong & correct_dir & ~df["fu_ok"]).sum()),
    }
    numbers["regressions"] = {
        "broken_total": int(len(broken)),
        "broken_with_rule_fire": int(broken["rule_fire"].sum()),
        "broken_with_rule_fire_wrong_direction": int(
            (broken["rule_fire"] & ~(((broken["true_label"] == "HATE") & (broken["rule_hate_score"] > broken["rule_offensive_score"])) | ((broken["true_label"] == "OFFENSIVE") & (broken["rule_offensive_score"] >= broken["rule_hate_score"])))).sum()
        ),
        "broken_to_HATE": int((broken["fused_pred"] == "HATE").sum()),
    }

    # ---------------- cue taxonomy on the corrected HATE cases ----------------
    A = r"(?<![a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ])" \
        r"(dm|đm|dcm|cmm|cmnr|cm|cc|vcl|vkl|vl|đéo|deo|buồi|buoi|loxz|đjt|dit|lz)" \
        r"(?![a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ])"
    B = r"(chó|cho\b|lồn|lon\b|mả cha|mẹ mày|con mẹ|hậu môn|đít|súc vật|cút)"
    C = r"(ngu|nhục|sủa|dốt|thua|khôn hơn|óc)"
    cues = Counter()
    for _, r in fixed_hate.iterrows():
        t = str(r["text"]).lower()
        hit = False
        for name, pat in (("teencode", A), ("explicit", B), ("derogatory", C)):
            if re.search(pat, t):
                cues[name] += 1
                hit = True
        if not hit:
            cues["no_cue"] += 1
    numbers["cue_taxonomy_hate_corrections"] = {
        "n": int(len(fixed_hate)),
        **{k: int(v) for k, v in cues.items()},
        "multi_label": True,
    }

    # ---------------- significance ----------------
    sig = jload(FUSION / "fusion_significance.json")
    numbers["significance"] = {
        m: {
            "delta_ai_minus_fusion": sig["results"][m]["paired_bootstrap_macro_f1"]["delta(A_minus_B)"],
            "ci_low": sig["results"][m]["paired_bootstrap_macro_f1"]["delta_ci_low"],
            "ci_high": sig["results"][m]["paired_bootstrap_macro_f1"]["delta_ci_high"],
            "p_bootstrap": sig["results"][m]["paired_bootstrap_macro_f1"]["p"],
            "mcnemar_b": sig["results"][m]["mcnemar"]["b"],
            "mcnemar_c": sig["results"][m]["mcnemar"]["c"],
            "mcnemar_p": sig["results"][m]["mcnemar"]["exact_p"],
        }
        for m in MODELS
    }

    (OUT / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- markdown report ----------------
    L = []
    L.append("# ICCIES 2027 - paper numbers (single source of truth)\n")
    L.append(f"Focus model: `{FOCUS}`; rule evidence threshold `{EPS}`; rule scores from `{rule_path().relative_to(ROOT)}`.\n")
    L.append("## Test distribution\n")
    for k, v in numbers["test_distribution"].items():
        L.append(f"- {k}: {v} ({v / n * 100:.2f}%)")
    L.append("\n## Components\n")
    L.append("| system | acc | macro-F1 | HATE F1 | OFF F1 | CLEAN F1 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| AI-only (argmax) | {focus['ai_only_reference']['accuracy']:.4f} | {focus['ai_only_reference']['macro_f1']:.4f} | {focus['ai_only_reference']['HATE_f1']:.4f} | {focus['ai_only_reference']['OFFENSIVE_f1']:.4f} | {focus['ai_only_reference']['CLEAN_f1']:.4f} |")
    for k, v in numbers["fusion_variants"].items():
        L.append(f"| {k} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | {v['HATE_f1']:.4f} | {v['OFFENSIVE_f1']:.4f} | {v['CLEAN_f1']:.4f} |")
    L.append(f"\nRule engine standalone (tuned): acc {numbers['rule_engine_standalone']['accuracy']:.4f}, macro-F1 {numbers['rule_engine_standalone']['macro_f1']:.4f}.\n")
    L.append("## Rule channel\n")
    for k, v in numbers["rule_channel"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Corrections / regressions\n")
    for k, v in numbers["confusion"].items():
        L.append(f"- {k}: {v}")
    L.append("\nFixed transitions: " + ", ".join(f"{k}={v}" for k, v in numbers["fixed_transitions"].items()))
    L.append("\nBroken transitions: " + ", ".join(f"{k}={v}" for k, v in numbers["broken_transitions"].items()))
    L.append("\n## Correction provenance\n")
    for k, v in numbers["correction_provenance"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Opportunities / regressions\n")
    for k, v in numbers["opportunities"].items():
        L.append(f"- {k}: {v}")
    for k, v in numbers["regressions"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Cue taxonomy (HATE corrections, multi-label)\n")
    for k, v in numbers["cue_taxonomy_hate_corrections"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Backbones\n")
    L.append("| model | alpha | AI macro-F1 | fusion macro-F1 | delta |")
    L.append("|---|---|---|---|---|")
    for m, v in numbers["backbones"].items():
        L.append(f"| {m} | {v['alpha']} | {v['ai_macro_f1']:.4f} | {v['fusion_macro_f1']:.4f} | {v['delta_macro_f1']:+.4f} |")
    L.append("\n## Significance\n")
    L.append("| model | delta(AI-fusion) | 95% CI | p_boot | b | c | p_McNemar |")
    L.append("|---|---|---|---|---|---|---|")
    for m, v in numbers["significance"].items():
        L.append(f"| {m} | {v['delta_ai_minus_fusion']:+.4f} | [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {v['p_bootstrap']:.4f} | {v['mcnemar_b']} | {v['mcnemar_c']} | {v['mcnemar_p']:.4f} |")

    (OUT / "paper_numbers.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[47] wrote {OUT/'numbers.json'}")
    print(f"[47] wrote {OUT/'paper_numbers.md'}")
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
