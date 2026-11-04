#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 50 - component ablations of the lexical rule channel (revision protocol 5.2).

Every variant is a *real* switch in the shipped code, not a re-description:

    full            the tuned configuration (fidelity check against the paper)
    no_severity     scoring.severity  {HATE: 1.0, OFFENSIVE: 1.0}
    no_confidence   lexicon copy with ``confidence`` and ``label_confidences`` = 1.0
    no_context      scoring.quote_factor = 1.0  (the only context factor still
                    active: tuning already drove negation and context_required to 1.0)
    no_views        obfuscation.enabled = false (exact + fuzzy only)
    no_fuzzy        matcher.fuzzy_enabled = false

Protocol, identical to the paper's: each variant re-scores validation AND test,
re-tunes alpha / thresholds on the **validation** split with the shipped fusion
code, freezes them, and scores test once.  Nothing is tuned on test, and no
existing artefact is overwritten - everything lands in
``outputs/evaluation/rule_ablations/``.

    python scripts/50_rule_component_ablations.py
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.analysis import mcnemar, paired_bootstrap_delta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/rule_engine_tuned.yaml"
BASE_LEXICON = ROOT / "data/processed/lexicon/verified_lexicon.csv"
OUT_ROOT = ROOT / "outputs/evaluation/rule_ablations"
MODEL = "sensitiveai-vi-custom"
LABELS = ["HATE", "OFFENSIVE", "CLEAN"]


def _variant_configs() -> dict[str, dict]:
    """Build each variant from the tuned config by deep-copying and overriding."""
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    variants: dict[str, dict] = {}

    def make(name: str):
        cfg = copy.deepcopy(base)
        cfg["dictionary_version"] = f"v1-abl-{name}"
        variants[name] = cfg
        return cfg

    make("full")

    cfg = make("no_severity")
    cfg["scoring"]["severity"] = {"HATE": 1.0, "OFFENSIVE": 1.0}

    cfg = make("no_confidence")
    # Written next to the variant so the ablation is self-describing; the
    # original lexicon is never modified.
    cfg["_confidence_neutralised"] = True

    cfg = make("no_context")
    cfg["scoring"]["quote_factor"] = 1.0

    cfg = make("no_views")
    cfg["obfuscation"]["enabled"] = False

    cfg = make("no_fuzzy")
    cfg["matcher"]["fuzzy_enabled"] = False

    return variants


def _write_confidence_free_lexicon(target: Path) -> Path:
    """Copy the lexicon with every confidence column set to 1.0."""
    frame = pd.read_csv(BASE_LEXICON)
    if "confidence" in frame.columns:
        frame["confidence"] = 1.0
    for column in ("label_confidences", "label_severities"):
        if column == "label_confidences" and column in frame.columns:
            frame[column] = frame[column].map(
                lambda cell: ";".join(
                    f"{part.split(':')[0]}:1.0" for part in str(cell).split(";") if part
                )
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8")
    return target


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(
            "command failed: " + " ".join(cmd[-6:]) + "\n" + (result.stderr or "")[-2500:]
        )


def _score_variant(name: str, cfg: dict, directive: Path) -> Path:
    """Run the rule engine on validation and test for one variant."""
    payload = copy.deepcopy(cfg)
    payload.pop("_confidence_neutralised", None)
    config_path = directive / "rule_config.yaml"
    if cfg.get("_confidence_neutralised"):
        lexicon = _write_confidence_free_lexicon(directive / "lexicon_conf1.csv")
        payload["lexicon_path"] = str(lexicon)
    directive.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for split in ("validation", "test"):
        _run(
            [
                sys.executable,
                "scripts/17_evaluate_rule_engine.py",
                "--config",
                str(config_path),
                "--split",
                split,
                "--out",
                str(directive),
            ]
        )
        # Step 19 exports the per-comment rule_scores.csv that fusion consumes
        # (step 17 metrics alone are not enough).
        _run(
            [
                sys.executable,
                "scripts/19_plot_rule_engine.py",
                "--config",
                str(config_path),
                "--split",
                split,
                "--out",
                str(directive),
            ]
        )
    return directive


def _fuse(name: str, directive: Path) -> Path:
    """Re-tune alpha/thresholds on validation and score test with the shipped fusion."""
    out_dir = OUT_ROOT / "fusion"
    _run(
        [
            sys.executable,
            "scripts/23_evaluate_fusion.py",
            "--rule-eval-dir",
            str(directive),
            "--out-dir",
            str(out_dir),
            "--model",
            MODEL,
        ]
    )
    produced = out_dir / MODEL
    if not produced.exists():
        raise SystemExit(f"fusion produced no metrics for {name}")
    # Keep one snapshot per variant so variants do not overwrite each other.
    snapshot = OUT_ROOT / "results" / name
    snapshot.mkdir(parents=True, exist_ok=True)
    for file in ("metrics.json", "fusion_predictions.csv"):
        source = produced / file
        if source.exists():
            (snapshot / file).write_bytes(source.read_bytes())
    return snapshot


def _summarise(name: str, snapshot: Path, rule_dir: Path) -> dict:
    meta = json.loads((snapshot / "metrics.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(snapshot / "fusion_predictions.csv")
    rule = pd.read_csv(rule_dir / "test" / "rule_scores.csv")

    fused = meta["test"]["fusion_per_channel_thresholds"]
    ai = meta["test"]["ai_only_argmax"]
    fired = (rule[["rule_hate_score", "rule_offensive_score"]].max(axis=1) > 0.01).to_numpy()
    gold = rule["true_label"] if "true_label" in rule.columns else None

    row = {
        "variant": name,
        "fires": int(fired.sum()),
        "coverage": float(fired.mean()),
        "alpha": meta["per_channel_alphas"],
        "thresholds": meta["per_channel_thresholds"],
        "macro_f1": fused["macro_f1"],
        "accuracy": fused["accuracy"],
        "HATE_f1": fused["HATE_f1"],
        "OFFENSIVE_f1": fused["OFFENSIVE_f1"],
        "CLEAN_f1": fused["CLEAN_f1"],
        "ai_macro_f1": ai["macro_f1"],
        "delta_vs_ai": fused["macro_f1"] - ai["macro_f1"],
        "correct": int((preds["fused_pred"] == preds["true_label"]).sum()),
        "corrections": int(
            ((preds["ai_pred"] != preds["true_label"]) & (preds["fused_pred"] == preds["true_label"])).sum()
        ),
        "regressions": int(
            ((preds["ai_pred"] == preds["true_label"]) & (preds["fused_pred"] != preds["true_label"])).sum()
        ),
    }
    if gold is not None:
        fired_toxic = fired & (gold.to_numpy() != "CLEAN")
        row["precision_on_fired"] = float(fired_toxic.sum() / max(1, fired.sum()))
    return row, preds


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    variants = _variant_configs()
    summaries: dict[str, dict] = {}
    preds_by_variant: dict[str, pd.DataFrame] = {}

    for name, cfg in variants.items():
        directive = OUT_ROOT / name
        print(f"[50] {name}: scoring validation + test ...", flush=True)
        _score_variant(name, cfg, directive)
        print(f"[50] {name}: re-tuning fusion on validation ...", flush=True)
        snapshot = _fuse(name, directive)
        row, preds = _summarise(name, snapshot, directive)
        summaries[name] = row
        preds_by_variant[name] = preds
        print(
            "      macro-F1 %.4f  acc %.4f  coverage %.2f%%  corr/regr %d/%d"
            % (row["macro_f1"], row["accuracy"], 100 * row["coverage"], row["corrections"], row["regressions"]),
            flush=True,
        )

    # significance of every variant against the full model, same test rows
    reference = preds_by_variant["full"]
    code = {label: index for index, label in enumerate(LABELS)}
    y_ref = reference["true_label"].map(code).to_numpy()
    p_ref = reference["fused_pred"].map(code).to_numpy()
    for name, preds in preds_by_variant.items():
        if name == "full":
            continue
        aligned = preds.set_index("index").loc[reference["index"]]
        p_var = aligned["fused_pred"].map(code).to_numpy()
        summaries[name]["vs_full"] = {
            "macro_f1": paired_bootstrap_delta(y_ref, p_var, p_ref, "macro_f1", n_boot=10000, seed=42),
            "mcnemar": mcnemar(y_ref, p_ref, p_var),
        }

    payload = {
        "protocol": (
            "Each variant re-scores validation and test with the shipped rule engine, "
            "re-tunes alpha and thresholds on validation with the shipped fusion code, "
            "freezes them, and scores test once. Test is never used to tune."
        ),
        "model": MODEL,
        "variants": summaries,
    }
    (OUT_ROOT / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\n%-14s %8s %8s %9s %9s %7s %10s" % ("variant", "macroF1", "acc", "coverage", "d_vs_AI", "d_McN", "corr/regr"))
    for name, row in summaries.items():
        extra = row.get("vs_full", {})
        p_mcn = extra.get("mcnemar", {}).get("p_value", float("nan")) if extra else float("nan")
        print(
            "%-14s %8.4f %8.4f %8.2f%% %+9.4f %7.3f %5d/%-4d"
            % (
                name,
                row["macro_f1"],
                row["accuracy"],
                100 * row["coverage"],
                row["delta_vs_ai"],
                p_mcn,
                row["corrections"],
                row["regressions"],
            )
        )
    print("\nwritten ->", OUT_ROOT / "ablation_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
