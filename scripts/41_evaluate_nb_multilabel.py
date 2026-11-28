"""Step 41 - DO F1: lexicon NB xac suat + multi-label co tot hon khong?

    python scripts/41_evaluate_nb_multilabel.py
    python scripts/41_evaluate_nb_multilabel.py --min-prob 0.4

Buoc cuoi cua huong "mo hinh xac suat + multi-label" (script 40). Chay rule
engine tren validation voi cac bien the:

    A. current                 : 668 entry hien tai (moc)
    B. nb only                 : chi lexicon do NB sinh
    C. current + nb            : union
    D. nb multilabel only      : chi giu term mang CA 2 nhan (kiem tra rieng
                                 xem multi-label co phai nguon cai thien khong)

Moi bien the tune threshold tren validation (plan section 44). Kiem dinh
paired bootstrap + McNemar so voi current.

Outputs (outputs/lexicon_experiments/nb_multilabel/)
    lexicon_nb_union.csv
    nb_eval.md / nb_eval.json / nb_significance.json
Khong cham test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.constants import SPLITS_DIR  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.analysis import mcnemar, paired_bootstrap_delta  # noqa: E402
from src.evaluation.metrics import compute_metrics, predict_with_thresholds  # noqa: E402
from src.evaluation.thresholds import optimize_thresholds  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402
from src.rule_engine.lexical_classifier import Lexicon  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.41_evaluate_nb_multilabel")

OUT_DIR = Path("outputs/lexicon_experiments/nb_multilabel")
CURRENT_LEXICON = Path("data/processed/lexicon/verified_lexicon.csv")
RULE_CONFIG = Path("configs/rule_engine.yaml")


def load_validation() -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(SPLITS_DIR / "validation.csv")
    return df["text"].astype(str).tolist(), df["label"].to_numpy(int)


def run_lexicon(path: Path, texts: Sequence[str], y: np.ndarray) -> Dict[str, Any]:
    config = load_rule_engine_config(RULE_CONFIG)
    config.lexicon_path = str(path)
    lexicon = Lexicon.load(path, config.scoring.min_review_statuses)
    engine = RuleEngine(lexicon, config)
    hate = np.zeros(len(texts))
    off = np.zeros(len(texts))
    for i, text in enumerate(texts):
        r = engine.analyze(text)
        hate[i] = float(r["rule_hate_score"])
        off[i] = float(r["rule_offensive_score"])
    clean = np.clip(1.0 - np.maximum(hate, off), 0.0, 1.0)
    proba = np.stack([hate, off, clean], axis=1)
    payload = optimize_thresholds(y, proba, objective="macro_f1", grid_size=101, coordinate_rounds=3)
    thr = [float(t) for t in payload["threshold_list"]]
    pred = predict_with_thresholds(proba, thr)
    m = compute_metrics(y, pred, proba)
    return {
        "entries": len(lexicon), "thresholds": [round(t, 4) for t in thr],
        "macro_f1": float(m["macro_f1"]),
        "macro_f1_argmax": float(compute_metrics(y, proba.argmax(axis=1), proba)["macro_f1"]),
        "accuracy": float(m["accuracy"]), "HATE_f1": float(m["HATE_f1"]),
        "OFFENSIVE_f1": float(m["OFFENSIVE_f1"]), "CLEAN_f1": float(m["CLEAN_f1"]),
        "pred": pred,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Do F1 lexicon NB multi-label (val).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--min-prob", type=float, default=0.25)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    # rebuild voi min-prob mong muon
    from importlib import import_module
    mod = import_module("scripts.40_nb_multilabel_lexicon")
    sdf_path = out_dir / "term_probabilities.csv"
    if not sdf_path.is_file():
        LOGGER.error("[41] thieu %s - chay scripts/40_nb_multilabel_lexicon.py truoc", sdf_path)
        return 1
    sdf = pd.read_csv(sdf_path)
    ml = mod.select_multilabel(sdf, args.min_prob)
    nb_csv = out_dir / "lexicon_nb_multilabel.csv"
    ml.to_csv(nb_csv, index=False, encoding="utf-8")
    LOGGER.info("[41] nb lexicon: %d entry (%d multi-label), min_prob=%.2f",
                len(ml), int(ml["lexical_labels"].str.contains(";").sum()), args.min_prob)

    # multi-label-only variant
    multi_only = ml[ml["lexical_labels"].str.contains(";")].copy()
    ml_only_csv = out_dir / "lexicon_nb_multilabel_only.csv"
    multi_only.to_csv(ml_only_csv, index=False, encoding="utf-8")

    # union
    base = pd.read_csv(CURRENT_LEXICON)
    existing = set(base["term"].astype(str))
    union = pd.concat([base, ml[~ml["term"].astype(str).isin(existing)]], ignore_index=True)
    union = union.drop_duplicates(subset=["term"], keep="first").reset_index(drop=True)
    union_csv = out_dir / "lexicon_nb_union.csv"
    union.to_csv(union_csv, index=False, encoding="utf-8")
    LOGGER.info("[41] union: %d entry", len(union))

    texts, y = load_validation()
    LOGGER.info("[41] validation n=%d", len(y))

    variants = {
        "current": CURRENT_LEXICON,
        "nb only": nb_csv if len(ml) else None,
        "current + nb": union_csv,
        "nb multilabel only": ml_only_csv if len(multi_only) else None,
    }
    scored: Dict[str, Dict[str, Any]] = {}
    for name, path in variants.items():
        if path is None:
            continue
        LOGGER.info("[41] scoring %s", name)
        scored[name] = run_lexicon(Path(path), texts, y)
        m = scored[name]
        LOGGER.info("[41]   macro_f1=%.4f argmax=%.4f entries=%d thr=%s",
                    m["macro_f1"], m["macro_f1_argmax"], m["entries"], m["thresholds"])

    base_m = scored["current"]
    results = {n: {k: v for k, v in m.items() if k != "pred"} for n, m in scored.items()}

    lines: List[str] = []
    add = lines.append
    add("# Rule engine voi lexicon Naive Bayes + multi-label\n")
    add(f"> Validation n={len(y)}, tune threshold rieng tung lexicon (plan §44). "
        f"NB alpha, nguong multi-label P>={args.min_prob}. Khong cham test.\n")
    add("## Ket qua\n")
    add("| Lexicon | Entries | Macro-F1 (thr) | Macro-F1 (argmax) | Accuracy | HATE F1 | OFF F1 | CLEAN F1 |")
    add("|---|---|---|---|---|---|---|---|")
    for n, m in scored.items():
        add(f"| {n} | {m['entries']} | **{m['macro_f1']:.4f}** | {m['macro_f1_argmax']:.4f} | "
            f"{m['accuracy']:.4f} | {m['HATE_f1']:.4f} | {m['OFFENSIVE_f1']:.4f} | {m['CLEAN_f1']:.4f} |")
    add("\n## Chenh lech + kiem dinh so voi current\n")
    add("| Lexicon | Δ macro-F1 | CI 95% | p (bootstrap) | McNemar +/− | p | Ket luan |")
    add("|---|---|---|---|---|---|---|")
    sig: Dict[str, Any] = {}
    for n, m in scored.items():
        if n == "current":
            continue
        d = paired_bootstrap_delta(y, m["pred"], base_m["pred"], metric="macro_f1",
                                   n_boot=args.n_boot, seed=42)
        mc = mcnemar(y, m["pred"], base_m["pred"])
        significant = (d["ci_low"] > 0.0) or (d["ci_high"] < 0.0)
        verdict = "CO y nghia" if significant else "khong (CI chua 0)"
        add(f"| {n} | {d['delta']:+.4f} | [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | "
            f"{d['p_value']:.3f} | {mc['a_correct_b_wrong']}/{mc['a_wrong_b_correct']} | "
            f"{mc['p_value']:.3f} | {verdict} |")
        sig[n] = {"delta": round(float(d["delta"]), 6),
                  "ci": [round(float(d["ci_low"]), 6), round(float(d["ci_high"]), 6)],
                  "p_value": round(float(d["p_value"]), 6),
                  "right": mc["a_correct_b_wrong"], "wrong": mc["a_wrong_b_correct"],
                  "mcnemar_p": round(float(mc["p_value"]), 6), "significant_95": bool(significant)}
        LOGGER.info("[41] %s: d=%+.4f CI[%+.4f,%+.4f] p=%.3f sig=%s",
                    n, d["delta"], d["ci_low"], d["ci_high"], d["p_value"], significant)
    add("\n## Cach doc\n")
    add("- `nb multilabel only` chi giu term mang CA HAI nhan - kiem tra rieng xem "
        "multi-label co phai nguon cai thien khong.")
    add("- CI chua 0 => khong co y nghia. Day la rule engine dung rieng, anh huong "
        "len fusion con nho hon.")
    (out_dir / "nb_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "nb_eval.json").write_text(
        json.dumps({"results": results, "significance": sig, "n": int(len(y)),
                    "min_prob": args.min_prob}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "nb_significance.json").write_text(json.dumps(sig, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    LOGGER.info("[41] wrote %s/nb_eval.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
