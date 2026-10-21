"""Step 38 - DO F1: lexicon do embedding (fastText) gan nhan co tot hon khong?

    python scripts/38_evaluate_embedding_lexicon.py

Buoc cuoi trong pipeline cua mentor:
    1. tokenize tieng Viet                       -> script 34/37
    2. train embedding tren 3 dataset            -> script 37
    3. ghi lai trong so thien class              -> script 37
    4. filter bot                                -> script 37
    5. train xem no nam o label nao + DO diem    -> script nay

Script 37 cho term + nhan du doan (HATE/OFFENSIVE) + class_margin. Script nay
bien chung thanh lexicon that (dung schema verified_lexicon.csv) va chay rule
engine tren validation de so:

    A. current                  : 668 entry hien tai
    B. embedding only           : term do embedding gan nhan (nhan tu fastText)
    C. current + embedding      : union

Moi bien the tune threshold rieng tren validation (plan section 44). Sau do
kiem dinh paired bootstrap + McNemar so voi current, de biet chenh lech co
y nghia hay chi la nhieu.

Outputs (outputs/lexicon_experiments/fasttext_embedding/)
    lexicon_embedding_only.csv / lexicon_embedding_union.csv
    embedding_lexicon_eval.md / .json     ket qua F1
    embedding_significance.md / .json     kiem dinh
Khong ghi de lexicon chinh thuc (plan section 44).
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

LOGGER = get_logger("sensitiveai.scripts.38_evaluate_embedding_lexicon")

OUT_DIR = Path("outputs/lexicon_experiments/fasttext_embedding")
CURRENT_LEXICON = Path("data/processed/lexicon/verified_lexicon.csv")
RULE_CONFIG = Path("configs/rule_engine.yaml")
SEVERITY = {"HATE": 0.9, "OFFENSIVE": 0.6}


def load_validation() -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(SPLITS_DIR / "validation.csv")
    return df["text"].astype(str).tolist(), df["label"].to_numpy(int)


def build_embedding_lexicon(scores_csv: Path, out_csv: Path, min_margin: float) -> int:
    """Bien term_class_scores.csv (script 37) thanh lexicon dung schema."""
    df = pd.read_csv(scores_csv)
    df = df[(df["pred_label"] != "CLEAN") & (df["class_margin"] >= min_margin)].copy()
    df = df.drop_duplicates(subset=["term"]).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        label = str(r["pred_label"])
        rows.append({
            "entry_id": f"emb-{i:05d}",
            "term": r["term"],
            "canonical_form": r["term"],
            "lexical_labels": label,
            "severity": SEVERITY.get(label, 0.6),
            "confidence": round(float(r["weight_hate"] + r["weight_offensive"]), 4)
            if "weight_hate" in r else 0.5,
            "context_required": "false",
            "source_datasets": "vihsd;vicm;bontori",
            "source_split": "train",
            "review_status": "auto",
            "dictionary_version": "fasttext-embedding-2026-10-20",
        })
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    LOGGER.info("[38] wrote %s (%d entries, margin>=%.3f)", out_csv, len(out), min_margin)
    return len(out)


def build_union(embedding_csv: Path, out_csv: Path) -> int:
    base = pd.read_csv(CURRENT_LEXICON)
    emb = pd.read_csv(embedding_csv)
    existing = set(base["term"].astype(str))
    emb_new = emb[~emb["term"].astype(str).isin(existing)]
    combined = pd.concat([base, emb_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["term"], keep="first").reset_index(drop=True)
    combined.to_csv(out_csv, index=False, encoding="utf-8")
    LOGGER.info("[38] wrote %s (%d entries: %d current + %d moi)",
                out_csv, len(combined), len(base), len(emb_new))
    return len(combined)


def run_lexicon(lexicon_path: Path, texts: Sequence[str], y: np.ndarray) -> Dict[str, Any]:
    config = load_rule_engine_config(RULE_CONFIG)
    config.lexicon_path = str(lexicon_path)
    lexicon = Lexicon.load(lexicon_path, config.scoring.min_review_statuses)
    engine = RuleEngine(lexicon, config)
    hate = np.zeros(len(texts))
    offensive = np.zeros(len(texts))
    for i, text in enumerate(texts):
        result = engine.analyze(text)
        hate[i] = float(result["rule_hate_score"])
        offensive[i] = float(result["rule_offensive_score"])
    clean = np.clip(1.0 - np.maximum(hate, offensive), 0.0, 1.0)
    proba = np.stack([hate, offensive, clean], axis=1)
    payload = optimize_thresholds(y, proba, objective="macro_f1", grid_size=101, coordinate_rounds=3)
    thresholds = [float(t) for t in payload["threshold_list"]]
    pred = predict_with_thresholds(proba, thresholds)
    metrics = compute_metrics(y, pred, proba)
    return {
        "entries": len(lexicon),
        "thresholds": [round(t, 4) for t in thresholds],
        "macro_f1": float(metrics["macro_f1"]),
        "macro_f1_argmax": float(compute_metrics(y, proba.argmax(axis=1), proba)["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "HATE_f1": float(metrics["HATE_f1"]),
        "OFFENSIVE_f1": float(metrics["OFFENSIVE_f1"]),
        "CLEAN_f1": float(metrics["CLEAN_f1"]),
        "pred": pred,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Do F1 lexicon do embedding gan nhan (val).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    scores_csv = out_dir / "term_class_scores.csv"
    if not scores_csv.is_file():
        LOGGER.error("[38] thieu %s - chay scripts/37_embedding_class_weight.py truoc", scores_csv)
        return 1

    emb_csv = out_dir / "lexicon_embedding_only.csv"
    union_csv = out_dir / "lexicon_embedding_union.csv"
    build_embedding_lexicon(scores_csv, emb_csv, args.min_margin)
    build_union(emb_csv, union_csv)

    texts, y = load_validation()
    LOGGER.info("[38] validation n=%d", len(y))

    variants = {
        "current": CURRENT_LEXICON,
        "embedding only": emb_csv,
        "current + embedding": union_csv,
    }
    scored: Dict[str, Dict[str, Any]] = {}
    for name, path in variants.items():
        LOGGER.info("[38] scoring %s", name)
        scored[name] = run_lexicon(path, texts, y)
        m = scored[name]
        LOGGER.info("[38]   macro_f1=%.4f argmax=%.4f entries=%d thr=%s",
                    m["macro_f1"], m["macro_f1_argmax"], m["entries"], m["thresholds"])

    base = scored["current"]
    results = {name: {k: v for k, v in m.items() if k != "pred"}
               for name, m in scored.items()}

    # ---- report -------------------------------------------------------------
    lines: List[str] = []
    add = lines.append
    add("# Rule engine voi lexicon do embedding (fastText) gan nhan\n")
    add("> Tren **validation** (n=%d), tune threshold rieng tung lexicon (plan §44). "
        "Nhan term do fastText gan (script 37), khong phai nguoi gan. Khong cham test.\n" % len(y))
    add("## Ket qua\n")
    add("| Lexicon | Entries | Macro-F1 (thr) | Macro-F1 (argmax) | Accuracy | HATE F1 | OFF F1 | CLEAN F1 |")
    add("|---|---|---|---|---|---|---|---|")
    for name, m in scored.items():
        add(f"| {name} | {m['entries']} | **{m['macro_f1']:.4f}** | {m['macro_f1_argmax']:.4f} | "
            f"{m['accuracy']:.4f} | {m['HATE_f1']:.4f} | {m['OFFENSIVE_f1']:.4f} | {m['CLEAN_f1']:.4f} |")
    add("\n## Chenh lech + kiem dinh so voi current\n")
    add("| Lexicon | Δ macro-F1 | CI 95% | p (bootstrap) | McNemar a>b / a<b | p | Ket luan |")
    add("|---|---|---|---|---|---|---|")
    sig: Dict[str, Any] = {}
    for name, m in scored.items():
        if name == "current":
            continue
        delta = paired_bootstrap_delta(y, m["pred"], base["pred"], metric="macro_f1",
                                       n_boot=args.n_boot, seed=42)
        mcn = mcnemar(y, m["pred"], base["pred"])
        significant = (delta["ci_low"] > 0.0) or (delta["ci_high"] < 0.0)
        verdict = "CO y nghia" if significant else "KHONG (CI chua 0)"
        add(f"| {name} | {delta['delta']:+.4f} | [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] | "
            f"{delta['p_value']:.3f} | {mcn['a_correct_b_wrong']} / {mcn['a_wrong_b_correct']} | "
            f"{mcn['p_value']:.3f} | {verdict} |")
        sig[name] = {
            "delta": round(float(delta["delta"]), 6),
            "ci": [round(float(delta["ci_low"]), 6), round(float(delta["ci_high"]), 6)],
            "p_value": round(float(delta["p_value"]), 6),
            "mcnemar": {"new_right_old_wrong": mcn["a_correct_b_wrong"],
                        "new_wrong_old_right": mcn["a_wrong_b_correct"],
                        "p_value": round(float(mcn["p_value"]), 6)},
            "significant_95": bool(significant),
        }
        LOGGER.info("[38] %s: d=%+.4f CI[%+.4f,%+.4f] p=%.3f sig=%s",
                    name, delta["delta"], delta["ci_low"], delta["ci_high"],
                    delta["p_value"], significant)
    add("\n## Cach doc\n")
    add("- Δ la so voi lexicon hien tai. CI chua 0 => chenh lech nam trong nhieu.")
    add("- Day la rule engine **dung rieng**; anh huong len fusion cuoi cung nho hon nua.")
    add("- Nhan do embedding gan co the sai (vd tu trung tinh bi gan OFFENSIVE) - xem script 37.")
    (out_dir / "embedding_lexicon_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "embedding_lexicon_eval.json").write_text(
        json.dumps({"results": results, "significance": sig, "n": int(len(y))},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "embedding_significance.json").write_text(
        json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[38] wrote %s/embedding_lexicon_eval.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
