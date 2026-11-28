"""Step 35 - DO THAT: lexicon mo rong bang tokenizer tieng Viet co tot hon khong?

    python scripts/35_evaluate_vi_tokenizer_lexicon.py

Cau hoi cua mentor: tach lexicon tu ViHSD/ViCM/bontori bang tokenize tieng Viet
(script 34) co lam rule engine tot hon khong?

Script 34 cho thay che do `word` (underthesea) tim duoc 138 term MA che do
`syllable` (hien tai) bo sot - nhung cung MAT 270 term. Cau hoi chi co the tra
loi bang SO DO tren validation, khong bang ly thuyet.

Script nay tao 3 lexicon roi chay rule engine tren CUNG validation split, tune
threshold tung lexicon (plan section 44: chi validation), so macro-F1:

    A. current     : data/processed/lexicon/verified_lexicon.csv (dang dung)
    B. +word terms : A + 138 term moi tu script 34 (che do word)
    C. word-only   : chi giu term cua che do word  (de xem rieng no ra sao)

Term moi duoc gan nhan bang cach tai su dung `assign_labels` cua
`src/data/lexical_labeler.py` (cung logic da tao ra 668 entry hien tai), nen
nhan/severity/confidence nhat quan voi lexicon goc.

Outputs (outputs/lexicon_experiments/vi_tokenizer/)
    lexicon_expanded_word.csv     lexicon B (A + term word-mode)
    lexicon_word_only.csv         lexicon C
    tokenizer_lexicon_eval.md     bang so sanh macro-F1 val + per-class
    tokenizer_lexicon_eval.json   cung so lieu dang may doc

Khong ghi de lexicon chinh thuc nao (plan section 44).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.constants import EVALUATION_DIR, ID2LABEL, LABELS, SPLITS_DIR  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.thresholds import optimize_thresholds  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402
from src.rule_engine.lexical_classifier import Lexicon  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.35_evaluate_vi_tokenizer_lexicon")

OUT_DIR = Path("outputs/lexicon_experiments/vi_tokenizer")
CURRENT_LEXICON = Path("data/processed/lexicon/verified_lexicon.csv")
RULE_CONFIG = Path("configs/rule_engine.yaml")


def load_validation() -> tuple[List[str], np.ndarray]:
    path = SPLITS_DIR / "validation.csv"
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist(), df["label"].to_numpy(int)


def read_terms(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path} - chay scripts/34_lexicon_vi_tokenizer.py truoc")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_expanded_lexicon(new_terms: Sequence[str], out_csv: Path, only_new: bool = False) -> None:
    """Them term moi vao verified_lexicon.csv, gan nhan bang logic cua lexical_labeler."""
    base = pd.read_csv(CURRENT_LEXICON)
    existing_terms = set(base["term"].astype(str))
    candidates = [t for t in dict.fromkeys(new_terms) if t not in existing_terms and len(t.replace(" ", "")) >= 2]
    LOGGER.info("[35] %d term moi (sau khi loai trung)", len(candidates))

    new_rows = _label_terms(candidates)
    if only_new:
        combined = new_rows
    else:
        combined = pd.concat([base, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["term"], keep="first").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False, encoding="utf-8")
    LOGGER.info("[35] wrote %s (%d entries)", out_csv, len(combined))


def _label_terms(terms: Sequence[str]) -> pd.DataFrame:
    """Gan nhan term moi bang DUNG pipeline cua lexical_labeler (script 16).

    Tai su dung `load_training_rows` + `collect_evidence` + `label_all` de nhan/
    severity/confidence nhat quan voi 668 entry hien tai, thay vi tu bia.
    """
    from src.data.lexical_labeler import (
        LabelingConfig,
        collect_evidence,
        label_all,
        load_training_rows,
    )

    cfg = LabelingConfig()
    candidates = [t for t in terms if t]
    rows = load_training_rows()
    LOGGER.info("[35] labeller: %d candidate, %d training rows", len(candidates), len(rows))
    evidence, term_datasets, rows_by_label, _ = collect_evidence(candidates, rows)
    records = label_all(candidates, evidence, term_datasets, rows_by_label, cfg)

    rows_out = []
    for i, record in enumerate(records, start=1):
        labels = record.get("lexical_labels") or []
        if isinstance(labels, str):
            labels = [x for x in labels.replace(",", ";").split(";") if x]
        if not labels:
            continue  # labeller tu choi term nay (khong du bang chung)
        rows_out.append({
            "entry_id": f"exp-vi-{i:05d}",
            "term": record["term"],
            "canonical_form": record.get("canonical_form") or record["term"],
            "lexical_labels": ";".join(labels),
            "severity": record.get("severity", 0.9),
            "confidence": record.get("confidence", 0.5),
            "context_required": str(bool(record.get("context_required", False))).lower(),
            "source_datasets": record.get("source_datasets") or "vihsd;vicm;bontori",
            "source_split": "train",
            "review_status": record.get("review_status", "auto"),
            "dictionary_version": "corpus-vocab-expansion-2026-10-20",
        })
    LOGGER.info("[35] labeller accepted %d / %d term", len(rows_out), len(candidates))
    return pd.DataFrame(rows_out)


def score_lexicon(lexicon_path: Path, texts: Sequence[str], y: np.ndarray) -> Dict[str, Any]:
    """Chay rule engine voi 1 lexicon, tune threshold tren validation, tra metrics."""
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
    pred = _predict_with_thresholds(proba, payload["threshold_list"])
    metrics = compute_metrics(y, pred, proba)
    return {
        "entries": len(lexicon),
        "thresholds": [float(t) for t in payload["threshold_list"]],
        "macro_f1_argmax": round(float(compute_metrics(y, proba.argmax(axis=1), proba)["macro_f1"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "accuracy": round(float(metrics["accuracy"]), 6),
        "HATE_f1": round(float(metrics["HATE_f1"]), 6),
        "OFFENSIVE_f1": round(float(metrics["OFFENSIVE_f1"]), 6),
        "CLEAN_f1": round(float(metrics["CLEAN_f1"]), 6),
    }


def _predict_with_thresholds(proba: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    from src.evaluation.metrics import predict_with_thresholds

    return predict_with_thresholds(proba, thresholds)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Do rule engine voi lexicon mo rong bang tokenizer tieng Viet.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    texts, y = load_validation()
    LOGGER.info("[35] validation n=%d", len(y))

    word_terms = read_terms(out_dir / "word_mode_vihsd.txt") + read_terms(out_dir / "word_mode_vicm.txt") + read_terms(out_dir / "word_mode_bontori.txt")
    LOGGER.info("[35] doc duoc %d term che do word", len(word_terms))

    expanded_csv = out_dir / "lexicon_expanded_word.csv"
    word_only_csv = out_dir / "lexicon_word_only.csv"
    build_expanded_lexicon(word_terms, expanded_csv, only_new=False)
    build_expanded_lexicon(word_terms, word_only_csv, only_new=True)

    variants = {
        "current (verified_lexicon.csv)": CURRENT_LEXICON,
        "current + word terms": expanded_csv,
        "word-mode terms only": word_only_csv,
    }
    results: Dict[str, Any] = {}
    for name, path in variants.items():
        if name == "word-mode terms only" and not pd.read_csv(path).shape[0]:
            LOGGER.warning("[35] %s rong - bo qua", name)
            continue
        LOGGER.info("[35] scoring: %s", name)
        results[name] = score_lexicon(path, texts, y)
        LOGGER.info("[35]   macro-F1 %.4f (argmax %.4f), %d entries",
                    results[name]["macro_f1"], results[name]["macro_f1_argmax"], results[name]["entries"])

    # ---- report ---------------------------------------------------------------
    lines: List[str] = []
    add = lines.append
    add("# Rule engine với lexicon mở rộng bằng tokenizer tiếng Việt\n")
    add("> Đo trên **validation** (n=2,672), tune threshold riêng cho từng lexicon "
        "(plan §44). Không chấm test. Lexicon chính thức không bị sửa.\n")
    add("## Kết quả\n")
    add("| Lexicon | Số entry | Macro-F1 (threshold) | Macro-F1 (argmax) | Accuracy | HATE F1 | OFF F1 | CLEAN F1 |")
    add("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        add(f"| {name} | {r['entries']} | **{r['macro_f1']:.4f}** | {r['macro_f1_argmax']:.4f} | "
            f"{r['accuracy']:.4f} | {r['HATE_f1']:.4f} | {r['OFFENSIVE_f1']:.4f} | {r['CLEAN_f1']:.4f} |")
    base = results.get("current (verified_lexicon.csv)")
    if base:
        add("\n## Chênh lệch so với lexicon hiện tại\n")
        add("| Lexicon | Δ macro-F1 | Δ HATE F1 | Δ OFF F1 |")
        add("|---|---|---|---|")
        for name, r in results.items():
            add(f"| {name} | {r['macro_f1'] - base['macro_f1']:+.4f} | "
                f"{r['HATE_f1'] - base['HATE_f1']:+.4f} | {r['OFFENSIVE_f1'] - base['OFFENSIVE_f1']:+.4f} |")
    add("\n## Cách đọc\n")
    add("- Δ dương = lexicon mở rộng tốt hơn trên validation.")
    add("- Nếu Δ ≈ 0 hoặc âm: tokenizer tiếng Việt **không** giúp rule engine — "
        "term mới (138) không bù được term mất (270), hoặc term mới không match được text nhiễu.")
    add("- Số này là **rule engine đứng riêng**; ảnh hưởng lên fusion còn nhỏ hơn nữa "
        "(fusion chỉ chiếm trọng số β nhỏ).")
    (out_dir / "tokenizer_lexicon_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "tokenizer_lexicon_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[35] wrote %s/tokenizer_lexicon_eval.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
