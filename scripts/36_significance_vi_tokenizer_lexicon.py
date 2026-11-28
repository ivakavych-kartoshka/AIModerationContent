"""Step 36 - kiem dinh y nghia cua lexicon mo rong bang tokenizer tieng Viet (val).

    python scripts/36_significance_vi_tokenizer_lexicon.py

Script 35 do duoc: lexicon hien tai (668 entry) 0.4555 macro-F1, con "current +
261 term che do word" (929 entry) 0.4628 tren validation. Cau hoi tiep theo -
MA CAU HOI DUNG - la: +0.0074 do la tin hieu that hay chi la nhieu?

Script nay tra loi bang kiem dinh thong ke, TAT CA tren validation (n=2672),
khong cham test (plan section 44). Voi moi cap so sanh:

    paired bootstrap (10000 lan resample, CUNG chi so) -> CI 95% + p-value
    McNemar exact hai phia                            -> dem mau bat dong

Neu CI chua 0 va p > 0.05 -> ket luan trung thuc: KHONG co y nghia thong ke,
khong duoc viet vao paper nhu "cai thien".

Outputs (outputs/lexicon_experiments/vi_tokenizer/)
    significance.json   CI + p-value + bang McNemar dang may doc
    significance.md     bang doc duoc
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

from src.common.logging_utils import get_logger  # noqa: E402
from src.common.constants import SPLITS_DIR  # noqa: E402
from src.evaluation.analysis import bootstrap_ci, mcnemar, paired_bootstrap_delta  # noqa: E402
from src.evaluation.metrics import compute_metrics, predict_with_thresholds  # noqa: E402
from src.evaluation.thresholds import optimize_thresholds  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402
from src.rule_engine.lexical_classifier import Lexicon  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.36_significance_vi_tokenizer_lexicon")

OUT_DIR = Path("outputs/lexicon_experiments/vi_tokenizer")
CURRENT_LEXICON = Path("data/processed/lexicon/verified_lexicon.csv")
RULE_CONFIG = Path("configs/rule_engine.yaml")
N_BOOT = 10000
SEED = 42


def load_validation() -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(SPLITS_DIR / "validation.csv")
    return df["text"].astype(str).tolist(), df["label"].to_numpy(int)


def run_lexicon(lexicon_path: Path, texts: Sequence[str], y: np.ndarray) -> Dict[str, Any]:
    """Chay rule engine, tune threshold tren validation, tra (pred, proba, metrics)."""
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
        "thresholds": thresholds,
        "pred": pred,
        "proba": proba,
        "macro_f1": float(metrics["macro_f1"]),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiem dinh significance lexicon tokenizer VI (val).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    texts, y = load_validation()
    LOGGER.info("[36] validation n=%d", len(y))

    variants = {
        "current": CURRENT_LEXICON,
        "current_plus_word": out_dir / "lexicon_expanded_word.csv",
        "word_only": out_dir / "lexicon_word_only.csv",
    }
    scored: Dict[str, Dict[str, Any]] = {}
    for name, path in variants.items():
        if not path.is_file():
            LOGGER.warning("[36] thieu %s - chay script 35 truoc", path)
            continue
        LOGGER.info("[36] scoring %s (%s)", name, path)
        scored[name] = run_lexicon(path, texts, y)
        m = scored[name]
        LOGGER.info("[36]   macro_f1=%.4f entries=%d thr=%s",
                    m["macro_f1"], m["entries"], [round(t, 3) for t in m["thresholds"]])

    if "current" not in scored or "current_plus_word" not in scored:
        LOGGER.error("[36] can ca 'current' va 'current_plus_word' - dung")
        return 1

    report: Dict[str, Any] = {
        "split": "validation",
        "n": int(len(y)),
        "n_boot": int(args.n_boot),
        "seed": SEED,
        "point_estimates": {
            name: {"macro_f1": round(m["macro_f1"], 6), "entries": m["entries"],
                   "thresholds": [round(t, 4) for t in m["thresholds"]]}
            for name, m in scored.items()
        },
        "comparisons": {},
    }

    base = scored["current"]
    lines: List[str] = []
    add = lines.append
    add("# Kiem dinh y nghia: lexicon mo rong bang tokenizer tieng Viet\n")
    add(f"> Tren **validation** (n={len(y)}), {args.n_boot} lan paired bootstrap (seed={SEED}). "
        "Tune threshold rieng cho tung lexicon (plan §44). Khong cham test.\n")
    add("## Diem (validation)\n")
    add("| Lexicon | Entries | Macro-F1 |")
    add("|---|---|---|")
    for name, m in scored.items():
        add(f"| {name} | {m['entries']} | {m['macro_f1']:.4f} |")

    add("\n## Kiem dinh so voi lexicon hien tai\n")
    add("| So sanh | Δ macro-F1 | CI 95% | p (bootstrap) | McNemar a>b / a<b | p (McNemar) | Ket luan |")
    add("|---|---|---|---|---|---|---|")

    for name, m in scored.items():
        if name == "current":
            continue
        delta = paired_bootstrap_delta(y, m["pred"], base["pred"], metric="macro_f1",
                                       n_boot=args.n_boot, seed=SEED)
        mcn = mcnemar(y, m["pred"], base["pred"])
        significant = (delta["ci_low"] > 0.0) or (delta["ci_high"] < 0.0)
        verdict = "CO y nghia" if significant else "KHONG co y nghia (CI chua 0)"
        add(f"| {name} vs current | {delta['delta']:+.4f} | "
            f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] | {delta['p_value']:.3f} | "
            f"{mcn['a_correct_b_wrong']} / {mcn['a_wrong_b_correct']} | {mcn['p_value']:.3f} | {verdict} |")
        report["comparisons"][name] = {
            "vs": "current",
            "paired_bootstrap": {k: round(float(v), 6) for k, v in delta.items()},
            "mcnemar": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                        for k, v in mcn.items()},
            "significant_95": bool(significant),
        }
        LOGGER.info("[36] %s vs current: d=%+.4f CI[%+.4f,%+.4f] p=%.3f sig=%s",
                    name, delta["delta"], delta["ci_low"], delta["ci_high"],
                    delta["p_value"], significant)

    # absolute CI for the headline numbers (independent of the comparison)
    abs_ci: Dict[str, Any] = {}
    for name, m in scored.items():
        ci = bootstrap_ci(y, m["pred"], metric="macro_f1", n_boot=args.n_boot, seed=SEED)
        abs_ci[name] = {k: round(float(v), 6) for k, v in ci.items()}
    report["absolute_ci"] = abs_ci

    add("\n## Cach doc (trung thuc)\n")
    add("- CI 95% **chua 0** => khong the ket luan lexicon mo rong tot hon; chenh lech nam trong nhieu.")
    add("- Day la so cua **rule engine dung rieng**. Trong he thong fusion, rule chi chiem trong so nho, "
        "nen anh huong len macro-F1 cuoi cung con nho hon nua.")
    add("- McNemar dem so mau doi ket qua dung/sai: cot `a>b` = lexicon moi dung ma cu sai, "
        "`a<b` = nguoc lai.")

    (out_dir / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "significance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    LOGGER.info("[36] wrote %s/significance.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
