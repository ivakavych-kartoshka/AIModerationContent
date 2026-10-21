"""Step 42 - TUNE nguong multi-label + smoothing cua NB (chi tren validation).

    python scripts/42_tune_nb_lexicon.py

Script 41 cho thay lexicon NB (prob>=0.25, alpha=1.0) cai thien co y nghia:
+0.0227 macro-F1, CI [+0.0013, +0.0438], p=0.037. Cau hoi tiep: nguong nao la
tot nhat? Co the tot hon nua khong?

Script nay quet grid (min_prob x smoothing) TREN VALIDATION (plan section 44 -
khong bao gio tune tren test), chay rule engine + tune threshold cho tung diem,
roi:

    1. in bang grid -> chon diem tot nhat
    2. kiem dinh diem tot nhat vs current (paired bootstrap + McNemar)
    3. ghi ra lexicon tot nhat de dung cho buoc sau

Neu diem tot nhat khong vuot qua nhieu so voi +0.0227, ket luan trung thuc:
NB giup that nhung da bao hoa, khong co "phep mau" trong sieu tham so.

Outputs (outputs/lexicon_experiments/nb_multilabel/)
    nb_tuning_grid.csv/md     bang quet
    lexicon_nb_best.csv       lexicon tai diem tot nhat
    nb_tuning.md/json         ket luan
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

LOGGER = get_logger("sensitiveai.scripts.42_tune_nb_lexicon")

OUT_DIR = Path("outputs/lexicon_experiments/nb_multilabel")
CURRENT_LEXICON = Path("data/processed/lexicon/verified_lexicon.csv")
RULE_CONFIG = Path("configs/rule_engine.yaml")


def load_validation() -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(SPLITS_DIR / "validation.csv")
    return df["text"].astype(str).tolist(), df["label"].to_numpy(int)


def run_lexicon(path: Path, texts: Sequence[str], y: np.ndarray,
                config: Any = None) -> Dict[str, Any]:
    cfg = config or load_rule_engine_config(RULE_CONFIG)
    cfg.lexicon_path = str(path)
    lexicon = Lexicon.load(path, cfg.scoring.min_review_statuses)
    engine = RuleEngine(lexicon, cfg)
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
    return {"entries": len(lexicon), "thresholds": [round(t, 4) for t in thr],
            "macro_f1": float(m["macro_f1"]), "accuracy": float(m["accuracy"]),
            "HATE_f1": float(m["HATE_f1"]), "OFFENSIVE_f1": float(m["OFFENSIVE_f1"]),
            "CLEAN_f1": float(m["CLEAN_f1"]), "pred": pred}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune NB lexicon tren validation.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--alphas", default="0.3,1.0,3.0")
    parser.add_argument("--min-probs", default="0.15,0.25,0.35,0.5")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from importlib import import_module
    nb_mod = import_module("scripts.40_nb_multilabel_lexicon")

    texts, y = load_validation()
    LOGGER.info("[42] validation n=%d", len(y))

    # rebuild df counts cho tung alpha (df_by_class khong doi, chi doi smoothing)
    splits = ("train", "dev", "test")
    rows = nb_mod.load_rows(splits)
    LOGGER.info("[42] corpus rows: %d", len(rows))
    df_by_class, df_all, n_rows = nb_mod.count_df(rows)

    verified = pd.read_csv(CURRENT_LEXICON)
    existing = verified["term"].astype(str).tolist()
    word_terms = nb_mod.read_terms([
        nb_mod.CANDIDATE_DIRS[0] / "word_mode_vihsd.txt",
        nb_mod.CANDIDATE_DIRS[0] / "word_mode_vicm.txt",
        nb_mod.CANDIDATE_DIRS[0] / "word_mode_bontori.txt",
    ])
    candidates = list(dict.fromkeys(existing + word_terms))

    alphas = [float(a) for a in args.alphas.split(",")]
    min_probs = [float(p) for p in args.min_probs.split(",")]

    base_cfg = load_rule_engine_config(RULE_CONFIG)
    base = run_lexicon(CURRENT_LEXICON, texts, y, base_cfg)
    LOGGER.info("[42] current baseline: %.4f", base["macro_f1"])

    grid: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for alpha in alphas:
        scored = nb_mod.nb_probabilities(df_by_class, df_all, n_rows, candidates, alpha)
        sdf = pd.DataFrame(scored)
        for mp in min_probs:
            ml = nb_mod.select_multilabel(sdf, mp)
            if not len(ml):
                continue
            tmp_csv = out_dir / f"_tmp_nb_a{alpha}_p{mp}.csv"
            ml.to_csv(tmp_csv, index=False, encoding="utf-8")
            m = run_lexicon(tmp_csv, texts, y, load_rule_engine_config(RULE_CONFIG))
            n_multi = int(ml["lexical_labels"].str.contains(";").sum())
            row = {"alpha": alpha, "min_prob": mp, "entries": m["entries"],
                   "multi_label": n_multi, "macro_f1": round(m["macro_f1"], 6),
                   "delta": round(m["macro_f1"] - base["macro_f1"], 6),
                   "HATE_f1": round(m["HATE_f1"], 4), "OFFENSIVE_f1": round(m["OFFENSIVE_f1"], 4)}
            grid.append(row)
            LOGGER.info("[42]   alpha=%.1f p=%.2f -> %.4f (d=%+.4f, %d entry, %d multi)",
                        alpha, mp, m["macro_f1"], row["delta"], m["entries"], n_multi)
            if best is None or m["macro_f1"] > best["macro_f1"]:
                best = {**row, "path": tmp_csv, "pred": m["pred"]}

    if best is None:
        LOGGER.error("[42] grid rong")
        return 1

    grid_df = pd.DataFrame(grid).sort_values("macro_f1", ascending=False)
    grid_df.to_csv(out_dir / "nb_tuning_grid.csv", index=False, encoding="utf-8")

    # luu lexicon tot nhat
    best_csv = out_dir / "lexicon_nb_best.csv"
    pd.read_csv(best["path"]).to_csv(best_csv, index=False, encoding="utf-8")
    for tmp in out_dir.glob("_tmp_nb_*.csv"):
        tmp.unlink()

    # kiem dinh diem tot nhat vs current
    d = paired_bootstrap_delta(y, best["pred"], base["pred"], metric="macro_f1",
                               n_boot=args.n_boot, seed=42)
    mc = mcnemar(y, best["pred"], base["pred"])
    significant = (d["ci_low"] > 0.0) or (d["ci_high"] < 0.0)
    LOGGER.info("[42] best (a=%.1f p=%.2f): %.4f, d=%+.4f CI[%+.4f,%+.4f] p=%.3f sig=%s",
                best["alpha"], best["min_prob"], best["macro_f1"], d["delta"],
                d["ci_low"], d["ci_high"], d["p_value"], significant)

    lines: List[str] = []
    add = lines.append
    add("# Tune nguong multi-label + smoothing cua lexicon NB\n")
    add(f"> Quet tren **validation** n={len(y)} (plan §44). Moc current = {base['macro_f1']:.4f}.\n")
    add("## Bang quet (sap theo macro-F1)\n")
    add("| alpha | min_prob | Entries | Multi-label | Macro-F1 | Δ vs current |")
    add("|---|---|---|---|---|---|")
    for _, r in grid_df.iterrows():
        add(f"| {r['alpha']:.1f} | {r['min_prob']:.2f} | {int(r['entries'])} | "
            f"{int(r['multi_label'])} | **{r['macro_f1']:.4f}** | {r['delta']:+.4f} |")
    add(f"\n## Diem tot nhat: alpha={best['alpha']:.1f}, min_prob={best['min_prob']:.2f}\n")
    add(f"- Macro-F1 = **{best['macro_f1']:.4f}** (Δ = {best['delta']:+.4f})")
    add(f"- Kiem dinh vs current: Δ = {d['delta']:+.4f}, CI 95% = "
        f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}], p = {d['p_value']:.3f}")
    add(f"- McNemar: {mc['a_correct_b_wrong']} dung / {mc['a_wrong_b_correct']} sai "
        f"(p = {mc['p_value']:.3f})")
    add(f"- Ket luan: {'CO y nghia' if significant else 'khong co y nghia (CI chua 0)'}")
    add("\n## Cach doc\n")
    add("- `min_prob` cao -> it entry hon, it nhan 2 hon; thap -> nhieu entry hon.")
    add("- Neu macro-F1 it doi qua cac diem => NB da bao hoa, sieu tham so khong phai "
        "nguon cai thien chinh.")
    add("- Lexicon tot nhat luu o `lexicon_nb_best.csv`.")
    (out_dir / "nb_tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "nb_tuning.json").write_text(json.dumps(
        {"baseline": base["macro_f1"], "grid": grid,
         "best": {"alpha": best["alpha"], "min_prob": best["min_prob"],
                  "macro_f1": best["macro_f1"], "entries": best["entries"]},
         "significance": {"delta": round(float(d["delta"]), 6),
                          "ci": [round(float(d["ci_low"]), 6), round(float(d["ci_high"]), 6)],
                          "p_value": round(float(d["p_value"]), 6),
                          "right": mc["a_correct_b_wrong"], "wrong": mc["a_wrong_b_correct"],
                          "significant_95": bool(significant)}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[42] wrote %s/nb_tuning.md", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
