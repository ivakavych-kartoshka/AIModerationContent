"""Step 25 - paired significance between AI-only and hybrid fusion (plan 43-44).

    python scripts/25_fusion_significance.py
    python scripts/25_fusion_significance.py --models sensitiveai-vi-stage2

Outputs
-------
    outputs/evaluation/fusion/significance_significance.json
    outputs/evaluation/fusion/significance_report.md
    papers/tables/fusion_significance.tex
""" 

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.constants import LABEL2ID, LABELS  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.common.paths import ensure_dir, per_model_eval_dir  # noqa: E402
from src.evaluation.analysis import (  # noqa: E402
    mcnemar,
    paired_bootstrap_delta,
)
from src.evaluation.metrics import apply_decision_rule  # noqa: E402
from src.evaluation.thresholds import load_thresholds  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.25_significance")

FUSION_ROOT = Path("outputs/evaluation/fusion")
PER_MODEL_ROOT = Path("outputs/evaluation/per_model")
PAPER_DIR = Path("papers")


def _fmt(v: object) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "-" 


def _compute_ai_pred_from_proba(proba: np.ndarray, thresholds: Optional[Sequence[float]]) -> np.ndarray:
    if thresholds is None:
        return proba.argmax(axis=1)
    return apply_decision_rule(proba, "thresholds", list(thresholds))


def _to_ids(series: "pd.Series") -> np.ndarray:
    """Fusion / per-model CSVs store label NAMES; the tests need integer ids."""
    if pd.api.types.is_numeric_dtype(series):
        return series.values.astype(int)
    values: List[int] = []
    for value in series:
        if isinstance(value, str) and value in LABEL2ID:
            values.append(int(LABEL2ID[value]))
        else:
            values.append(int(value))
    return np.asarray(values, dtype=int)


def load_systems(model_name: str) -> Dict[str, np.ndarray]:
    res = {}
    # fusion predictions
    fusion_path = FUSION_ROOT / model_name / "fusion_predictions.csv"
    if not fusion_path.is_file():
        raise FileNotFoundError(f"fusion predictions not found: {fusion_path}")
    df_f = pd.read_csv(fusion_path)
    # per-model AI test predictions (with probas)
    pm_path = PER_MODEL_ROOT / model_name / "test_predictions.csv"
    if not pm_path.is_file():
        raise FileNotFoundError(f"per-model predictions not found: {pm_path}")
    df_pm = pd.read_csv(pm_path)

    # align by index
    df_f = df_f.set_index("index")
    df_pm = df_pm.set_index("index")
    common = df_f.index.intersection(df_pm.index)
    df_f = df_f.loc[common]
    df_pm = df_pm.loc[common]

    res["y_true"] = _to_ids(df_f["true_label"])

    # fusion pred (primary variant chosen on validation)
    res["pred_fusion"] = _to_ids(df_f["fused_pred"])

    # AI pred: per-model frozen thresholds when the evaluator wrote them, else argmax.
    ai_proba = df_pm[["prob_HATE", "prob_OFFENSIVE", "prob_CLEAN"]].values
    th_path = per_model_eval_dir(model_name) / "thresholds.json"
    thresholds = load_thresholds(th_path) if th_path.is_file() else None
    res["pred_ai"] = _compute_ai_pred_from_proba(ai_proba, thresholds)

    return res


def _compute_metrics_report(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        delta = paired_bootstrap_delta(y_true, pred_a, pred_b, metric="macro_f1", n_boot=10000, seed=42)
        out["paired_bootstrap_macro_f1"] = {
            "delta(A_minus_B)": round(float(delta["delta"]), 4),
            "delta_ci_low": round(float(delta["ci_low"]), 4),
            "delta_ci_high": round(float(delta["ci_high"]), 4),
            "p": round(float(delta["p_value"]), 6),
            "n_boot": int(delta.get("n_boot", 0)),
        }
    except Exception as exc:
        LOGGER.warning("paired bootstrap failed: %s", exc)
        out["paired_bootstrap_macro_f1"] = {"error": str(exc)}
    try:
        m = mcnemar(y_true, pred_a, pred_b)
        out["mcnemar"] = {
            "b": m["a_correct_b_wrong"],
            "c": m["a_wrong_b_correct"],
            "n": m["discordant"],
            "exact_p": round(float(m["p_value"]), 6),
            "chi2_p": round(float(m["p_value_chi2"]), 6),
        }
    except Exception as exc:
        LOGGER.warning("mcnemar failed: %s", exc)
        out["mcnemar"] = {"error": str(exc)}
    return out


def write_report(out_dir: Path, payload: Dict[str, object]) -> Path:
    lines = [
        "# Paired significance: AI-only (frozen thresholds) vs hybrid fusion",
        "",
        f"Models: {', '.join(payload['models'])}",
        f"Test samples: {payload.get('n_samples', '-')}",
        "",
    ]
    for model in payload["models"]:
        res = payload["results"].get(model, {})
        lines.append(f"## {model}")
        lines.append("")
        if "error" in res:
            lines.append(f"- FAILED: {res['error']}")
            lines.append("")
            continue
        pb = res.get("paired_bootstrap_macro_f1", {})
        if "error" not in pb and pb:
            lines.append(
                f"- Paired bootstrap (macro-F1, A=AI, B=fusion, Δ=A-B): {pb['delta(A_minus_B)']:.4f} "
                f"[{pb['delta_ci_low']:.4f}, {pb['delta_ci_high']:.4f}], p={pb['p']:.6f}"
            )
        else:
            lines.append(f"- Paired bootstrap: {pb.get('error', 'not computed')}")
        mc = res.get("mcnemar", {})
        if "error" not in mc and mc:
            lines.append(
                f"- McNemar (correctness): b={mc['b']} (A right, B wrong), c={mc['c']} (A wrong, B right), n={mc['n']}, exact p={mc['exact_p']:.6f}"
            )
        else:
            lines.append(f"- McNemar: {mc.get('error', 'not computed')}")
        lines.append("")
    path = out_dir / "significance_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_tex(payload: Dict[str, object], path: Path) -> Path:
    models = payload["models"]
    lines = [
        "% AUTO-GENERATED by scripts/25_fusion_significance.py - do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Paired statistical tests between AI-only (frozen thresholds) and hybrid fusion on the test split: "
        "paired bootstrap for macro-F1 difference ($\\Delta = \\text{AI}-\\text{Fusion}$) and McNemar on correctness. "
        "Negative $\\Delta$ means fusion improves macro-F1.}",
        "\\label{tab:fusion-significance}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & $\\Delta_{\\text{macroF1}}$ & 95\\% CI ($\\Delta$) & $p_{\\text{boot}}$ & $b$ ($A\\!+\\!,B\\!-$) & $c$ ($A\\!-,B\\!+$) & $p_{\\text{McNemar}}$ \\\\",
        "\\midrule",
    ]
    for model in models:
        res = payload["results"].get(model, {})
        pb = res.get("paired_bootstrap_macro_f1", {})
        mc = res.get("mcnemar", {})
        if "error" in res or not pb or "error" in pb:
            row = [f"{model}", "--", "--", "--"]
        else:
            row = [f"{model}", f"{pb['delta(A_minus_B)']:.4f}", f"[{pb['delta_ci_low']:.4f}, {pb['delta_ci_high']:.4f}]", f"{pb['p']:.4f}"]
        if "error" in res or not mc or "error" in mc:
            row.extend(["--", "--", "--"])
        else:
            row.extend([str(mc["b"]), str(mc["c"]), f"{mc['exact_p']:.4f}"])
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _discover_models() -> List[str]:
    """Every model that has fusion output available (used when --models is omitted)."""
    if not FUSION_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in FUSION_ROOT.iterdir() if (p / "fusion_predictions.csv").is_file()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python scripts/25_fusion_significance.py",
        description="Paired bootstrap + McNemar comparing AI-only vs hybrid fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="models with fusion output (default: every model under outputs/evaluation/fusion)",
    )
    parser.add_argument("--out-dir", default="outputs/evaluation/fusion")
    args = parser.parse_args(argv)

    models = list(args.models) if args.models else _discover_models()
    if not models:
        LOGGER.error("[25] no fusion output found under %s - run scripts/23_evaluate_fusion.py first", FUSION_ROOT)
        return 1
    results = {}
    n_samples = None
    for model in models:
        try:
            data = load_systems(model)
            if n_samples is None:
                n_samples = len(data["y_true"])
            r = _compute_metrics_report(data["y_true"], data["pred_ai"], data["pred_fusion"])
            results[model] = r
            LOGGER.info("[25] %s done", model)
        except Exception as exc:
            LOGGER.error("[25] %s failed: %s", model, exc)
            results[model] = {"error": str(exc)}

    payload = {
        "models": models,
        "n_samples": int(n_samples) if n_samples is not None else None,
        "results": results,
    }
    out_dir = ensure_dir(Path(args.out_dir))
    (out_dir / "fusion_significance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, payload)
    write_tex(payload, PAPER_DIR / "tables" / "fusion_significance.tex")
    LOGGER.info("[25] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
