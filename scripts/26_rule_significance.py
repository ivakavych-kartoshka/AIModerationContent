"""Step 26 - paired significance test between two rule-engine runs (plan 43-44).

    python scripts/26_rule_significance.py --a outputs/evaluation/rule_engine_tuned/test/predictions.csv --b outputs/evaluation/rule_engine_tuned_p010/test/predictions.csv

Plan section 43 compares systems on the SAME fixed test set, and section 44
refuses any accuracy claim that is not backed by a proper test. This script
takes two runs of ``scripts/17_evaluate_rule_engine.py --split test`` (frozen
thresholds) and reports:

  * macro-F1 / accuracy with a 95% percentile bootstrap CI for each system
  * the paired bootstrap CI and two-sided p-value for ``metric(A) - metric(B)``
  * the exact two-sided McNemar p-value on the per-row correctness vectors

Rows are aligned on ``index`` and ``y_true``; any mismatch aborts, because a
paired test over differently ordered predictions is meaningless.

Outputs
-------
    outputs/evaluation/rule_significance.json
    papers/tables/rule_significance.tex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.analysis import (  # noqa: E402
    bootstrap_ci,
    confusion_from_labels,
    mcnemar,
    metrics_from_confusion,
    paired_bootstrap_delta,
)

LOGGER = get_logger("sensitiveai.scripts.26_rule_significance")

DEFAULT_OUT = Path("outputs/evaluation/rule_significance.json")
DEFAULT_TEX = Path("papers/tables/rule_significance.tex")
METRICS = ("macro_f1", "accuracy")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path) -> Dict[str, Any]:
    """Read one ``predictions.csv`` plus its sibling ``metrics.json``."""
    if not path.is_file():
        raise FileNotFoundError(f"predictions not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    for column in ("index", "y_true", "y_pred"):
        if column not in frame.columns:
            raise ValueError(f"{path} is missing the {column!r} column")
    context: Dict[str, Any] = {}
    metrics_path = path.parent / "metrics.json"
    if metrics_path.is_file():
        source = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("split", "config", "lexicon", "n_lexicon_entries", "thresholds", "objective"):
            if key in source:
                context[key] = source[key]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "n_rows": int(len(frame)),
        "index": frame["index"].to_numpy(),
        "y_true": frame["y_true"].to_numpy(dtype=np.int64),
        "y_pred": frame["y_pred"].to_numpy(dtype=np.int64),
        "context": context,
    }


def align(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    if not np.array_equal(a["index"], b["index"]):
        raise ValueError("runs are not aligned: the 'index' column differs")
    if not np.array_equal(a["y_true"], b["y_true"]):
        raise ValueError("runs are not aligned: the 'y_true' column differs")
    if a["n_rows"] != b["n_rows"]:
        raise ValueError("runs are not aligned: row counts differ")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    full = metrics_from_confusion(confusion_from_labels(y_true, y_pred))
    return {key: float(full[key]) for key in ("macro_f1", "accuracy", "HATE_f1", "OFFENSIVE_f1", "CLEAN_f1")}


def analyse(
    a: Dict[str, Any],
    b: Dict[str, Any],
    label_a: str,
    label_b: str,
    n_boot: int,
    seed: int,
    alpha: float,
) -> Dict[str, Any]:
    y_true, pred_a, pred_b = a["y_true"], a["y_pred"], b["y_pred"]

    payload: Dict[str, Any] = {
        "comparison": f"{label_a} vs {label_b}",
        "n_test": int(y_true.shape[0]),
        "protocol": {
            "split": a["context"].get("split", "test"),
            "thresholds": "frozen (tuned on validation by scripts/17 --split validation)",
            "n_boot": n_boot,
            "seed": seed,
            "alpha": alpha,
            "source": "scripts/17_evaluate_rule_engine.py --split test -> predictions.csv",
        },
        "a": {"label": label_a, "path": a["path"], "sha256": a["sha256"], **a["context"]},
        "b": {"label": label_b, "path": b["path"], "sha256": b["sha256"], **b["context"]},
        "metrics": {"a": _metrics(y_true, pred_a), "b": _metrics(y_true, pred_b)},
        "bootstrap_ci": {},
        "paired": {},
    }

    for metric in METRICS:
        payload["bootstrap_ci"][metric] = {
            "a": bootstrap_ci(y_true, pred_a, metric=metric, n_boot=n_boot, alpha=alpha, seed=seed),
            "b": bootstrap_ci(y_true, pred_b, metric=metric, n_boot=n_boot, alpha=alpha, seed=seed),
        }
        payload["paired"][metric] = paired_bootstrap_delta(
            y_true, pred_a, pred_b, metric=metric, n_boot=n_boot, alpha=alpha, seed=seed
        )

    payload["mcnemar"] = mcnemar(y_true, pred_a, pred_b)

    # Verdict (section 44: no claim without a test).
    f1 = payload["paired"]["macro_f1"]
    acc = payload["paired"]["accuracy"]
    verdict = []
    for name, block in (("macro-F1", f1), ("accuracy", acc)):
        if block["p_value"] < 0.05 and block["ci_low"] * block["ci_high"] > 0:
            direction = "higher" if block["delta"] > 0 else "lower"
            verdict.append(f"{label_a} is significantly {direction} than {label_b} on {name} (p={block['p_value']:.4f})")
        else:
            verdict.append(f"no significant {name} difference between {label_a} and {label_b} (p={block['p_value']:.4f})")
    payload["verdict"] = verdict
    return payload


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def write_tex(payload: Dict[str, Any], path: Path) -> Path:
    a, b = payload["a"], payload["b"]
    ma, mb = payload["metrics"]["a"], payload["metrics"]["b"]
    ci = payload["bootstrap_ci"]
    paired = payload["paired"]
    n = payload["n_test"]
    mcn = payload["mcnemar"]

    def cell(metric: str, side: str) -> str:
        block = ci[metric][side]
        return f"{block['point']:.4f} [{block['ci_low']:.4f}, {block['ci_high']:.4f}]"

    def pval(p: float) -> str:
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    rows = [
        "% AUTO-GENERATED by scripts/26_rule_significance.py - do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        ("\\caption{Lexicon-size ablation of the lexical rule engine on the frozen test split "
         f"({n:,} items; {a.get('n_lexicon_entries', '?')} vs {b.get('n_lexicon_entries', '?')} dictionary entries; "
         "thresholds tuned on validation and frozen). Accuracy and macro-F1 carry 95\\% percentile "
         f"bootstrap intervals ({payload['protocol']['n_boot']:,} resamples). "
         "$A-B$ rows give the paired-bootstrap difference with its 95\\% interval and $p_{\\text{F1}}$; "
         "$p_{\\text{acc}}$ is the exact two-sided McNemar $p$-value. "
         + ("No difference is significant at $\\alpha=0.05$."
            if all(v.startswith("no significant") for v in payload["verdict"])
            else "See the verdict in the text.")
         + "}"),
        "\\label{tab:rule-pruning-significance}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "System & Acc. [95\\% CI] & M-F1 [95\\% CI] & Entries & \\\\",
        "\\midrule",
        f"{a['label']} ($A$) & {cell('accuracy', 'a')} & {cell('macro_f1', 'a')} & "
        f"{a.get('n_lexicon_entries', '?')} & \\\\",
        f"{b['label']} ($B$) & {cell('accuracy', 'b')} & {cell('macro_f1', 'b')} & "
        f"{b.get('n_lexicon_entries', '?')} & \\\\",
        "\\midrule",
        "$A-B$ & $\\Delta$Acc [95\\% CI] & $p_{\\text{acc}}$ & "
        "$\\Delta$M-F1 [95\\% CI] & $p_{\\text{F1}}$ \\\\",
        "\\midrule",
        (f"{a['label']}$-${b['label']} & {_fmt(paired['accuracy']['delta'])} "
         f"[{_fmt(paired['accuracy']['ci_low'])}, {_fmt(paired['accuracy']['ci_high'])}] & "
         f"{pval(mcn['p_value'])} & {_fmt(paired['macro_f1']['delta'])} "
         f"[{_fmt(paired['macro_f1']['ci_low'])}, {_fmt(paired['macro_f1']['ci_high'])}] & "
         f"{pval(paired['macro_f1']['p_value'])} \\\\"),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paired significance test between two rule-engine runs.")
    p.add_argument("--a", required=True, help="predictions.csv of system A (reference, e.g. the full lexicon).")
    p.add_argument("--b", required=True, help="predictions.csv of system B (e.g. the pruned lexicon).")
    p.add_argument("--label-a", default=None, help="Display label for A (default: parent folder name).")
    p.add_argument("--label-b", default=None, help="Display label for B (default: parent folder name).")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Where the JSON report is written.")
    p.add_argument("--tex", default=str(DEFAULT_TEX), help="Where the paper table is written.")
    p.add_argument("--no-tex", action="store_true", help="Skip the LaTeX table.")
    p.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples.")
    p.add_argument("--seed", type=int, default=42, help="Bootstrap seed (paired across systems).")
    p.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI level complement.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    a = load_run(Path(args.a))
    b = load_run(Path(args.b))
    align(a, b)

    label_a = args.label_a or Path(args.a).parent.parent.name or "A"
    label_b = args.label_b or Path(args.b).parent.parent.name or "B"
    LOGGER.info("[26] comparing %s vs %s on %d aligned rows", label_a, label_b, a["n_rows"])

    payload = analyse(a, b, label_a, label_b, args.n_boot, args.seed, args.alpha)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[26] wrote %s", out)

    if not args.no_tex:
        tex = write_tex(payload, Path(args.tex))
        LOGGER.info("[26] wrote %s", tex)

    for metric in METRICS:
        block = payload["paired"][metric]
        LOGGER.info(
            "[26] %-10s delta=%+.4f ci=[%+.4f, %+.4f] p=%.4f",
            metric, block["delta"], block["ci_low"], block["ci_high"], block["p_value"],
        )
    LOGGER.info(
        "[26] McNemar: A-right/B-wrong=%d A-wrong/B-right=%d discordant=%d p=%.4f",
        payload["mcnemar"].get("a_correct_b_wrong", 0),
        payload["mcnemar"].get("a_wrong_b_correct", 0),
        payload["mcnemar"].get("discordant", 0),
        payload["mcnemar"].get("p_value", float("nan")),
    )
    for line in payload["verdict"]:
        LOGGER.info("[26] %s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
