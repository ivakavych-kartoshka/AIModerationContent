"""Step 24 - obfuscation robustness: rule engine vs AI vs hybrid fusion (plan 44).

    python scripts/24_robustness_comparison.py
    python scripts/24_robustness_comparison.py --models sensitiveai-vi-customdlr2stage
    python scripts/24_robustness_comparison.py --modes clean leet no_accent

Why this experiment
-------------------
The main paper already reports that the neural systems are weakest on comments
*without diacritics* and on *teencode* text (Sect. Results, per-class behaviour).
The lexical rule engine was built for exactly that regime (accentless / spaced /
leetspeak alias views), so the interesting question is whether the two signal
sources *complement* each other when the input is deliberately obfuscated - and
whether a hybrid model keeps the AI's accuracy while keeping the rule's
resilience.

Protocol (plan section 44, identical to ``scripts/17_evaluate_rule_engine.py``)
-------------------------------------------------------------------------------
* the obfuscation transformations are imported from script 17 so the perturbed
  inputs are byte-for-byte the same ones the rule-engine robustness report used;
* every system uses thresholds/alpha frozen on VALIDATION - nothing here is
  tuned on the test split;
* two readouts per (system, mode):
    - ``macro-F1`` / accuracy / per-class F1 over the whole test split (headline),
    - ``detection_rate`` over the first 400 non-CLEAN test comments, i.e. the
      fraction still flagged as HATE/OFFENSIVE - directly comparable with the
      existing ``rule_engine*/test/robustness.json`` numbers.

Outputs
-------
    outputs/evaluation/robustness/robustness_summary.json
    outputs/evaluation/robustness/robustness_report.md
    papers/tables/robustness.tex
    papers/figures/robustness.png
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.common.constants import LABELS, SPLITS_DIR  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.common.paths import ensure_dir, per_model_eval_dir  # noqa: E402
from src.evaluation.metrics import apply_decision_rule, compute_metrics, run_inference  # noqa: E402
from src.evaluation.run_evaluation import (  # noqa: E402
    PRECISION_TO_DTYPE,
    load_model_for_eval,
)
from src.evaluation.thresholds import load_thresholds  # noqa: E402
from src.fusion.run_fusion import resolve_ai_source  # noqa: E402
from src.fusion.weighted_fusion import fuse_scores_per_channel  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.24_robustness")

#: Perturbation modes, in report order; ``clean`` is the unperturbed control.
MODES: Tuple[str, ...] = ("clean", "leet", "spaced", "no_accent", "repeat")
#: The rule-engine robustness report scored the first 400 positive comments.
POSITIVE_LIMIT = 400
CLEAN_ID = LABELS.index("CLEAN")

DEFAULT_RULE_EVAL_DIR = Path("outputs/evaluation/rule_engine_tuned")
DEFAULT_FUSION_ROOT = Path("outputs/evaluation/fusion")
DEFAULT_MODELS = ("sensitiveai-vi-customdlr2stage", "sensitiveai-vi-stage2")
PAPER_DIR = Path(__file__).resolve().parent.parent / "papers"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-tty
            pass


def load_script17_obfuscate():
    """Import ``obfuscate`` straight from script 17.

    The perturbations must be byte-for-byte identical to the ones used for the
    rule-engine robustness report, so they are imported rather than re-typed.
    """
    path = Path(__file__).resolve().parent / "17_evaluate_rule_engine.py"
    spec = importlib.util.spec_from_file_location("rule_engine_script17", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.obfuscate


def load_split(path: Path) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.asarray(labels, dtype=int)


# --------------------------------------------------------------------------- #
# AI inference over a *perturbed* text list
# --------------------------------------------------------------------------- #
class _EncodedTexts(Dataset):
    """Minimal tokenized-text dataset with the batch shape ``run_inference`` expects."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: np.ndarray):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def _collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def predict_proba(
    model,
    tokenizer,
    texts: Sequence[str],
    y_true: np.ndarray,
    device: torch.device,
    max_length: int,
    batch_size: int,
    amp_dtype=None,
) -> np.ndarray:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = _EncodedTexts(encoded["input_ids"], encoded["attention_mask"], y_true)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    outputs = run_inference(model, loader, device, amp_dtype=amp_dtype)
    return outputs["y_proba"]


# --------------------------------------------------------------------------- #
# system definitions (all parameters frozen on validation)
# --------------------------------------------------------------------------- #
def load_ai_system(model_name: str, reports_root: Path, device: torch.device):
    """Return (model, tokenizer, cfg, thresholds|None) for one trained AI model."""
    from transformers import AutoTokenizer

    _, checkpoint, cfg, run_id = resolve_ai_source(model_name, reports_root)
    model = load_model_for_eval(cfg, checkpoint).to(device).eval()
    tokenizer_dir = checkpoint if (checkpoint / "tokenizer_config.json").is_file() else cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    thresholds = load_thresholds(per_model_eval_dir(model_name) / "thresholds.json")
    LOGGER.info("[24] %s run=%s thresholds=%s", model_name, run_id, thresholds)
    return model, tokenizer, cfg, thresholds


def load_fusion_config(model_name: str, fusion_root: Path) -> Optional[Dict[str, object]]:
    path = fusion_root / model_name / "metrics.json"
    if not path.is_file():
        LOGGER.warning("[24] no fusion metrics for %s at %s - skipping fusion", model_name, path)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = payload.get("primary_variant") or ""
    if "per_channel" in primary:
        alphas = [float(payload["per_channel_alphas"][label]) for label in LABELS]
        thresholds = [float(payload["per_channel_thresholds"][label]) for label in LABELS]
    else:
        alpha = float(payload.get("alpha", 1.0))
        alphas = [alpha] * len(LABELS)
        thresholds = (
            [float(payload["thresholds"][label]) for label in LABELS]
            if payload.get("thresholds")
            else None
        )
    if thresholds is None:
        LOGGER.warning("[24] %s: fusion has no frozen thresholds - skipping fusion", model_name)
        return None
    return {"alphas": alphas, "thresholds": thresholds, "primary_variant": primary}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def decision_from_proba(
    proba: np.ndarray, thresholds: Optional[Sequence[float]]
) -> np.ndarray:
    if thresholds is None:
        return proba.argmax(axis=1)
    return apply_decision_rule(proba, "thresholds", list(thresholds))


def positive_detection_rate(y_true: np.ndarray, y_pred: np.ndarray, limit: int = POSITIVE_LIMIT) -> Dict[str, object]:
    """Detection rate over the first ``limit`` non-CLEAN comments (script 17 protocol)."""
    positive_idx = np.flatnonzero(y_true != CLEAN_ID)[:limit]
    if positive_idx.size == 0:  # pragma: no cover - defensive
        return {"detected": 0, "total": 0, "detection_rate": 0.0}
    detected = int(np.sum(y_pred[positive_idx] != CLEAN_ID))
    return {
        "detected": detected,
        "total": int(positive_idx.size),
        "detection_rate": round(detected / int(positive_idx.size), 4),
    }


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict[str, object]:
    metrics = compute_metrics(y_true, y_pred, proba)
    return {
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "HATE_f1": round(float(metrics["HATE_f1"]), 4),
        "OFFENSIVE_f1": round(float(metrics["OFFENSIVE_f1"]), 4),
        "CLEAN_f1": round(float(metrics["CLEAN_f1"]), 4),
        "detection": positive_detection_rate(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# main experiment
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reports_root = Path(args.reports_dir)
    obfuscate = load_script17_obfuscate()

    texts, y_true = load_split(Path(SPLITS_DIR) / "test.csv")
    LOGGER.info("[24] test split: %d comments, %d positive", len(texts), int(np.sum(y_true != CLEAN_ID)))

    # ---- rule engine (frozen; optional) ----------------------------------- #
    # The rule engine needs its lexicon on disk.  When it is unavailable (e.g.
    # the source CSV was archived away) the experiment degrades gracefully to an
    # AI-only robustness run rather than aborting - the AI half needs no lexicon.
    rule_eval_dir = Path(args.rule_eval_dir)
    engine = None
    rule_thresholds = None
    try:
        rule_metrics = json.loads((rule_eval_dir / "metrics.json").read_text(encoding="utf-8"))
        rule_config = load_rule_engine_config(rule_metrics.get("config", "configs/rule_engine.yaml"))
        engine = RuleEngine.from_lexicon(rule_config.lexicon_path, rule_config)
        rule_thresholds = load_thresholds(rule_eval_dir / "thresholds.json")
        if rule_thresholds is None:
            raise SystemExit(f"Frozen rule thresholds missing in {rule_eval_dir} (plan §44).")
        LOGGER.info("[24] rule source: %s thresholds=%s", rule_eval_dir, rule_thresholds)
    except Exception as exc:
        engine = None
        rule_thresholds = None
        LOGGER.warning(
            "[24] rule engine unavailable (%s: %s) - running AI-only robustness",
            type(exc).__name__, exc,
        )

    # ---- AI + fusion systems (frozen) ------------------------------------ #
    models: List[str] = list(args.models)
    ai_systems: Dict[str, Dict[str, object]] = {}
    fusion_systems: Dict[str, Dict[str, object]] = {}
    for model_name in models:
        model, tokenizer, cfg, ai_thresholds = load_ai_system(model_name, reports_root, device)
        ai_systems[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "max_length": int(cfg.get("tokenizer", {}).get("max_length", 256)),
            "precision": str(cfg.get("training", {}).get("precision", "fp32")).lower(),
            "thresholds": ai_thresholds,
        }
        if engine is None:
            continue
        fusion_cfg = load_fusion_config(model_name, Path(args.fusion_root))
        if fusion_cfg is not None:
            fusion_systems[model_name] = fusion_cfg

    # ---- run every perturbation ------------------------------------------ #
    system_names: List[str] = (
        (["rule"] if engine is not None else [])
        + [f"ai::{m}" for m in models]
        + [f"fusion::{m}" for m in fusion_systems]
    )
    results: Dict[str, Dict[str, Dict[str, object]]] = {name: {} for name in system_names}
    timings: Dict[str, float] = {}

    for mode in args.modes:
        variants = [text if mode == "clean" else obfuscate(text, mode) for text in texts]
        LOGGER.info("[24] mode=%s", mode)

        rule_proba = None
        if engine is not None:
            started = time.perf_counter()
            rule_proba, _, _ = run_engine(engine, variants)
            if "rule" not in timings:
                timings["rule"] = round((time.perf_counter() - started) * 1000.0 / max(len(variants), 1), 4)
            results["rule"][mode] = metrics_block(
                y_true, decision_from_proba(rule_proba, rule_thresholds), rule_proba
            )

        for model_name, system in ai_systems.items():
            started = time.perf_counter()
            ai_proba = predict_proba(
                system["model"],
                system["tokenizer"],
                variants,
                y_true,
                device,
                system["max_length"],
                args.batch_size,
                amp_dtype=PRECISION_TO_DTYPE.get(system["precision"]),
            )
            if model_name not in timings:
                timings[model_name] = round(
                    (time.perf_counter() - started) * 1000.0 / max(len(variants), 1), 4
                )
            results[f"ai::{model_name}"][mode] = metrics_block(
                y_true, decision_from_proba(ai_proba, system["thresholds"]), ai_proba
            )

            fusion_cfg = fusion_systems.get(model_name)
            if fusion_cfg is not None:
                fused = fuse_scores_per_channel(ai_proba, rule_proba[:, 0], rule_proba[:, 1], fusion_cfg["alphas"])
                results[f"fusion::{model_name}"][mode] = metrics_block(
                    y_true, decision_from_proba(fused, fusion_cfg["thresholds"]), fused
                )

    # ---- drop vs clean ---------------------------------------------------- #
    for name, per_mode in results.items():
        if "clean" not in per_mode:
            continue
        clean_f1 = per_mode["clean"]["macro_f1"]
        clean_det = per_mode["clean"]["detection"]["detection_rate"]
        for mode, block in per_mode.items():
            block["macro_f1_drop_vs_clean"] = round(block["macro_f1"] - clean_f1, 4)
            block["detection_drop_vs_clean"] = round(
                block["detection"]["detection_rate"] - clean_det, 4
            )

    payload = {
        "split": "test",
        "n_comments": len(texts),
        "n_positive": int(np.sum(y_true != CLEAN_ID)),
        "positive_limit": POSITIVE_LIMIT,
        "modes": list(args.modes),
        "rule_eval_dir": str(rule_eval_dir),
        "models": models,
        "protocol": (
            "obfuscations imported from scripts/17_evaluate_rule_engine.py; all "
            "thresholds/alphas frozen on validation; test is scored once (plan §44)."
        ),
        "latency_ms_per_comment": timings,
        "systems": results,
    }
    out_dir = ensure_dir(Path(args.out_dir))
    summary_path = out_dir / "robustness_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[24] summary -> %s", summary_path)

    report = write_report(out_dir, payload)
    LOGGER.info("[24] report -> %s", report)
    write_paper_table(payload, PAPER_DIR / "tables" / "robustness.tex")
    try:
        write_paper_figure(payload, PAPER_DIR / "figures" / "robustness.png")
    except Exception as exc:  # pragma: no cover - matplotlib optional
        LOGGER.warning("[24] figure skipped: %s", exc)
    LOGGER.info("[24] paper table -> %s", PAPER_DIR / "tables" / "robustness.tex")
    return 0


def run_engine(engine: RuleEngine, texts: Sequence[str]):
    hate = np.zeros(len(texts))
    offensive = np.zeros(len(texts))
    for index, text in enumerate(texts):
        result = engine.analyze(text)
        hate[index] = float(result["rule_hate_score"])
        offensive[index] = float(result["rule_offensive_score"])
    from src.fusion.weighted_fusion import rule_pseudo_proba

    return rule_pseudo_proba(hate, offensive), None, None


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
def _fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def write_report(out_dir: Path, payload: Dict[str, object]) -> Path:
    lines: List[str] = [
        "# Obfuscation robustness - rule engine vs AI vs hybrid",
        "",
        f"Test split: {payload['n_comments']} comments, {payload['n_positive']} non-CLEAN.",
        f"Perturbations imported from `scripts/17_evaluate_rule_engine.py`; "
        f"positive-subset protocol: first {payload['positive_limit']} non-CLEAN comments.",
        "All thresholds/alphas frozen on validation (plan §44).",
        "",
        "## Macro-F1 on the full test split (higher = more robust)",
        "",
        "| system | " + " | ".join(payload["modes"]) + " | Δ worst |",
        "|---|" + "---|" * (len(payload["modes"]) + 1),
    ]
    for name, per_mode in payload["systems"].items():
        row = [_fmt(per_mode[mode]["macro_f1"]) for mode in payload["modes"] if mode in per_mode]
        drops = [per_mode[mode]["macro_f1_drop_vs_clean"] for mode in payload["modes"] if mode in per_mode]
        lines.append(f"| {name} | " + " | ".join(row) + f" | {min(drops):+.4f} |")

    lines += [
        "",
        f"## Detection rate on the first {payload['positive_limit']} non-CLEAN comments",
        "",
        "(still flagged HATE/OFFENSIVE; matches `rule_engine*/test/robustness.json`)",
        "",
        "| system | " + " | ".join(payload["modes"]) + " |",
        "|---|" + "---|" * len(payload["modes"]),
    ]
    for name, per_mode in payload["systems"].items():
        row = [
            _fmt(per_mode[mode]["detection"]["detection_rate"])
            for mode in payload["modes"]
            if mode in per_mode
        ]
        lines.append(f"| {name} | " + " | ".join(row) + " |")

    lines += ["", "## Per-class F1 (full test split)", ""]
    for mode in payload["modes"]:
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| system | macro-F1 | HATE | OFFENSIVE | CLEAN |")
        lines.append("|---|---|---|---|---|")
        for name, per_mode in payload["systems"].items():
            if mode not in per_mode:
                continue
            block = per_mode[mode]
            lines.append(
                f"| {name} | {_fmt(block['macro_f1'])} | {_fmt(block['HATE_f1'])} | "
                f"{_fmt(block['OFFENSIVE_f1'])} | {_fmt(block['CLEAN_f1'])} |"
            )
        lines.append("")

    lines += [
        "## Inference latency (ms/comment, fp32, indicative only)",
        "",
        "| system | ms/comment |",
        "|---|---|",
    ]
    for name, value in payload["latency_ms_per_comment"].items():
        lines.append(f"| {name} | {value:.4f} |")
    lines.append("")

    path = out_dir / "robustness_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_paper_table(payload: Dict[str, object], path: Path) -> Path:
    modes = list(payload["modes"])
    lines = [
        "% AUTO-GENERATED by scripts/24_robustness_comparison.py - do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Obfuscation robustness on the ViHSD test split: macro-F1 after",
        "deliberately perturbing every comment (Sect.~\\ref{sec:robustness}).  Every",
        "system keeps the thresholds/weights frozen on validation, so each column is one",
        "scoring pass; the control column is the unperturbed split.}",
        "\\label{tab:robustness}",
        "\\begin{tabular}{l" + "c" * len(modes) + "c}",
        "\\toprule",
        "System & " + " & ".join(m.replace("_", "-") for m in modes) + " & min $\\Delta$ \\\\",
        "\\midrule",
    ]
    for name, per_mode in payload["systems"].items():
        if not all(mode in per_mode for mode in modes):
            continue
        cells = [_fmt(per_mode[mode]["macro_f1"]) for mode in modes]
        drop = min(per_mode[mode]["macro_f1_drop_vs_clean"] for mode in modes)
        pretty = name.replace("::", " + ").replace("sensitiveai-vi", "")
        pretty = {
            "rule": "Rule engine",
        }.get(pretty.strip(), pretty)
        lines.append(f"{pretty} & " + " & ".join(cells) + f" & {drop:+.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_paper_figure(payload: Dict[str, object], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(payload["modes"])
    system_names = [
        name
        for name in payload["systems"]
        if all(mode in payload["systems"][name] for mode in modes)
    ]
    x = np.arange(len(modes))
    width = 0.8 / max(len(system_names), 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for index, name in enumerate(system_names):
        per_mode = payload["systems"][name]
        values = [per_mode[mode]["macro_f1"] for mode in modes]
        offset = (index - (len(system_names) - 1) / 2) * width
        label = name.replace("::", " + ").replace("sensitiveai-vi", "")
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x, [m.replace("_", "-") for m in modes])
    ax.set_ylabel("Macro-F1 (full test split)")
    ax.set_ylim(0.0, 0.75)
    ax.set_title("Robustness to obfuscation: rule engine, AI and hybrid")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/24_robustness_comparison.py",
        description="Obfuscation robustness: rule engine vs AI vs hybrid fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="AI models to test")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--rule-eval-dir", default=str(DEFAULT_RULE_EVAL_DIR))
    parser.add_argument("--fusion-root", default=str(DEFAULT_FUSION_ROOT))
    parser.add_argument("--out-dir", default="outputs/evaluation/robustness")
    parser.add_argument("--reports-dir", default="outputs/reports")
    parser.add_argument("--device", default=None, help="force 'cuda' or 'cpu'")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_utf8_console()
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
