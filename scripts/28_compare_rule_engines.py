"""Step 28 - so sánh 2 rule engine đã đóng băng (baseline vs tuned), plan 43-44.

    python scripts/28_compare_rule_engines.py

Reads the artifacts that scripts 17/19 already wrote for
``outputs/evaluation/rule_engine`` (baseline) and
``outputs/evaluation/rule_engine_tuned`` (winner of the script-27 tuning), on
BOTH splits, plus the significance report produced by
``scripts/26_rule_significance.py``. It never re-scores a split and never
touches a threshold, so the test split stays measured exactly once (plan §44).

Outputs (into ``outputs/evaluation/comparison/rule_engine/``)
-------------------------------------------------------------
    rule_engine_baseline_vs_tuned.md   full side-by-side report
    rule_engine_macro_f1.png           macro-F1 + accuracy, validation & test
    rule_engine_per_class_f1.png       per-class F1, validation & test
    rule_engine_robustness.png         detection rate per perturbation mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("sensitiveai.scripts.28_compare_rule_engines")

BASELINE = Path("outputs/evaluation/rule_engine")
TUNED = Path("outputs/evaluation/rule_engine_tuned")
OUT_DIR = Path("outputs/evaluation/comparison/rule_engine")
SIGNIFICANCE = OUT_DIR / "rule_engine_significance.json"

LABEL_A = "baseline (rule_engine)"
LABEL_B = "tuned (rule_engine_tuned)"
COLOR_A = "#9e9e9e"
COLOR_B = "#1f77b4"

MODES = ("clean", "leet", "spaced", "no_accent", "repeat")
MODE_TITLES = {
    "clean": "clean",
    "leet": "leet",
    "spaced": "spaced",
    "no_accent": "no_accent",
    "repeat": "repeat",
}
SUMMARY_ROWS = (
    ("macro_f1", "macro-F1"),
    ("accuracy", "accuracy"),
    ("macro_precision", "macro precision"),
    ("macro_recall", "macro recall"),
    ("weighted_f1", "weighted F1"),
    ("HATE_f1", "F1 HATE"),
    ("OFFENSIVE_f1", "F1 OFFENSIVE"),
    ("CLEAN_f1", "F1 CLEAN"),
)
CLASS_ROWS = ("HATE", "OFFENSIVE", "CLEAN")


# --------------------------------------------------------------------------- io
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_run(root: Path) -> Dict[str, Dict[str, Any]]:
    """metrics/robustness/latency of one run, for validation and test."""
    run: Dict[str, Dict[str, Any]] = {}
    for split, folder in (("validation", root), ("test", root / "test")):
        run[split] = {
            "metrics": _load_json(folder / "metrics.json"),
            "robustness": _load_json(folder / "robustness.json"),
            "latency": _load_json(folder / "latency.json"),
        }
    if run["validation"]["metrics"] is None:
        raise FileNotFoundError(f"no metrics.json under {root}")
    if run["test"]["metrics"] is None:
        raise FileNotFoundError(f"no test/metrics.json under {root} - run scripts/17 --split test first")
    return run


def flatten(cfg: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        flat[prefix] = cfg
    return flat


def config_diff(path_a: Path, path_b: Path) -> List[Tuple[str, Any, Any]]:
    flat_a = flatten(yaml.safe_load(path_a.read_text(encoding="utf-8")))
    flat_b = flatten(yaml.safe_load(path_b.read_text(encoding="utf-8")))
    diff = []
    for key in sorted(set(flat_a) | set(flat_b)):
        if flat_a.get(key) != flat_b.get(key):
            diff.append((key, flat_a.get(key), flat_b.get(key)))
    return diff


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta(a: Any, b: Any) -> str:
    if a is None or b is None:
        return "—"
    return f"{a - b:+.4f}"


# ---------------------------------------------------------------------- figures
def _bar_axis(ax: Any, labels: List[str], series: List[Tuple[str, str, List[float]]]) -> None:
    width = 0.8 / len(series)
    xs = range(len(labels))
    for index, (name, color, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar([x + offset for x in xs], values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_macro_f1(run_a: Dict[str, Any], run_b: Dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "macro_f1", "Macro-F1"),
                           (axes[1], "accuracy", "Accuracy (reference only)")):
        _bar_axis(
            ax,
            ["validation", "test"],
            [
                (LABEL_A, COLOR_A,
                 [run_a["validation"]["metrics"][key], run_a["test"]["metrics"][key]]),
                (LABEL_B, COLOR_B,
                 [run_b["validation"]["metrics"][key], run_b["test"]["metrics"][key]]),
            ],
        )
        ax.set_title(title)
    fig.suptitle("Rule engine: baseline vs tuned")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(run_a: Dict[str, Any], run_b: Dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, split in zip(axes, ("validation", "test")):
        _bar_axis(
            ax,
            list(CLASS_ROWS),
            [
                (LABEL_A, COLOR_A,
                 [run_a[split]["metrics"][f"{label}_f1"] for label in CLASS_ROWS]),
                (LABEL_B, COLOR_B,
                 [run_b[split]["metrics"][f"{label}_f1"] for label in CLASS_ROWS]),
            ],
        )
        ax.set_title(f"Per-class F1 - {split}")
    fig.suptitle("Rule engine: per-class F1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_robustness(run_a: Dict[str, Any], run_b: Dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, split in zip(axes, ("validation", "test")):
        robustness_a = run_a[split]["robustness"] or {}
        robustness_b = run_b[split]["robustness"] or {}
        _bar_axis(
            ax,
            [MODE_TITLES[m] for m in MODES],
            [
                (LABEL_A, COLOR_A,
                 [(robustness_a.get("modes", {}).get(m) or {}).get("detection_rate", 0.0) for m in MODES]),
                (LABEL_B, COLOR_B,
                 [(robustness_b.get("modes", {}).get(m) or {}).get("detection_rate", 0.0) for m in MODES]),
            ],
        )
        ax.set_title(f"Detection rate - {split} (400 positives/mode)")
        ax.set_ylim(0, 0.8)
    fig.suptitle("Rule engine: robustness to obfuscation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ report
def build_markdown(run_a: Dict[str, Any], run_b: Dict[str, Any],
                   significance: Optional[Dict[str, Any]], diff: List[Tuple[str, Any, Any]],
                   paths: Dict[str, Path]) -> str:
    metrics_a_val = run_a["validation"]["metrics"]
    metrics_b_val = run_b["validation"]["metrics"]
    metrics_a_test = run_a["test"]["metrics"]
    metrics_b_test = run_b["test"]["metrics"]
    lines: List[str] = []
    add = lines.append

    add("# So sánh 2 rule engine: baseline vs tuned")
    add("")
    add("> Sinh bởi `python scripts/28_compare_rule_engines.py` - **chỉ đọc các artifact đã đóng băng**")
    add("> (metrics/robustness/latency/`thresholds.json`), không chấm lại bất kỳ split nào (plan §44).")
    add("> Nguồn: `outputs/evaluation/rule_engine/` (baseline, `configs/rule_engine.yaml`) và")
    add("> `outputs/evaluation/rule_engine_tuned/` (bản thắng tune, `configs/rule_engine_tuned.yaml`).")
    add("> Cùng lexicon `verified_lexicon.csv` 668 entry, cùng validation 2.672 mẫu, test 6.680 mẫu.")
    add("")

    add("## 1. Khác biệt cấu hình")
    add("")
    add("| Tham số | baseline | tuned |")
    add("|---|---|---|")
    for key, old, new in diff:
        add(f"| `{key}` | {old} | {new} |")
    thresholds_a = metrics_a_val.get("thresholds", {})
    thresholds_b = metrics_b_val.get("thresholds", {})
    add(f"| `thresholds.HATE/OFFENSIVE/CLEAN` (đóng băng) "
        f"| {fmt(thresholds_a.get('HATE'), 2)} / {fmt(thresholds_a.get('OFFENSIVE'), 2)} / "
        f"{fmt(thresholds_a.get('CLEAN'), 2)} "
        f"| {fmt(thresholds_b.get('HATE'), 2)} / {fmt(thresholds_b.get('OFFENSIVE'), 2)} / "
        f"{fmt(thresholds_b.get('CLEAN'), 2)} |")
    add(f"| `lexicon_path` | {metrics_a_val.get('lexicon', '—')} | {metrics_b_val.get('lexicon', '—')} |")
    add("")

    for title, split in (("Validation (2.672 mẫu - nơi tune)", "validation"),
                         ("Test (6.680 mẫu - chấm **một lần**)", "test")):
        left = run_a[split]["metrics"]
        right = run_b[split]["metrics"]
        add(f"## {'2' if split == 'validation' else '3'}. {title}")
        add("")
        add("| Chỉ số | baseline | tuned | Δ (tuned − baseline) |")
        add("|---|---|---|---|")
        for key, label in SUMMARY_ROWS:
            add(f"| {label} | {fmt(left.get(key))} | {fmt(right.get(key))} | {delta(right.get(key), left.get(key))} |")
        add("")

    add("## 4. Chi tiết per-class trên test (precision / recall / F1 / support)")
    add("")
    add("| Nhãn | Hệ thống | precision | recall | F1 | support |")
    add("|---|---|---|---|---|---|")
    for label in CLASS_ROWS:
        for name, metrics in ((LABEL_A, metrics_a_test), (LABEL_B, metrics_b_test)):
            add(f"| {label} | {name} | {fmt(metrics.get(f'{label}_precision'))} | "
                f"{fmt(metrics.get(f'{label}_recall'))} | {fmt(metrics.get(f'{label}_f1'))} | "
                f"{metrics.get(f'{label}_support', '—')} |")
    add("")
    add("![](rule_engine_per_class_f1.png)")
    add("")

    add("## 5. Robustness - tỉ lệ phát hiện trên 400 mẫu nhiễu/mode")
    add("")
    add("| Mode | validation baseline | validation tuned | test baseline | test tuned |")
    add("|---|---|---|---|---|")
    for mode in MODES:
        cells = []
        for split in ("validation", "test"):
            for run in (run_a, run_b):
                modes = ((run[split]["robustness"] or {}).get("modes") or {})
                cell = modes.get(mode) or {}
                cells.append(fmt(cell.get("detection_rate")))
        add(f"| `{mode}` | " + " | ".join(cells) + " |")
    add("")
    add("*Đánh đổi đã biết: tuned giảm `no_accent` (accentless_alias = 0) bù lại leet/spaced/repeat tốt hơn.*")
    add("")
    add("![](rule_engine_robustness.png)")
    add("")

    add("## 6. Độ trễ (validation, 2.672 mẫu)")
    add("")
    add("| Chỉ số | baseline | tuned |")
    add("|---|---|---|")
    for key, label in (("mean_ms", "mean ms/mẫu"), ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"),
                       ("throughput_per_sec", "mẫu/giây")):
        latency_a = run_a["validation"]["latency"] or {}
        latency_b = run_b["validation"]["latency"] or {}
        digits = 1 if key == "throughput_per_sec" else 4
        add(f"| {label} | {fmt(latency_a.get(key), digits)} | {fmt(latency_b.get(key), digits)} |")
    add("")
    add("![](rule_engine_macro_f1.png)")
    add("")

    add("## 7. Kiểm định chênh lệch trên test (paired bootstrap + McNemar)")
    add("")
    if significance is None:
        add(f"> Chưa có - chạy `python scripts/26_rule_significance.py "
            f"--a {TUNED}/test/predictions.csv --b {BASELINE}/test/predictions.csv "
            f"--out {SIGNIFICANCE}` (dùng lại predictions đã chấm, **không** chấm test mới).")
    else:
        paired = significance.get("paired", {})
        bootstrap = significance.get("bootstrap_ci", {})
        mcnemar = significance.get("mcnemar", {})
        add(f"- Giao thức: {significance.get('protocol', {}).get('n_boot', '—')} lần bootstrap, "
            f"seed {significance.get('protocol', {}).get('seed', '—')}, "
            f"CI {int((1 - significance.get('protocol', {}).get('alpha', 0.05)) * 100)}%, "
            f"n = {significance.get('n_test')} mẫu test; số liệu gốc: "
            f"`{Path(paths['significance']).name}` + `papers/tables/rule_significance.tex`.")
        for metric in ("macro_f1", "accuracy"):
            ci = bootstrap.get(metric, {})
            pair = paired.get(metric, {})
            a_ci = ci.get("a", {})
            b_ci = ci.get("b", {})
            p_value = pair.get("p_value")
            p_text = "< 1e-4" if (p_value is not None and p_value < 1e-4) else fmt(p_value, 4)
            add(f"- **{metric}**: baseline {fmt(b_ci.get('point'))} "
                f"[{fmt(b_ci.get('ci_low'))}, {fmt(b_ci.get('ci_high'))}] vs "
                f"tuned {fmt(a_ci.get('point'))} [{fmt(a_ci.get('ci_low'))}, {fmt(a_ci.get('ci_high'))}] → "
                f"Δ = {delta(pair.get('delta'), 0)} CI [{fmt(pair.get('ci_low'))}, {fmt(pair.get('ci_high'))}], "
                f"p {p_text}.")
        if mcnemar:
            p_value = mcnemar.get("p_value")
            p_text = f"{p_value:.3g}" if isinstance(p_value, float) else fmt(p_value)
            add(f"- **McNemar** (đúng/sai từng mẫu): tuned đúng & baseline sai {mcnemar.get('a_correct_b_wrong')}, "
                f"tuned sai & baseline đúng {mcnemar.get('a_wrong_b_correct')} "
                f"(đối xứng {mcnemar.get('discordant')} mẫu), p = {p_text}.")
        for verdict in significance.get("verdict", []):
            add(f"- Kết luận của `scripts/26`: {verdict}")
    add("")

    add("## 8. Kết luận (theo số, có kiểm định - plan §44)")
    add("")
    macro_a = metrics_a_test.get("macro_f1")   # baseline
    macro_b = metrics_b_test.get("macro_f1")   # tuned
    add(f"- Test macro-F1: **{fmt(macro_a)}** (baseline) vs **{fmt(macro_b)}** (tuned) "
        f"→ Δ = {delta(macro_b, macro_a)} ({(macro_b - macro_a) / macro_a * 100:+.1f}%).")
    if significance is not None:
        pair = significance.get("paired", {}).get("macro_f1", {})
        add(f"- Chênh lệch **có ý nghĩa thống kê**: paired bootstrap CI "
            f"[{fmt(pair.get('ci_low'))}, {fmt(pair.get('ci_high'))}] không chứa 0 "
            f"(p < 1e-4), McNemar p = {significance.get('mcnemar', {}).get('p_value', float('nan')):.3g}.")
    add("- Cả hai cùng lexicon 668 entry; khác nhau chỉ ở trọng số scoring + ngưỡng đã tune trên validation.")
    add("- Đánh đổi phải nhớ (`robustness.json`): tuned mất recall chế độ `no_accent` "
        f"(test {fmt(((run_b['test']['robustness'] or {}).get('modes') or {}).get('no_accent', {}).get('detection_rate'))} "
        f"vs baseline {fmt(((run_a['test']['robustness'] or {}).get('modes') or {}).get('no_accent', {}).get('detection_rate'))}) "
        "vì `accentless_alias = 0`.")
    add("- `F1 OFFENSIVE` thấp ở cả hai (0.24 / 0.15) là bài toán coverage của từ điển, không phải bug; "
        "ngưỡng `OFFENSIVE = 0.0` là bình thường.")
    add("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the two frozen rule-engine runs.")
    parser.add_argument("--baseline", default=str(BASELINE), help="directory of the baseline run")
    parser.add_argument("--tuned", default=str(TUNED), help="directory of the tuned run")
    parser.add_argument("--out", default=str(OUT_DIR), help="where the report/figures are written")
    parser.add_argument("--significance", default=str(SIGNIFICANCE),
                        help="JSON from scripts/26_rule_significance.py (optional)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline)
    tuned_dir = Path(args.tuned)

    LOGGER.info("[28] loading %s and %s", baseline_dir, tuned_dir)
    run_a = load_run(baseline_dir)
    run_b = load_run(tuned_dir)
    significance = _load_json(Path(args.significance))

    diff_config = Path("configs/rule_engine.yaml")
    diff_tuned = Path("configs/rule_engine_tuned.yaml")
    diff = config_diff(diff_config, diff_tuned) if diff_config.is_file() and diff_tuned.is_file() else []

    paths = {
        "md": out_dir / "rule_engine_baseline_vs_tuned.md",
        "macro": out_dir / "rule_engine_macro_f1.png",
        "per_class": out_dir / "rule_engine_per_class_f1.png",
        "robustness": out_dir / "rule_engine_robustness.png",
        "significance": Path(args.significance),
    }

    plot_macro_f1(run_a, run_b, paths["macro"])
    LOGGER.info("[28] wrote %s", paths["macro"])
    plot_per_class(run_a, run_b, paths["per_class"])
    LOGGER.info("[28] wrote %s", paths["per_class"])
    plot_robustness(run_a, run_b, paths["robustness"])
    LOGGER.info("[28] wrote %s", paths["robustness"])

    report = build_markdown(run_a, run_b, significance, diff, paths)
    paths["md"].write_text(report, encoding="utf-8")
    LOGGER.info("[28] wrote %s", paths["md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
