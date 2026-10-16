"""Step 18 - smoke test for the lexical rule engine (plan section 35-39).

    python scripts/18_smoke_test_rule_engine.py
    python scripts/18_smoke_test_rule_engine.py --limit 100
    python scripts/18_smoke_test_rule_engine.py --split all --limit 0
    python scripts/18_smoke_test_rule_engine.py --out outputs_smoke/evaluation_smoke/rule_engine

Fast sanity checks that run in seconds and require **no GPU**. They verify the
rule engine actually holds together before you spend time on a real evaluation:

  * config loads and validates, lexicon loads, engine builds
  * normalization keeps Vietnamese diacritics + exact offset map (section 37)
  * a real verified term matches exactly, with correct raw span, no double count
  * obfuscation views + transformation classifier return the expected types (section 38)
  * empty / None / very-long text never crash
  * scores stay in [0, 1]
  * a miniature end-to-end pass over the validation split produces finite metrics

`--split` selects `validation` (default), `test`, `train` or `all`; `--limit 0`
runs every row instead of the default 60. Results are written to
`outputs_smoke/evaluation_smoke/rule_engine/` as `smoke_test.json` +
`smoke_test.md`; override with `--out DIR` or skip with `--no-write`. The
end-to-end pass runs entirely in memory.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.common.logging_utils import get_logger  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.rule_engine import RuleEngine, load_rule_engine_config  # noqa: E402
from src.rule_engine.lexical_classifier import Lexicon  # noqa: E402
from src.rule_engine.matcher import bounded_levenshtein  # noqa: E402
from src.rule_engine.normalize import normalize  # noqa: E402
from src.rule_engine.obfuscation import build_views, classify_transformation  # noqa: E402

LOGGER = get_logger("sensitiveai.smoke.rule_engine")

DEFAULT_OUT_DIR = "outputs_smoke/evaluation_smoke/rule_engine"

FAILURES: List[str] = []
CHECKS: List[Dict[str, Any]] = []
EVAL_RESULTS: Dict[str, Any] = {}
META: Dict[str, Any] = {}
OUT_DIR = DEFAULT_OUT_DIR
WRITE_OUTPUTS = True


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "OK  " if ok else "FAIL"
    LOGGER.info("[%s] %-46s %s", status, label, detail)
    CHECKS.append({"label": label, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILURES.append(label)


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _pick_verified_hate_term(lexicon: Lexicon) -> str | None:
    for entry in lexicon.by_labels("HATE"):
        if entry.review_status == "verified" and len(entry.term) >= 3:
            return entry.term
    return None


def main(argv: List[str] | None = None) -> int:
    _ensure_utf8_console()
    args = argv if argv is not None else sys.argv[1:]
    limit = 60
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    split_arg = "validation"
    if "--split" in args:
        split_arg = args[args.index("--split") + 1].lower()
    out_dir = DEFAULT_OUT_DIR
    if "--out" in args:
        out_dir = args[args.index("--out") + 1]
    write_outputs = "--no-write" not in args

    config_path = Path("configs/rule_engine.yaml")

    global OUT_DIR, WRITE_OUTPUTS
    OUT_DIR = out_dir
    WRITE_OUTPUTS = write_outputs
    META.update({"config_path": str(config_path), "split": split_arg, "limit": limit})

    # ----1. config / lexicon / engine ---------------------------------- #
    try:
        config = load_rule_engine_config(config_path)
        check("config loads + validates", True, str(config_path))
    except Exception as exc:  # pragma: no cover - failure path
        check("config loads + validates", False, str(exc))
        return _finish()

    lexicon = Lexicon.load(config.lexicon_path, config.scoring.min_review_statuses)
    check("lexicon loads", len(lexicon) > 0, f"{len(lexicon)} entries")
    META["lexicon_entries"] = len(lexicon)
    META["lexicon_path"] = str(config.lexicon_path)

    engine = RuleEngine(lexicon, config)
    check("engine builds", len(engine.matcher.fuzzy_index) >= 0, "matcher ready")

    # ----2. normalization: diacritics + offset map (section 37) --------- #
    raw = "Thằng Mất Dạy,  tao  ghét\n\n nó"
    nt = normalize(raw, config.normalize)
    check(
        "keeps Vietnamese diacritics",
        "ấ" in nt.normalized_text and "ạ" in nt.normalized_text,
        repr(nt.normalized_text),
    )
    check("lowercases", nt.normalized_text == nt.normalized_text.lower())
    check(
        "collapses whitespace runs",
        "  " not in nt.normalized_text and "\n" not in nt.normalized_text,
        repr(nt.normalized_text),
    )
    check(
        "offset map length == normalized length",
        len(nt.index_map) == len(nt.normalized_text),
        f"{len(nt.index_map)} vs {len(nt.normalized_text)}",
    )
    check(
        "offset map monotonic",
        all(a <= b for a, b in zip(nt.index_map, nt.index_map[1:])),
    )
    # raw_slice must reproduce the untouched raw text
    start, end = nt.index_map[0], nt.index_map[5] + 1
    check(
        "raw_slice reproduces raw text",
        nt.raw_text[start:end].lower().startswith("thằng"),
        repr(nt.raw_text[start:end]),
    )

    # ----3. exact match on a real verified term (section 39) ------------- #
    term = _pick_verified_hate_term(lexicon)
    if term is None:
        check("verified HATE term available", False, "none in lexicon")
    else:
        result = engine.analyze(term)
        hate_matches = [m for m in result["matches"] if m["label"] == "HATE"]
        check("exact match found", bool(hate_matches), f"term={term!r}")
        if hate_matches:
            match = hate_matches[0]
            check(
                "match_type is exact",
                match["match_type"] == "exact",
                match["match_type"],
            )
            check(
                "raw_span == input text",
                match["raw_span"] == term,
                repr(match["raw_span"]),
            )
            check(
                "scores in [0, 1]",
                all(
                    0.0 <= result[key] <= 1.0
                    for key in ("rule_score", "rule_hate_score", "rule_offensive_score")
                ),
                f"rule_score={result['rule_score']:.4f}",
            )
            spans = [(m["start"], m["end"], m["label"]) for m in result["matches"]]
            check("no double count for a span+label", len(spans) == len(set(spans)))

    # ----4. obfuscation views + transformation classifier (section 38) --- #
    views = build_views(normalize("thằng ngu", config.normalize), config.obfuscation)
    check(
        "base/plain/compact views built",
        {"base", "plain", "compact"} <= set(views),
        ",".join(sorted(views)),
    )
    obf_cases = [
        ("c0n ch0", "con chó", None, "leet"),
        ("con cho", "con chó", None, "accentless"),
        ("nguuu", "ngu", None, "collapse"),
        ("n g u", "ngu", True, "separator"),
    ]
    for span, dict_term, compact, expected in obf_cases:
        classified = classify_transformation(
            span, dict_term, config.obfuscation, compact=bool(compact)
        )
        ok = classified is not None and expected in classified[1]
        check(
            f"classify {span!r}->{dict_term!r}",
            ok,
            f"{expected}, got {classified}",
        )

    # ----5. bounded levenshtein ----------------------------------------- #
    check(
        "bounded_levenshtein distance",
        bounded_levenshtein("ngu", "nguu", 1) == 1
        and bounded_levenshtein("ngu", "ngu", 1) == 0
        and bounded_levenshtein("ngu", "abcdef", 1) > 1,
    )

    # ----6. robustness: empty / None / very long ------------------------- #
    for label, payload in [
        ("empty string", ""),
        ("None", None),
        ("very long text", "thằng ngu này " * 4000),
    ]:
        try:
            out = engine.analyze(payload)
            ok = out["rule_score"] is not None and 0.0 <= out["rule_score"] <= 1.0
            detail = f"n_matches={out['meta']['n_matches']}"
            if payload and len(str(payload)) > config.normalize.max_text_chars:
                detail += f" truncated={out['meta']['truncated']}"
            check(f"handles {label}", ok, detail)
        except Exception as exc:  # pragma: no cover - failure path
            check(f"handles {label}", False, repr(exc))

    # ----7. end-to-end pass over one or more splits ---------------------- #
    split_files = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }
    if split_arg == "all":
        chosen = list(split_files)
    elif split_arg in split_files:
        chosen = [split_arg]
    else:
        check(f"split argument {split_arg!r}", False, "expected train/validation/test/all")
        return _finish()

    import csv as _csv
    import time as _time

    all_true: List[int] = []
    all_pred: List[int] = []
    for name in chosen:
        path = Path("data/splits") / split_files[name]
        if not path.is_file():
            check(f"{name} split present", False, str(path))
            continue
        texts, y_true = [], []
        with path.open(encoding="utf-8", newline="") as fh:
            for row in _csv.DictReader(fh):
                texts.append(row["text"])
                y_true.append(int(row["label"]))
                if limit > 0 and len(texts) >= limit:
                    break
        proba = np.zeros((len(texts), 3))
        t0 = _time.perf_counter()
        for i, text in enumerate(texts):
            out = engine.analyze(text)
            proba[i, 0] = out["rule_hate_score"]
            proba[i, 1] = out["rule_offensive_score"]
            proba[i, 2] = max(0.0, 1.0 - max(proba[i, 0], proba[i, 1]))
        elapsed = _time.perf_counter() - t0
        y_pred = proba.argmax(axis=1)
        metrics = compute_metrics(y_true, y_pred, proba)
        finite = all(
            np.isfinite(float(metrics[k]))
            for k in ("accuracy", "macro_f1", "HATE_f1", "OFFENSIVE_f1", "CLEAN_f1")
        )
        ms = (elapsed / max(1, len(texts))) * 1000.0
        EVAL_RESULTS[name] = {
            "rows": len(texts),
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "hate_f1": float(metrics["HATE_f1"]),
            "offensive_f1": float(metrics["OFFENSIVE_f1"]),
            "clean_f1": float(metrics["CLEAN_f1"]),
            "ms_per_row": ms,
        }
        check(
            f"eval {name} ({len(texts)} rows)",
            finite,
            f"macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} "
            f"HATE_f1={metrics['HATE_f1']:.4f} OFF_f1={metrics['OFFENSIVE_f1']:.4f} "
            f"({ms:.2f} ms/row)",
        )
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    if len(chosen) > 1 and all_true:
        combined = compute_metrics(all_true, all_pred)
        EVAL_RESULTS["combined"] = {
            "rows": len(all_true),
            "macro_f1": float(combined["macro_f1"]),
            "accuracy": float(combined["accuracy"]),
        }
        check(
            f"eval combined ({len(all_true)} rows)",
            True,
            f"macro_f1={combined['macro_f1']:.4f} acc={combined['accuracy']:.4f}",
        )

    return _finish()


def _finish() -> int:
    LOGGER.info("=" * 72)
    passed = not FAILURES
    if passed:
        LOGGER.info("SMOKE PASSED: all rule-engine checks OK")
    else:
        LOGGER.error("SMOKE FAILED: %d check(s) failed: %s", len(FAILURES), FAILURES)
    if WRITE_OUTPUTS:
        _write_outputs(passed)
    return 0 if passed else 1


def _write_outputs(passed: bool) -> None:
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")

    summary: Dict[str, Any] = {
        "generated_at": generated_at,
        "passed": passed,
        "n_checks": len(CHECKS),
        "n_failed": len(FAILURES),
        "failures": FAILURES,
        "meta": META,
        "eval": EVAL_RESULTS,
        "checks": CHECKS,
    }
    json_path = out / "smoke_test.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Rule engine smoke test",
        "",
        f"- generated_at: {generated_at}",
        f"- result: {'PASSED' if passed else 'FAILED'}",
        f"- checks: {len(CHECKS)} ({len(FAILURES)} failed)",
        f"- config: {META.get('config_path', '')}",
        f"- lexicon: {META.get('lexicon_entries', '?')} entries ({META.get('lexicon_path', '')})",
        f"- split: {META.get('split', '')} (limit={META.get('limit', '')})",
        "",
        "## Checks",
        "",
        "| # | Check | Result | Detail |",
        "|---|-------|--------|--------|",
    ]
    for i, chk in enumerate(CHECKS, 1):
        lines.append(f"| {i} | {chk['label']} | {'OK' if chk['ok'] else 'FAIL'} | {chk['detail']} |")
    if EVAL_RESULTS:
        lines += [
            "",
            "## End-to-end (argmax, no threshold tuning)",
            "",
            "| Split | Rows | macro-F1 | accuracy | HATE-F1 | OFF-F1 | CLEAN-F1 | ms/row |",
            "|-------|------|----------|----------|---------|--------|----------|--------|",
        ]
        for name, res in EVAL_RESULTS.items():
            def _fmt(key: str) -> str:
                value = res.get(key)
                return f"{value:.4f}" if isinstance(value, (int, float)) else ""

            lines.append(
                f"| {name} | {res.get('rows', '')} | "
                f"{_fmt('macro_f1')} | {_fmt('accuracy')} | "
                f"{_fmt('hate_f1')} | {_fmt('offensive_f1')} | {_fmt('clean_f1')} | "
                f"{_fmt('ms_per_row')} |"
            )
    (out / "smoke_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    LOGGER.info("wrote %s and %s", json_path, out / "smoke_test.md")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
