"""Extract sensitive words from the ViHSD corpus into the project lexicon.

The cleaned ViHSD splits (``data/processed/vihsd/*.csv``) are scanned to
discover sensitive words that are NOT yet covered by the GitHub lexicons
built with ``scripts/12_build_lexicon.py``.

Method (high precision, document-frequency based)
-------------------------------------------------
1. Every row is normalized like the lexicon (NFC, lower-case, whitespace
   collapsed) and tokenized into syllables; punctuation-only / digit-only /
   emoji-only tokens are dropped, edge punctuation is stripped.
2. Document frequency is counted separately for the toxic rows
   (HATE + OFFENSIVE) and the clean rows (CLEAN).
3. A syllable (unigram) or adjacent syllable pair (bigram) becomes a
   candidate when ``toxic_df >= min_toxic_df`` AND ``clean_df <= max_clean_df``
   - i.e. it shows up repeatedly in toxic comments and (almost) never in
   clean ones.  Candidates already present in the combined lexicon are
   dropped.
4. The surviving candidates are appended to a per-source file
   ``vihsd_extracted_words.txt`` and merged into
   ``sensitive_words_combined.txt``; ``lexicon_report.json`` gains a
   ``vihsd_extraction`` section with parameters, counts and SHA-256.

The report records ``"estimated_on": "all ViHSD splits"`` - this lexicon is
an auxiliary filtering resource. It is NOT used by any of the four
classifiers in the ablation study, so scanning test text here does not
leak into the research results.

Usage
-----
    python -m src.data.vihsd_lexicon
    python -m src.data.vihsd_lexicon --min-toxic-df 10 --show-terms 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..common.constants import (
    LABEL2ID,
    LEXICON_PROCESSED_DIR,
    PROCESSED_DIR,
    VIHSD_PROCESSED_DIR,
)
from ..common.logging_utils import get_logger
from .lexicon import OUTPUT_COMBINED, OUTPUT_REPORT, _sha256, normalize_term, sources_dir

LOGGER = get_logger("sensitiveai.vihsd_lexicon")

VIHSD_OUTPUT = "vihsd_extracted_words.txt"
DEFAULT_SPLITS = ("train", "dev", "test")

_PUNCT_CATEGORIES = ("P", "S")


def strip_token(token: str) -> str:
    """Remove edge punctuation/symbols from a token, keep the core."""
    text = unicodedata.normalize("NFC", token)
    start, end = 0, len(text)
    while start < end and unicodedata.category(text[start])[0] in _PUNCT_CATEGORIES:
        start += 1
    while end > start and unicodedata.category(text[end - 1])[0] in _PUNCT_CATEGORIES:
        end -= 1
    return text[start:end]


def tokenize_row(text: str) -> List[str]:
    """Normalize one row and return its content-bearing syllable tokens."""
    normalized = normalize_term(text)
    if not normalized:
        return []
    tokens: List[str] = []
    for raw in normalized.split():
        token = strip_token(raw)
        if not token:
            continue
        if not any(ch.isalpha() for ch in token):
            continue  # pure punctuation / emoji / digits / symbols
        tokens.append(token)
    return tokens


def _load_rows(processed_dir: Path, splits: Sequence[str]) -> List[Tuple[str, int, str]]:
    """Return ``(split, label_id, text)`` for every row of the given splits."""
    import pandas as pd

    rows: List[Tuple[str, int, str]] = []
    for split in splits:
        path = processed_dir / f"{split}.csv"
        if not path.is_file():
            LOGGER.warning("[vihsd-lexicon] missing %s - skipped", path)
            continue
        df = pd.read_csv(path, dtype={"text": "string"})
        for label, text in zip(df["label"], df["text"]):
            rows.append((split, int(label), str(text) if pd.notna(text) else ""))
        LOGGER.info("[vihsd-lexicon] loaded %s rows from %s", len(df), path.name)
    return rows


def _existing_lexicon(output_dir: Path) -> Set[str]:
    """Base lexicon = the two GitHub sources (NOT the combined file, so a re-run
    rebuilds the combined list from scratch instead of accumulating stale terms)."""
    from .lexicon import OUTPUT_BEHITEK, OUTPUT_BLUEEYES

    word_dir = sources_dir(output_dir)
    terms: Set[str] = set()
    for name in (OUTPUT_BEHITEK, OUTPUT_BLUEEYES):
        path = word_dir / name
        if path.is_file():
            terms.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    combined = word_dir / OUTPUT_COMBINED
    if not terms and combined.is_file():
        terms.update(line.strip() for line in combined.read_text(encoding="utf-8").splitlines() if line.strip())
    return terms


def _lexicon_hit_pattern(lexicon: Set[str]) -> Optional[re.Pattern[str]]:
    if not lexicon:
        return None
    alternatives = sorted((re.escape(term) for term in lexicon), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)")


def extract(
    processed_dir: str | Path = VIHSD_PROCESSED_DIR,
    output_dir: str | Path = LEXICON_PROCESSED_DIR,
    splits: Sequence[str] = DEFAULT_SPLITS,
    min_toxic_df: int = 5,
    max_clean_df: int = 1,
) -> Dict[str, Any]:
    """Scan ViHSD and merge newly discovered sensitive words into the lexicon."""
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    rows = _load_rows(processed_dir, splits)
    if not rows:
        raise FileNotFoundError(f"No ViHSD csv found under {processed_dir}")

    toxic_id = LABEL2ID["HATE"]
    offensive_id = LABEL2ID["OFFENSIVE"]
    clean_id = LABEL2ID["CLEAN"]

    toxic_df: Counter = Counter()
    clean_df: Counter = Counter()
    rows_by_label: Counter = Counter()
    rows_with_lexicon_hit: Counter = Counter()
    existing = _existing_lexicon(output_dir)
    pattern = _lexicon_hit_pattern(existing)

    for _split, label, text in rows:
        rows_by_label[label] += 1
        tokens = tokenize_row(text)
        if not tokens:
            continue
        if pattern is not None and pattern.search(normalize_term(text)):
            rows_with_lexicon_hit[label] += 1
        unique_unigrams = set(tokens)
        unique_bigrams = {f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if len(a) + len(b) >= 3}
        if label in (toxic_id, offensive_id):
            toxic_df.update(unique_unigrams)
            toxic_df.update(unique_bigrams)
        elif label == clean_id:
            clean_df.update(unique_unigrams)
            clean_df.update(unique_bigrams)

    candidates_uni = set()
    candidates_bi = set()
    for term, df in toxic_df.items():
        if df < min_toxic_df or clean_df.get(term, 0) > max_clean_df:
            continue
        if term in existing or len(term) < 2:
            continue
        (candidates_bi if " " in term else candidates_uni).add(term)

    new_terms = candidates_uni | candidates_bi
    word_dir = sources_dir(output_dir)
    out_path = word_dir / VIHSD_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for term in sorted(new_terms):
            fh.write(term + "\n")
    LOGGER.info("[vihsd-lexicon] wrote %s (+%d new terms)", out_path, len(new_terms))

    combined = existing | new_terms
    combined_path = word_dir / OUTPUT_COMBINED
    with combined_path.open("w", encoding="utf-8", newline="\n") as fh:
        for term in sorted(combined):
            fh.write(term + "\n")
    LOGGER.info("[vihsd-lexicon] rebuilt %s (%d terms total)", combined_path, len(combined))

    report = _update_report(
        output_dir=output_dir,
        rows_total=len(rows),
        rows_by_label=rows_by_label,
        rows_with_lexicon_hit=rows_with_lexicon_hit,
        candidates_uni=candidates_uni,
        candidates_bi=candidates_bi,
        combined_total=len(combined),
        splits=splits,
        min_toxic_df=min_toxic_df,
        max_clean_df=max_clean_df,
    )
    return report


def _update_report(
    output_dir: Path,
    rows_total: int,
    rows_by_label: Counter,
    rows_with_lexicon_hit: Counter,
    candidates_uni: Set[str],
    candidates_bi: Set[str],
    combined_total: int,
    splits: Sequence[str],
    min_toxic_df: int,
    max_clean_df: int,
) -> Dict[str, Any]:
    report_path = output_dir / OUTPUT_REPORT
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"generated_by": "python scripts/12_build_lexicon.py", "outputs": {}, "counts": {}}

    label_names = {LABEL2ID["HATE"]: "HATE", LABEL2ID["OFFENSIVE"]: "OFFENSIVE", LABEL2ID["CLEAN"]: "CLEAN"}
    outputs = report.setdefault("outputs", {})
    outputs[VIHSD_OUTPUT] = {
        "terms": len(candidates_uni) + len(candidates_bi),
        "unigram_terms": len(candidates_uni),
        "bigram_terms": len(candidates_bi),
        "multiword_terms": len(candidates_bi),
        "sha256": _sha256(sources_dir(output_dir) / VIHSD_OUTPUT),
    }
    outputs[OUTPUT_COMBINED] = {
        "terms": combined_total,
        "sha256": _sha256(sources_dir(output_dir) / OUTPUT_COMBINED),
    }
    counts = report.setdefault("counts", {})
    counts["vihsd_extracted"] = len(candidates_uni) + len(candidates_bi)
    counts["combined"] = combined_total

    report["vihsd_extraction"] = {
        "estimated_on": "all ViHSD splits (auxiliary lexicon - NOT used by the four classifiers)",
        "source_dir": str(PROCESSED_DIR / "vihsd"),
        "splits_scanned": list(splits),
        "rows_scanned": rows_total,
        "rows_by_label": {label_names.get(k, str(k)): v for k, v in sorted(rows_by_label.items())},
        "rows_with_existing_lexicon_hit": {
            label_names.get(k, str(k)): v for k, v in sorted(rows_with_lexicon_hit.items())
        },
        "parameters": {"min_toxic_df": min_toxic_df, "max_clean_df": max_clean_df},
        "candidates": {
            "unigrams": len(candidates_uni),
            "bigrams": len(candidates_bi),
            "total": len(candidates_uni) + len(candidates_bi),
        },
    }

    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    LOGGER.info("[vihsd-lexicon] updated %s", report_path)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.vihsd_lexicon",
        description="Extract sensitive words from ViHSD into the project lexicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed-dir", default=None, help="Override the cleaned-ViHSD directory.")
    p.add_argument("--output-dir", default=None, help="Override the lexicon output directory.")
    p.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS), help="Which splits to scan.")
    p.add_argument("--min-toxic-df", type=int, default=5, help="Minimum toxic document frequency.")
    p.add_argument("--max-clean-df", type=int, default=1, help="Maximum clean document frequency.")
    p.add_argument("--show-terms", type=int, default=0, help="Print N extracted terms (0 = none).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = extract(
        processed_dir=Path(args.processed_dir) if args.processed_dir else VIHSD_PROCESSED_DIR,
        output_dir=Path(args.output_dir) if args.output_dir else LEXICON_PROCESSED_DIR,
        splits=args.splits,
        min_toxic_df=args.min_toxic_df,
        max_clean_df=args.max_clean_df,
    )
    section = report["vihsd_extraction"]
    LOGGER.info(
        "[vihsd-lexicon] scanned %d rows (%s); new terms: %d unigrams + %d bigrams; combined lexicon now %d",
        section["rows_scanned"],
        section["parameters"],
        section["candidates"]["unigrams"],
        section["candidates"]["bigrams"],
        report["counts"]["combined"],
    )
    if args.show_terms > 0:
        path = sources_dir(Path(args.output_dir) if args.output_dir else LEXICON_PROCESSED_DIR) / VIHSD_OUTPUT
        for term in path.read_text(encoding="utf-8").splitlines()[: args.show_terms]:
            LOGGER.info("[vihsd-lexicon]   sample: %s", term)
    LOGGER.info("[vihsd-lexicon] done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
