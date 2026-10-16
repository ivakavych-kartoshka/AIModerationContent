"""Extract sensitive words from multiple corpora into the project lexicon.

The cleaned ViHSD splits and the raw additional datasets (ViCM, bontori) are
scanned to discover sensitive words that are NOT yet covered by the GitHub
lexicons built with ``scripts/12_build_lexicon.py``.

Method (high precision, document-frequency based)
-------------------------------------------------
1. Each row is normalized like the lexicon (NFC, lower-case, whitespace
   collapsed) and tokenized into syllables; punctuation-only / digit-only /
   emoji-only tokens are dropped, edge punctuation is stripped.
2. Document frequency is counted separately for the toxic rows
   (HATE + OFFENSIVE) and the clean rows (CLEAN).
3. A syllable (unigram) or adjacent syllable pair (bigram) becomes a
   candidate when ``toxic_df >= min_toxic_df`` AND ``clean_df <= max_clean_df``.
   Candidates already present in the combined lexicon are dropped.
4. Per-source files are written (e.g. ``vihsd_extracted_words.txt``,
   ``vicm_extracted_words.txt``, ``bontori_extracted_words.txt``) and merged into
   ``sensitive_words_combined.txt``; ``lexicon_report.json`` is updated with
   a ``corpus_extraction`` section.

Usage
-----
    python -m src.data.corpus_lexicon
    python -m src.data.corpus_lexicon --min-toxic-df 5 --max-clean-df 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from ..common.constants import (
    LABEL2ID,
    LEXICON_PROCESSED_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from ..common.logging_utils import get_logger
from .lexicon import OUTPUT_COMBINED, OUTPUT_REPORT, _sha256, normalize_term

LOGGER = get_logger("sensitiveai.corpus_lexicon")

VIHSD_OUTPUT = "vihsd_extracted_words.txt"
VICM_OUTPUT = "vicm_extracted_words.txt"
BONTORI_OUTPUT = "bontori_extracted_words.txt"

DEFAULT_SPLITS_VIHSD = ("train", "dev", "test")
DEFAULT_FILES_VICM = ("train.xlsx", "dev.xlsx", "test.xlsx", "forum_data.csv")
DEFAULT_FILES_BONTORI = ("tdtu_train.parquet", "tdtu_dev.parquet", "tdtu_test.parquet")

_PUNCT_CATEGORIES = ("P", "S")


def strip_token(token: str) -> str:
    text = unicodedata.normalize("NFC", token)
    start, end = 0, len(text)
    while start < end and unicodedata.category(text[start])[0] in _PUNCT_CATEGORIES:
        start += 1
    while end > start and unicodedata.category(text[end - 1])[0] in _PUNCT_CATEGORIES:
        end -= 1
    return text[start:end]


def tokenize_row(text: str) -> List[str]:
    normalized = normalize_term(text)
    if not normalized:
        return []
    tokens: List[str] = []
    for raw in normalized.split():
        token = strip_token(raw)
        if not token:
            continue
        if not any(ch.isalpha() for ch in token):
            continue
        tokens.append(token)
    return tokens


def _existing_lexicon(output_dir: Path) -> Set[str]:
    from .lexicon import OUTPUT_BEHITEK, OUTPUT_BLUEEYES

    terms: Set[str] = set()
    for name in (OUTPUT_BEHITEK, OUTPUT_BLUEEYES):
        path = output_dir / name
        if path.is_file():
            terms.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    combined = output_dir / OUTPUT_COMBINED
    if not terms and combined.is_file():
        terms.update(line.strip() for line in combined.read_text(encoding="utf-8").splitlines() if line.strip())
    return terms


def _lexicon_hit_pattern(lexicon: Set[str]) -> Optional[re.Pattern[str]]:
    if not lexicon:
        return None
    alternatives = sorted((re.escape(term) for term in lexicon), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)")


def _load_rows_vihsd(processed_dir: Path, splits: Sequence[str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for split in splits:
        path = processed_dir / f"{split}.csv"
        if not path.is_file():
            LOGGER.warning("[corpus-lexicon] missing %s - skipped", path)
            continue
        df = pd.read_csv(path, dtype={"text": "string"})
        for label_id, text in zip(df["label"], df["text"]):
            lid = int(label_id)
            if lid == LABEL2ID["HATE"]:
                lab = "HATE"
            elif lid == LABEL2ID["OFFENSIVE"]:
                lab = "OFFENSIVE"
            elif lid == LABEL2ID["CLEAN"]:
                lab = "CLEAN"
            else:
                lab = str(lid)
            rows.append((str(text) if pd.notna(text) else "", lab))
        LOGGER.info("[corpus-lexicon] ViHSD loaded %d rows from %s", len(df), path.name)
    return rows





def _map_vicm_label(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    m = {"0": "CLEAN", "clean": "CLEAN", "Clean": "CLEAN", "1": "HATE", "hate": "HATE", "Hate": "HATE", "2": "OFFENSIVE", "offensive": "OFFENSIVE", "Offensive": "OFFENSIVE"}
    return m.get(s, m.get(s.lower()))


def _load_rows_vicm(base: Path, files: Sequence[str]) -> List[Tuple[str, str]]:
    from . import additional

    rows: List[Tuple[str, str]] = []
    for fname in files:
        p = base / fname
        if not p.is_file():
            LOGGER.warning("[corpus-lexicon] missing %s - skipped", p)
            continue
        if fname.endswith(".xlsx"):
            data = additional._xlsx_rows(p)
            c = 0
            for rec in data:
                txt = rec.get("text") or rec.get("Text") or rec.get("B") or ""
                lab = _map_vicm_label(rec.get("Label") or rec.get("label") or rec.get("C"))
                if lab:
                    rows.append((txt, lab))
                    c += 1
            LOGGER.info("[corpus-lexicon] ViCM loaded %d rows from %s", c, fname)
        elif fname.endswith(".csv"):
            try:
                df = pd.read_csv(p, dtype=str)
            except Exception:
                continue
            c = 0
            for _, r in df.iterrows():
                txt = str(r.get("text") or r.get("Text") or "")
                labv = r.get("Real_Label") or r.get("real_label") or r.get("label") or r.get("Label")
                if pd.isna(labv):
                    labv = ""
                lab = _map_vicm_label(str(labv).strip())
                if not lab:
                    continue
                rows.append((txt, lab))
                c += 1
            LOGGER.info("[corpus-lexicon] ViCM loaded %d rows from %s", c, fname)
    return rows


def _map_bontori_label(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    m = {"CLEAN": "CLEAN", "clean": "CLEAN", "OFFENSIVE": "OFFENSIVE", "offensive": "OFFENSIVE", "HATE": "HATE", "hate": "HATE"}
    return m.get(s, m.get(s.upper()))


def _load_rows_bontori(base: Path, files: Sequence[str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for fname in files:
        p = base / fname
        if not p.is_file():
            LOGGER.warning("[corpus-lexicon] missing %s - skipped", p)
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        c = 0
        for txt, labv in zip(df.get("text", []), df.get("label", [])):
            lab = _map_bontori_label(labv)
            if not lab:
                continue
            rows.append((str(txt) if pd.notna(txt) else "", lab))
            c += 1
        LOGGER.info("[corpus-lexicon] bontori loaded %d rows from %s", c, fname)
    return rows


def _extract_terms_from_rows(
    rows: Iterable[Tuple[str, str]],
    existing: Set[str],
    pattern: Optional[re.Pattern[str]],
    min_toxic_df: int,
    max_clean_df: int,
) -> Tuple[Set[str], Counter, Counter, Counter, Counter]:
    toxic_df: Counter = Counter()
    clean_df: Counter = Counter()
    rows_by_label: Counter = Counter()
    rows_with_lexicon_hit: Counter = Counter()
    for text, lab in rows:
        key = lab.upper()
        if key == "HATE":
            lidk = "HATE"
        elif key == "OFFENSIVE":
            lidk = "OFFENSIVE"
        elif key == "CLEAN":
            lidk = "CLEAN"
        else:
            continue
        rows_by_label[lidk] += 1
        tnorm = normalize_term(text)
        tokens = tokenize_row(text)
        if not tokens:
            continue
        if pattern is not None and pattern.search(tnorm):
            rows_with_lexicon_hit[lidk] += 1
        unique_unigrams = set(tokens)
        unique_bigrams = {f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if len(a) + len(b) >= 3}
        if lidk in ("HATE", "OFFENSIVE"):
            toxic_df.update(unique_unigrams)
            toxic_df.update(unique_bigrams)
        elif lidk == "CLEAN":
            clean_df.update(unique_unigrams)
            clean_df.update(unique_bigrams)
    candidates_uni: Set[str] = set()
    candidates_bi: Set[str] = set()
    for term, dfv in toxic_df.items():
        if dfv < min_toxic_df or clean_df.get(term, 0) > max_clean_df:
            continue
        if term in existing or len(term) < 2:
            continue
        if " " in term:
            candidates_bi.add(term)
        else:
            candidates_uni.add(term)
    return candidates_uni | candidates_bi, rows_by_label, rows_with_lexicon_hit, candidates_uni, candidates_bi


def _write_terms(path: Path, terms: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for term in sorted(terms):
            fh.write(term + "\n")


def extract_all(
    output_dir: str | Path = LEXICON_PROCESSED_DIR,
    min_toxic_df: int = 5,
    max_clean_df: int = 1,
    include_bontori_vihsd: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    existing = _existing_lexicon(output_dir)
    pattern = _lexicon_hit_pattern(existing)

    all_new: Set[str] = set()
    corpus_section: Dict[str, Any] = {}

    # ViHSD
    vihsd_dir = PROCESSED_DIR / "vihsd"
    vihsd_rows = _load_rows_vihsd(vihsd_dir, DEFAULT_SPLITS_VIHSD)
    vihsd_terms, vihsd_rb, vihsd_rh, vihsd_cu, vihsd_cb = _extract_terms_from_rows(vihsd_rows, existing, pattern, min_toxic_df, max_clean_df)
    if vihsd_terms:
        _write_terms(output_dir / VIHSD_OUTPUT, vihsd_terms)
    corpus_section["vihsd"] = {
        "output": VIHSD_OUTPUT,
        "rows_scanned": len(vihsd_rows),
        "rows_by_label": dict(sorted(vihsd_rb.items())),
        "rows_with_existing_lexicon_hit": dict(sorted(vihsd_rh.items())),
        "candidates": {"unigrams": len(vihsd_cu), "bigrams": len(vihsd_cb), "total": len(vihsd_terms)},
        "sha256": _sha256(output_dir / VIHSD_OUTPUT) if (output_dir / VIHSD_OUTPUT).is_file() else None,
    }
    all_new.update(vihsd_terms)

    # ViCM
    vicm_base = RAW_DIR / "additional" / "vicm-code-mixed"
    vicm_rows = _load_rows_vicm(vicm_base, DEFAULT_FILES_VICM)
    vicm_terms, vicm_rb, vicm_rh, vicm_cu, vicm_cb = _extract_terms_from_rows(vicm_rows, existing | all_new, pattern, min_toxic_df, max_clean_df)
    if vicm_terms:
        _write_terms(output_dir / VICM_OUTPUT, vicm_terms)
    corpus_section["vicm"] = {
        "output": VICM_OUTPUT,
        "rows_scanned": len(vicm_rows),
        "rows_by_label": dict(sorted(vicm_rb.items())),
        "rows_with_existing_lexicon_hit": dict(sorted(vicm_rh.items())),
        "candidates": {"unigrams": len(vicm_cu), "bigrams": len(vicm_cb), "total": len(vicm_terms)},
        "sha256": _sha256(output_dir / VICM_OUTPUT) if (output_dir / VICM_OUTPUT).is_file() else None,
    }
    all_new.update(vicm_terms)

    # bontori
    bontori_base = RAW_DIR / "additional" / "bontori-vietnamese-hsd-cleaned"
    bontori_rows_all = _load_rows_bontori(bontori_base, DEFAULT_FILES_BONTORI)
    if not include_bontori_vihsd:
        bontori_rows = [(t, l) for t, l in bontori_rows_all]
    else:
        bontori_rows = list(bontori_rows_all)
    bontori_terms, bontori_rb, bontori_rh, bontori_cu, bontori_cb = _extract_terms_from_rows(bontori_rows, existing | all_new, pattern, min_toxic_df, max_clean_df)
    if bontori_terms:
        _write_terms(output_dir / BONTORI_OUTPUT, bontori_terms)
    corpus_section["bontori"] = {
        "output": BONTORI_OUTPUT,
        "rows_scanned": len(bontori_rows),
        "rows_by_label": dict(sorted(bontori_rb.items())),
        "rows_with_existing_lexicon_hit": dict(sorted(bontori_rh.items())),
        "candidates": {"unigrams": len(bontori_cu), "bigrams": len(bontori_cb), "total": len(bontori_terms)},
        "sha256": _sha256(output_dir / BONTORI_OUTPUT) if (output_dir / BONTORI_OUTPUT).is_file() else None,
    }
    all_new.update(bontori_terms)

    combined = existing | all_new
    combined_path = output_dir / OUTPUT_COMBINED
    with combined_path.open("w", encoding="utf-8", newline="\n") as fh:
        for term in sorted(combined):
            fh.write(term + "\n")
    LOGGER.info("[corpus-lexicon] rebuilt %s (%d terms total)", combined_path, len(combined))

    report_path = output_dir / OUTPUT_REPORT
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"generated_by": "python scripts/12_build_lexicon.py", "outputs": {}, "counts": {}}
    outputs = report.setdefault("outputs", {})
    for k, sec in corpus_section.items():
        if sec["sha256"] and sec["output"]:
            outputs[sec["output"]] = {
                "terms": sec["candidates"]["total"],
                "unigram_terms": sec["candidates"]["unigrams"],
                "bigram_terms": sec["candidates"]["bigrams"],
                "multiword_terms": sec["candidates"]["bigrams"],
                "sha256": sec["sha256"],
            }
    outputs[OUTPUT_COMBINED] = {
        "terms": len(combined),
        "sha256": _sha256(combined_path),
    }
    counts = report.setdefault("counts", {})
    counts["corpus_extracted_total"] = len(all_new)
    counts["combined"] = len(combined)
    report["corpus_extraction"] = {
        "parameters": {"min_toxic_df": min_toxic_df, "max_clean_df": max_clean_df},
        "sources": corpus_section,
    }
    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    LOGGER.info("[corpus-lexicon] updated %s", report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.corpus_lexicon",
        description="Extract sensitive words from ViHSD, ViCM and bontori into the project lexicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default=None, help="Override the lexicon output directory.")
    p.add_argument("--min-toxic-df", type=int, default=5, help="Minimum toxic document frequency.")
    p.add_argument("--max-clean-df", type=int, default=1, help="Maximum clean document frequency.")
    p.add_argument("--show-terms", type=int, default=0, help="Print N sample new terms (0 = none).")
    p.add_argument("--exclude-bontori-vihsd", action="store_true", help="(not used) placeholder")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = extract_all(
        output_dir=Path(args.output_dir) if args.output_dir else LEXICON_PROCESSED_DIR,
        min_toxic_df=args.min_toxic_df,
        max_clean_df=args.max_clean_df,
    )
    LOGGER.info("[corpus-lexicon] extracted total new terms: %d; combined now %d", report["counts"].get("corpus_extracted_total", 0), report["counts"]["combined"])
    if args.show_terms > 0:
        outd = Path(args.output_dir or LEXICON_PROCESSED_DIR)
        for name in (VIHSD_OUTPUT, VICM_OUTPUT, BONTORI_OUTPUT):
            p = outd / name
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines()[: args.show_terms]:
                    LOGGER.info("[corpus-lexicon]   %s: %s", name, line)
                break
    LOGGER.info("[corpus-lexicon] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
