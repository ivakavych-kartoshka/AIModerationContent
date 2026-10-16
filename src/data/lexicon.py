"""Download and extract Vietnamese sensitive-word lexicons.

Two public lexicon datasets are fetched once into ``data/raw/lexicon/``
(kept byte-for-byte identical to upstream) and then parsed into normalized
word lists under ``data/processed/lexicon/``:

1. ``behitek/vietnam-sensitive-words``
   https://github.com/behitek/vietnam-sensitive-words
   ``bad_words.json`` (profanity + teencode), ``bad_words_22082018.json``
   (profanity + politically sensitive terms), ``hide.txt`` (core list).

2. ``blue-eyes-vn/vietnamese-offensive-words``
   https://github.com/blue-eyes-vn/vietnamese-offensive-words
   ``vn_offensive_words.txt`` - offensive words grouped by comment headers,
   each group containing accent-less, short-form and teencode variants.

Every extracted term is normalized identically: NFC, lower-cased,
underscores turned into spaces, whitespace collapsed, edges stripped.
Outputs are sorted, de-duplicated, one term per line (UTF-8), plus
``lexicon_report.json`` with counts, source overlap and SHA-256 of every
output file.

Usage
-----
    python -m src.data.lexicon                       # everything
    python -m src.data.lexicon --stage download       # fetch raw files only
    python -m src.data.lexicon --stage extract summary
    python -m src.data.lexicon --force-download       # re-fetch raw files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..common.constants import LEXICON_PROCESSED_DIR, LEXICON_RAW_DIR
from ..common.logging_utils import get_logger

LOGGER = get_logger("sensitiveai.lexicon")

STAGES = ("download", "extract", "summary")

#: upstream repo -> {file name: raw download URL}
LEXICON_SOURCES: Dict[str, Dict[str, str]] = {
    "behitek-vietnam-sensitive-words": {
        "repo": "https://github.com/behitek/vietnam-sensitive-words",
        "bad_words.json": (
            "https://raw.githubusercontent.com/behitek/vietnam-sensitive-words/master/bad_words.json"
        ),
        "bad_words_22082018.json": (
            "https://raw.githubusercontent.com/behitek/vietnam-sensitive-words/master/bad_words_22082018.json"
        ),
        "hide.txt": (
            "https://raw.githubusercontent.com/behitek/vietnam-sensitive-words/master/hide.txt"
        ),
        "README.md": (
            "https://raw.githubusercontent.com/behitek/vietnam-sensitive-words/master/README.md"
        ),
    },
    "blue-eyes-vn-vietnamese-offensive-words": {
        "repo": "https://github.com/blue-eyes-vn/vietnamese-offensive-words",
        "vn_offensive_words.txt": (
            "https://raw.githubusercontent.com/blue-eyes-vn/vietnamese-offensive-words/main/vn_offensive_words.txt"
        ),
        "vn_offensive_words_other_formats.txt": (
            "https://raw.githubusercontent.com/blue-eyes-vn/vietnamese-offensive-words/main/"
            "vn_offensive_words_other_formats.txt"
        ),
        "README.md": (
            "https://raw.githubusercontent.com/blue-eyes-vn/vietnamese-offensive-words/main/README.md"
        ),
        "README_vn.md": (
            "https://raw.githubusercontent.com/blue-eyes-vn/vietnamese-offensive-words/main/README_vn.md"
        ),
        "LICENSE.md": (
            "https://raw.githubusercontent.com/blue-eyes-vn/vietnamese-offensive-words/main/LICENSE.md"
        ),
    },
}

OUTPUT_BEHITEK = "behitek_sensitive_words.txt"
OUTPUT_BLUEEYES = "blueeyes_offensive_words.txt"
OUTPUT_COMBINED = "sensitive_words_combined.txt"
OUTPUT_REPORT = "lexicon_report.json"

#: Order applied by :func:`normalize_term` (recorded in the report).
NORMALIZATION_STEPS = ("NFC", "lowercase", "underscore_to_space", "collapse_whitespace", "strip_edges")


def normalize_term(term: object) -> str:
    """Normalize one raw term. Returns ``""`` when nothing visible remains."""
    if term is None:
        return ""
    text = unicodedata.normalize("NFC", str(term)).lower()
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, target: Path, force: bool = False) -> Path:
    if target.is_file() and not force:
        LOGGER.info("[lexicon] %s already present -> %s", target.name, target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[lexicon] fetching %s ...", url)
    request = urllib.request.Request(url, headers={"User-Agent": "sensitiveai-lexicon/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URLs
        payload = response.read()
    target.write_bytes(payload)
    LOGGER.info("[lexicon] saved %s (%d bytes)", target, len(payload))
    return target


def download_lexicons(
    raw_dir: str | Path = LEXICON_RAW_DIR,
    force: bool = False,
) -> Dict[str, Path]:
    """Fetch every raw lexicon file into ``raw_dir/<source>/``. Returns paths."""
    out = Path(raw_dir)
    paths: Dict[str, Path] = {}
    for source, spec in LEXICON_SOURCES.items():
        for name, url in spec.items():
            if name == "repo":
                continue
            paths[f"{source}/{name}"] = _download_file(url, out / source / name, force=force)
    write_raw_readmes(out)
    return paths


def write_raw_readmes(raw_dir: str | Path = LEXICON_RAW_DIR) -> Path:
    out = Path(raw_dir)
    for source, spec in LEXICON_SOURCES.items():
        readme = out / source / "README-source.md"
        if not readme.is_file():
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                f"# {source} (raw)\n\n"
                f"Auto-downloaded from {spec['repo']}\n\n"
                "Files are kept exactly as published upstream. "
                "Run `python scripts/12_build_lexicon.py` to (re)build the "
                "normalized word lists in `data/processed/lexicon/`.\n",
                encoding="utf-8",
            )
    return out


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_terms(path: Path) -> Set[str]:
    """One term per line; ``#`` comment / separator lines are skipped."""
    terms: Set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            term = normalize_term(stripped)
            if term:
                terms.add(term)
    return terms


def parse_behitek(raw_dir: str | Path = LEXICON_RAW_DIR) -> Tuple[Set[str], Dict[str, Any]]:
    """Parse the behitek sources. Returns ``(terms, stats)``."""
    src = Path(raw_dir) / "behitek-vietnam-sensitive-words"
    terms: Set[str] = set()
    stats: Dict[str, Any] = {"files": {}}

    for name in ("bad_words.json", "bad_words_22082018.json"):
        path = src / name
        if not path.is_file():
            LOGGER.warning("[lexicon] missing %s - skipped", path)
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            LOGGER.warning("[lexicon] %s is not a JSON object - skipped", path)
            continue
        file_terms = {normalize_term(key) for key in data}
        file_terms.discard("")
        stats["files"][name] = {
            "raw_entries": len(data),
            "unique_terms": len(file_terms),
            "sha256": _sha256(path),
        }
        if name == "bad_words_22082018.json":
            by_label: Dict[str, int] = {}
            for value in data.values():
                label = str(value.get("label", "unknown")) if isinstance(value, dict) else "unknown"
                by_label[label] = by_label.get(label, 0) + 1
            stats["files"][name]["entries_by_label"] = dict(sorted(by_label.items()))
        terms.update(file_terms)

    hide = src / "hide.txt"
    if hide.is_file():
        hide_terms = _load_terms(hide)
        stats["files"]["hide.txt"] = {
            "raw_lines": len(hide.read_text(encoding="utf-8").splitlines()),
            "unique_terms": len(hide_terms),
            "sha256": _sha256(hide),
        }
        terms.update(hide_terms)

    stats["unique_terms"] = len(terms)
    return terms, stats


def parse_blueeyes(raw_dir: str | Path = LEXICON_RAW_DIR) -> Tuple[Set[str], Dict[str, Any]]:
    """Parse the blue-eyes-vn offensive word list. Returns ``(terms, stats)``."""
    path = Path(raw_dir) / "blue-eyes-vn-vietnamese-offensive-words" / "vn_offensive_words.txt"
    if not path.is_file():
        LOGGER.warning("[lexicon] missing %s - no terms extracted", path)
        return set(), {"files": {}, "unique_terms": 0}

    terms = _load_terms(path)
    with path.open(encoding="utf-8") as fh:
        groups = sum(1 for line in fh if line.startswith("# ") and "and its variants" in line)

    stats = {
        "files": {
            "vn_offensive_words.txt": {
                "raw_lines": len(path.read_text(encoding="utf-8").splitlines()),
                "group_headers": groups,
                "unique_terms": len(terms),
                "sha256": _sha256(path),
            }
        },
        "unique_terms": len(terms),
    }
    return terms, stats


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _write_word_list(path: Path, terms: Set[str]) -> Dict[str, Any]:
    ordered = sorted(terms)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for term in ordered:
            fh.write(term + "\n")
    return {
        "terms": len(ordered),
        "multiword_terms": sum(1 for term in ordered if " " in term),
        "sha256": _sha256(path),
    }


def build_lexicons(
    raw_dir: str | Path = LEXICON_RAW_DIR,
    output_dir: str | Path = LEXICON_PROCESSED_DIR,
) -> Dict[str, Any]:
    """Parse raw lexicons and write normalized word lists + JSON report."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)

    behitek_terms, behitek_stats = parse_behitek(raw_dir)
    blueeyes_terms, blueeyes_stats = parse_blueeyes(raw_dir)
    combined = behitek_terms | blueeyes_terms
    overlap = behitek_terms & blueeyes_terms

    outputs: Dict[str, Any] = {}
    for name, terms in (
        (OUTPUT_BEHITEK, behitek_terms),
        (OUTPUT_BLUEEYES, blueeyes_terms),
        (OUTPUT_COMBINED, combined),
    ):
        outputs[name] = _write_word_list(output_dir / name, terms)
        LOGGER.info("[lexicon] wrote %s (%d terms)", name, outputs[name]["terms"])

    report: Dict[str, Any] = {
        "generated_by": "python scripts/12_build_lexicon.py",
        "normalization": list(NORMALIZATION_STEPS),
        "sources": {
            "behitek-vietnam-sensitive-words": {
                "repo": LEXICON_SOURCES["behitek-vietnam-sensitive-words"]["repo"],
                **behitek_stats,
            },
            "blue-eyes-vn-vietnamese-offensive-words": {
                "repo": LEXICON_SOURCES["blue-eyes-vn-vietnamese-offensive-words"]["repo"],
                **blueeyes_stats,
            },
        },
        "counts": {
            "behitek": len(behitek_terms),
            "blueeyes": len(blueeyes_terms),
            "overlap": len(overlap),
            "combined": len(combined),
        },
        "outputs": outputs,
    }
    report_path = output_dir / OUTPUT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    LOGGER.info("[lexicon] wrote %s", report_path)
    return report


def print_summary(report: Dict[str, Any]) -> None:
    counts = report.get("counts", {})
    LOGGER.info(
        "[lexicon] summary: behitek=%s blueeyes=%s overlap=%s combined=%s",
        counts.get("behitek"),
        counts.get("blueeyes"),
        counts.get("overlap"),
        counts.get("combined"),
    )
    for name, meta in sorted(report.get("outputs", {}).items()):
        LOGGER.info("[lexicon]   %-32s %6d terms  sha256=%s", name, meta.get("terms", 0), meta.get("sha256", "")[:16])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.lexicon",
        description="Download two Vietnamese lexicon datasets and extract sensitive words.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", nargs="*", default=list(STAGES), choices=STAGES, help="Which stages to run.")
    p.add_argument("--raw-dir", default=None, help="Override the raw lexicon directory.")
    p.add_argument("--output-dir", default=None, help="Override the extracted word-list directory.")
    p.add_argument("--force-download", action="store_true", help="Re-fetch raw files even if present.")
    p.add_argument("--show-terms", type=int, default=0, help="Print N sample combined terms (0 = none).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raw_dir = Path(args.raw_dir) if args.raw_dir else LEXICON_RAW_DIR
    output_dir = Path(args.output_dir) if args.output_dir else LEXICON_PROCESSED_DIR

    LOGGER.info("[lexicon] raw      -> %s", raw_dir)
    LOGGER.info("[lexicon] extracted -> %s", output_dir)

    if "download" in args.stage:
        download_lexicons(raw_dir, force=args.force_download)

    report: Optional[Dict[str, Any]] = None
    if "extract" in args.stage:
        report = build_lexicons(raw_dir, output_dir)

    if "summary" in args.stage:
        if report is None:
            report_path = output_dir / OUTPUT_REPORT
            if not report_path.is_file():
                LOGGER.error("[lexicon] %s not found - run the extract stage first.", report_path)
                return 1
            report = json.loads(report_path.read_text(encoding="utf-8"))
        print_summary(report)
        if args.show_terms > 0:
            combined = output_dir / OUTPUT_COMBINED
            if combined.is_file():
                terms = combined.read_text(encoding="utf-8").splitlines()[: args.show_terms]
                for term in terms:
                    LOGGER.info("[lexicon]   sample: %s", term)

    LOGGER.info("[lexicon] done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
