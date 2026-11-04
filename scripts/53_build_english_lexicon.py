"""Step 53 - mine the English rule-engine lexicon from the train split.

    python scripts/53_build_english_lexicon.py
    python scripts/53_build_english_lexicon.py --min-count 5 --min-lift 1.4
    python scripts/53_build_english_lexicon.py --merge-with data/processed/lexicon/verified_lexicon_en.csv \
        --output data/processed/lexicon/verified_lexicon_en_combined.csv

Reads ONLY ``data/splits_en/train.csv``.  Validation and test are never used,
so the extracted lexicon stays inside the project's leakage protocol.

Outputs
-------
``data/processed/lexicon/verified_lexicon_en_extracted.csv``
    mined entries, same schema as ``verified_lexicon.csv`` (review_status=auto)
``data/processed/lexicon/english/lexicon_report.json``
    parameters, class priors, top terms
``--output`` (optional, when ``--merge-with`` is given)
    seed + extracted terms de-duplicated by ``term``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.constants import HATE_SPEECH_OFFENSIVE_SPLITS_DIR, LEXICON_PROCESSED_DIR  # noqa: E402
from src.data.english_lexicon import extract_lexicon  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/53_build_english_lexicon.py",
        description="Extract an English HATE/OFFENSIVE/CLEAN lexicon from the train split only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train", default=str(HATE_SPEECH_OFFENSIVE_SPLITS_DIR / "train.csv"))
    p.add_argument("--output", default=str(LEXICON_PROCESSED_DIR / "verified_lexicon_en_extracted.csv"))
    p.add_argument("--report", default=str(LEXICON_PROCESSED_DIR / "english" / "lexicon_report.json"))
    p.add_argument("--merge-with", default=None, help="Seed/curated lexicon merged before the extracted terms.")
    p.add_argument("--min-count", type=int, default=15)
    p.add_argument("--min-lift", type=float, default=2.0)
    p.add_argument("--min-hate-precision", type=float, default=0.40)
    p.add_argument("--min-offensive-precision", type=float, default=0.85)
    p.add_argument("--min-clean-precision", type=float, default=0.60)
    p.add_argument("--min-wilson", type=float, default=0.35)
    p.add_argument("--min-prior-margin", type=float, default=0.08)
    p.add_argument("--max-terms", type=int, default=1500)
    p.add_argument("--include-bigrams", action="store_true", help="Also mine two-word phrases (needs stronger evidence).")
    p.add_argument("--include-clean", action="store_true", help="Keep CLEAN lexicon entries (default: drop them).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = extract_lexicon(
        train_csv=args.train,
        output_csv=args.output,
        report_path=args.report,
        min_count=args.min_count,
        min_lift=args.min_lift,
        min_hate_precision=args.min_hate_precision,
        min_offensive_precision=args.min_offensive_precision,
        min_clean_precision=args.min_clean_precision,
        min_wilson=args.min_wilson,
        min_prior_margin=args.min_prior_margin,
        max_terms=args.max_terms,
        include_bigrams=args.include_bigrams,
        include_clean=args.include_clean,
        merge_with=args.merge_with,
    )
    print(f"extracted : {report['n_extracted']}")
    print(f"written   : {report['n_written']}")
    print(f"output    : {report['output_csv']}")
    print(f"top hate  : {', '.join(report['top_hate'][:15])}")
    print(f"top off.  : {', '.join(report['top_offensive'][:15])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
