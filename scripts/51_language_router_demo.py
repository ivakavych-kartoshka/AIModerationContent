"""Step 51 - language routing demo (Vietnamese vs English vs code-mix).

    python scripts/51_language_router_demo.py
    python scripts/51_language_router_demo.py --text "may noi the la sai roi"
    python scripts/51_language_router_demo.py --json
    python scripts/51_language_router_demo.py --ai-probs 0.1,0.2,0.7   # simulated AI output

What it shows
-------------
1. :class:`LanguageDetector` -> ``vi`` / ``en`` / ``mix`` / ``other``
2. :class:`LanguageAwareTokenizer` -> per-token tags + per-language segments
3. :class:`LanguageRouter` -> which lexicon scans each span (both for ``mix``)
4. :class:`LangAwareFusion` -> rule scores + (optional) AI probabilities, fused
   with a per-language alpha from ``configs/language_router.yaml``

No training or evaluation is touched.  Without ``--ai-probs`` the fusion runs
rule-only (``ai_used=false``); that is the correct behaviour when no AI
checkpoint is attached yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.constants import LABELS  # noqa: E402
from src.fusion.lang_aware_fusion import LangAwareFusion  # noqa: E402
from src.inference.predictor import CheckpointPredictor  # noqa: E402

DEFAULT_TEXTS = [
    "Mày nói thế là xúc phạm quá đấy!",
    "You are such an idiot.",
    "Mày nói shit ghê thật đấy",
    "may noi the la sai roi",
    "the cat sat on the mat",
    "piece of shit",
    "đồ ngu",
    "ok",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/language_router.yaml")
    parser.add_argument("--text", action="append", default=None, help="Text to scan (repeatable).")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")
    parser.add_argument(
        "--ai-probs",
        default=None,
        help="Optional simulated AI probabilities 'HATE,OFFENSIVE,CLEAN' (e.g. 0.1,0.2,0.7).",
    )
    parser.add_argument("--vi-checkpoint", default=None, help="Vietnamese best_model directory.")
    parser.add_argument("--en-checkpoint", default=None, help="English best_model directory.")
    parser.add_argument("--vi-head", default="standard", choices=["standard", "custom"])
    parser.add_argument("--en-head", default="standard", choices=["standard", "custom"])
    parser.add_argument("--vi-max-length", type=int, default=256)
    parser.add_argument("--en-max-length", type=int, default=128)
    return parser


def _parse_probs(raw: str | None):
    if not raw:
        return None
    parts = [float(part) for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(parts) != len(LABELS):
        raise SystemExit(f"--ai-probs needs {len(LABELS)} values, got {len(parts)}")
    total = sum(parts)
    if total <= 0:
        raise SystemExit("--ai-probs values must sum to > 0")
    return [part / total for part in parts]


def main() -> int:
    args = build_parser().parse_args()
    probs = _parse_probs(args.ai_probs)
    predictor = (lambda texts, _p=probs: [_p for _ in texts]) if probs else None

    fusion = LangAwareFusion.from_config(args.config)
    if predictor is not None:
        fusion.set_model("multilingual", predictor)
        fusion.set_model("vi", predictor)
        fusion.set_model("en", predictor)

    if args.vi_checkpoint:
        fusion.set_model(
            "vi",
            CheckpointPredictor(args.vi_checkpoint, head=args.vi_head, max_length=args.vi_max_length),
        )
    if args.en_checkpoint:
        fusion.set_model(
            "en",
            CheckpointPredictor(args.en_checkpoint, head=args.en_head, max_length=args.en_max_length),
        )

    texts = args.text or DEFAULT_TEXTS
    results = [fusion.analyze(text).to_dict() for text in texts]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for row in results:
        matches = ", ".join(
            f"{match.get('term')}[{match.get('label')}]" for match in row.get("matches", [])
        ) or "-"
        scores = " ".join(
            f"{label}={row['fused_scores'].get(label, 0.0):.3f}" for label in LABELS
        )
        print(f"text      : {row['text']}")
        print(f"language  : {row['lang']} (confidence {row['confidence']:.3f})")
        print(f"ai model  : {row['model_language']}  alpha={row['alpha']:.2f}  ai_used={row['ai_used']}")
        print("segments  : " + " | ".join(f"{seg['lang']}:{seg['text']!r}" for seg in row["segments"]))
        print(f"matches   : {matches}")
        print(f"fused     : {scores} -> {row['label']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
