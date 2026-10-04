"""Step 4 - sanity checks before spending GPU hours.

    python scripts/04_check_setup.py

Checks:
  * Python / library versions, GPU and CUDA availability
  * every YAML config loads and validates
  * the custom head builds and reports its parameter count
  * the fixed split exists and matches its fingerprints (if already built)
  * a tiny end-to-end forward pass through both heads
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.common.config import get, load_config, resolve_config_path  # noqa: E402
from src.common.constants import MODEL_NAMES, NUM_LABELS  # noqa: E402
from src.common.env_info import format_environment  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.data.split import split_fingerprints, verify_split_fingerprint  # noqa: E402
from src.training.stages import resolve_stages, stages_summary  # noqa: E402

LOGGER = get_logger("sensitiveai.check")

FAILURES: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "OK  " if ok else "FAIL"
    LOGGER.info("[%s] %-44s %s", status, label, detail)
    if not ok:
        FAILURES.append(label)


def main() -> int:
    LOGGER.info("=" * 78)
    LOGGER.info(format_environment())
    LOGGER.info("=" * 78)

    check("CUDA available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    # ---- configs ------------------------------------------------------- #
    for model in MODEL_NAMES:
        try:
            path = resolve_config_path(model)
            cfg = load_config(path)
            stages = resolve_stages(cfg)
            total_epochs = sum(s.num_epochs for s in stages)
            check(
                f"config {model}",
                True,
                f"head={cfg['head']:<8} stages={len(stages)} epochs={total_epochs} lr={get(cfg, 'training.learning_rate')}",
            )
        except Exception as exc:
            check(f"config {model}", False, str(exc))

    # ---- split ---------------------------------------------------------- #
    splits_dir = Path("data/splits")
    if (splits_dir / "split_manifest.json").is_file():
        try:
            verify_split_fingerprint(splits_dir)
            check("fixed split", True, str({k: v[:12] for k, v in split_fingerprints(splits_dir).items()}))
        except Exception as exc:
            check("fixed split", False, str(exc))
    else:
        LOGGER.info("[SKIP] fixed split not built yet -> run: python scripts/00_prepare_data.py")

    # ---- models --------------------------------------------------------- #
    try:
        from transformers import AutoTokenizer

        from src.models.model import build_model, count_parameters

        tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=True)
        batch = tokenizer(
            ["xin chao Viet Nam", "do vo dich"], padding="max_length", truncation=True,
            max_length=32, return_tensors="pt",
        )
        for head in ("standard", "custom"):
            model = build_model(head=head, base_model="microsoft/mdeberta-v3-base", num_labels=NUM_LABELS)
            model.eval()
            with torch.no_grad():
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            counts = count_parameters(model)
            check(
                f"model head={head}",
                tuple(out.logits.shape) == (2, NUM_LABELS),
                f"logits={tuple(out.logits.shape)} params={counts['total_parameters']:,}",
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        check("model construction", False, str(exc))

    # ---- training plan -------------------------------------------------- #
    try:
        cfg = load_config(resolve_config_path("sensitiveai-vi-custom-curriculum"))
        LOGGER.info("\n%s", stages_summary(resolve_stages(cfg)))
    except Exception as exc:
        LOGGER.warning("could not print the curriculum plan: %s", exc)

    LOGGER.info("=" * 78)
    if FAILURES:
        LOGGER.error("%d check(s) failed: %s", len(FAILURES), ", ".join(FAILURES))
        return 1
    LOGGER.info("All checks passed. Next: python scripts/00_prepare_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
