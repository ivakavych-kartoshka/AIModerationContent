"""Checkpoint-backed predictors for the language-aware pipeline.

Wraps a trained ``best_model/`` directory into the small ``predict_proba``
interface consumed by :class:`src.fusion.lang_aware_fusion.LangAwareFusion`.
The predictor never sees labels; it only maps raw text to ``(N, 3)``
probabilities in the fixed ``[HATE, OFFENSIVE, CLEAN]`` order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from ..common.logging_utils import get_logger
from ..models.model import CustomHeadModel, StandardHeadModel

LOGGER = get_logger("sensitiveai.inference")


class CheckpointPredictor:
    """Load a trained checkpoint once and predict probabilities in batches."""

    def __init__(
        self,
        checkpoint: str | Path,
        head: str = "standard",
        max_length: int = 128,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
        self.checkpoint = checkpoint
        self.head = head
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
        if head == "custom":
            self.model = CustomHeadModel.from_pretrained(checkpoint)
        else:
            self.model = StandardHeadModel.from_pretrained(checkpoint)
        self.model.to(self.device).eval()
        LOGGER.info("[inference] loaded %s head from %s on %s", head, checkpoint, self.device)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        texts = ["" if text is None else str(text) for text in texts]
        if not texts:
            return np.zeros((0, 3), dtype=float)
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    __call__ = predict_proba


class LanguageRoutedPredictor:
    """Route each text to the checkpoint of its detected language.

    Falls back to ``default`` (or the English model) for ``mix`` / ``other``;
    this keeps a single ``multilingual`` slot usable by ``LangAwareFusion``.
    """

    def __init__(
        self,
        models: dict[str, Any],
        detector: Any = None,
        default_lang: str = "en",
    ) -> None:
        self.models = dict(models)
        self.default_lang = default_lang if default_lang in self.models else next(iter(self.models))
        if detector is None:
            from ..langid.detector import LanguageDetector

            detector = LanguageDetector()
        self.detector = detector

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        texts = ["" if text is None else str(text) for text in texts]
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        buckets: dict[str, List[int]] = {}
        for index, text in enumerate(texts):
            lang = self.detector.detect(text).lang
            if lang not in self.models:
                lang = self.default_lang
            buckets.setdefault(lang, []).append(index)
        for lang, indices in buckets.items():
            probs = np.asarray(self.models[lang].predict_proba([texts[i] for i in indices]), dtype=float)
            for row, index in enumerate(indices):
                results[index] = probs[row]
        return np.vstack([row for row in results if row is not None])

    __call__ = predict_proba


__all__ = ["CheckpointPredictor", "LanguageRoutedPredictor"]
