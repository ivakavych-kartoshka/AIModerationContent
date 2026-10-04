"""Torch datasets / dataloaders over the fixed ViHSD split.

Tokenisation is done **once** with the base-model tokenizer
(``microsoft/mdeberta-v3-base``) and cached in memory, so every epoch reuses the
same tensors and the four compared systems are guaranteed to see byte-identical
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..common.constants import LABEL_COLUMN, NUM_LABELS, TEXT_COLUMN
from ..common.logging_utils import get_logger
from ..common.seeding import make_generator, worker_init_fn
from .split import load_split

LOGGER = get_logger(__name__)


class TokenizedTextDataset(Dataset):
    """Pre-tokenized dataset returning ``input_ids``/``attention_mask``/``labels``."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.indices = list(indices) if indices is not None else list(range(len(self.labels)))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = int(self.indices[idx])
        return {
            "input_ids": self.input_ids[row],
            "attention_mask": self.attention_mask[row],
            "labels": self.labels[idx],
        }

    @property
    def label_counts(self) -> np.ndarray:
        return np.bincount(self.labels.numpy(), minlength=NUM_LABELS)


def _select_rows(
    df: pd.DataFrame,
    max_samples: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Optionally subsample (debug runs only - never used for the paper results)."""
    if not max_samples or max_samples >= len(df):
        return df
    # stratified subsample so the label balance survives the debug cut
    frac = max_samples / len(df)
    keep: List[int] = []
    for _, group in df.groupby(LABEL_COLUMN, sort=True):
        take = min(len(group), max(1, int(round(len(group) * frac))))
        keep.extend(group.sample(n=take, random_state=seed).index.tolist())
    return df.loc[sorted(keep)].reset_index(drop=True)


def empty_tokenized_dataset(max_length: int) -> TokenizedTextDataset:
    """A zero-length dataset with correctly shaped (empty) tensors."""
    return TokenizedTextDataset(
        texts=[],
        labels=[],
        input_ids=torch.empty((0, max_length), dtype=torch.long),
        attention_mask=torch.empty((0, max_length), dtype=torch.long),
    )


def encode_split(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    num_proc: Optional[int] = None,
) -> TokenizedTextDataset:
    """Tokenize one split into padded tensors."""
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(int).tolist()

    if not texts:
        # fast tokenizers reject an empty batch
        return empty_tokenized_dataset(max_length)

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return TokenizedTextDataset(
        texts=texts,
        labels=labels,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )


@dataclass
class DataBundle:
    """All three tokenized splits plus the raw frames, ready for training."""

    train: TokenizedTextDataset
    validation: TokenizedTextDataset
    test: TokenizedTextDataset
    raw: Dict[str, pd.DataFrame]
    max_length: int
    splits_dir: str

    @property
    def sizes(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def build_datasets(
    tokenizer,
    splits_dir: str | Path,
    max_length: int = 256,
    seed: int = 42,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    splits: Optional[Sequence[str]] = None,
) -> DataBundle:
    """Load the fixed split and tokenize it.

    ``splits`` limits which splits are actually tokenized (e.g. evaluation only
    needs ``test``); the remaining entries come back as empty datasets so the
    :class:`DataBundle` shape stays stable.
    """
    wanted = tuple(splits) if splits is not None else ("train", "validation", "test")
    unknown = set(wanted) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"Unknown splits requested: {sorted(unknown)}")

    frames = load_split(splits_dir)
    for name, df in frames.items():
        if df[text_column].isna().any():
            raise ValueError(f"Split '{name}' contains null text - re-run preprocessing.")
        labels = set(df[label_column].astype(int).unique())
        if not labels <= set(range(NUM_LABELS)):
            raise ValueError(f"Split '{name}' has labels outside 0..{NUM_LABELS - 1}: {sorted(labels)}")

    selected = {
        "train": _select_rows(frames["train"], max_train_samples, seed),
        "validation": _select_rows(frames["validation"], max_eval_samples, seed),
        "test": _select_rows(frames["test"], max_eval_samples, seed),
    }

    LOGGER.info(
        "[data] tokenising (max_length=%d) %s",
        max_length,
        ", ".join(f"{name}={len(selected[name])}" for name in wanted),
    )
    encoded = {
        name: encode_split(selected[name], tokenizer, max_length, text_column, label_column)
        for name in wanted
    }
    bundle = DataBundle(
        train=encoded.get("train") or empty_tokenized_dataset(max_length),
        validation=encoded.get("validation") or empty_tokenized_dataset(max_length),
        test=encoded.get("test") or empty_tokenized_dataset(max_length),
        raw={name: selected[name] for name in wanted},
        max_length=max_length,
        splits_dir=str(Path(splits_dir)),
    )
    if len(bundle.train):
        LOGGER.info(
            "[data] label counts (train): %s",
            dict(zip(range(NUM_LABELS), bundle.train.label_counts.tolist())),
        )
    return bundle


def collate_tokenized_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for :class:`TokenizedTextDataset`.

    Defined at module level (not as a lambda) so that it survives pickling by
    the multiprocessing ``spawn`` start method used on Windows.
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def make_dataloader(
    dataset: TokenizedTextDataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 2,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_tokenized_batch,
        generator=generator if generator is not None else (make_generator(seed) if shuffle else None),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def concat_datasets(datasets: Sequence[TokenizedTextDataset]) -> TokenizedTextDataset:
    """Concatenate tokenized splits while keeping unique global indices."""
    input_ids = torch.cat([d.input_ids for d in datasets], dim=0)
    attention_mask = torch.cat([d.attention_mask for d in datasets], dim=0)
    labels = torch.cat([d.labels for d in datasets], dim=0)
    texts: List[str] = []
    indices: List[int] = []
    for d in datasets:
        texts.extend(d.texts)
        indices.extend(d.indices)
    return TokenizedTextDataset(texts, labels.tolist(), input_ids, attention_mask, indices)
