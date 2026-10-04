"""Latency / FPS benchmark (plan section 27).

Latency and FPS are only comparable if they are measured with the same hardware,
the same batch size and the same inference configuration, so:

* the texts come from the **test split** of the fixed split;
* the tokenizer settings, ``max_length``, dtype and device are recorded in the
  output;
* ``torch.cuda.synchronize()`` is called around every timed region;
* a warm-up phase (default 10 batches) is discarded before timing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..common.constants import NUM_LABELS
from ..common.logging_utils import get_logger

LOGGER = get_logger(__name__)

PRECISION_TO_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
    precision: str = "fp32",
    warmup_batches: int = 10,
    timed_batches: int = 50,
    batch_sizes: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Return latency / throughput statistics for one model."""
    model.eval().to(device)
    dtype = PRECISION_TO_DTYPE.get(str(precision).lower(), torch.float32)
    use_amp = dtype != torch.float32 and device.type == "cuda"

    samples: List[str] = list(texts) or ["xin chào"]
    if len(samples) < batch_size * max(warmup_batches + timed_batches, 1):
        repeats = (batch_size * (warmup_batches + timed_batches) // max(len(samples), 1)) + 1
        samples = (samples * repeats)[: batch_size * (warmup_batches + timed_batches)]

    def encode(chunk: List[str]):
        return tokenizer(
            chunk,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def timed_run(bs: int, n_warmup: int, n_timed: int) -> Dict[str, float]:
        for _ in range(n_warmup):
            batch = encode([samples[i % len(samples)] for i in range(bs)])
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if device.type == "cuda":
                torch.cuda.synchronize()

        latencies: List[float] = []
        total = 0
        for step in range(n_timed):
            batch = encode([samples[(step * bs + i) % len(samples)] for i in range(bs)])
            batch = {k: v.to(device) for k, v in batch.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            total += bs

        arr = torch.tensor(latencies) * 1000.0  # ms per batch
        mean_ms = float(arr.mean())
        return {
            "batch_size": bs,
            "num_batches": n_timed,
            "num_samples": total,
            "latency_ms_per_batch": round(mean_ms, 4),
            "latency_ms_per_sample": round(mean_ms / bs, 4),
            "latency_p50_ms_per_batch": round(float(arr.median()), 4),
            "latency_p95_ms_per_batch": round(float(torch.quantile(arr, 0.95)), 4),
            "throughput_samples_per_second": round(total / (sum(latencies)), 3),
        }

    primary = timed_run(batch_size, warmup_batches, timed_batches)
    extra = {}
    for bs in batch_sizes or []:
        if int(bs) != int(batch_size):
            extra[f"batch_size_{int(bs)}"] = timed_run(int(bs), max(2, warmup_batches // 3), max(5, timed_batches // 3))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        batch = encode([samples[i % len(samples)] for i in range(batch_size)])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        peak_mb = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
    else:
        peak_mb = float("nan")

    result: Dict[str, Any] = {
        **primary,
        "fps": primary["throughput_samples_per_second"],
        "num_classes": NUM_LABELS,
        "max_length": int(max_length),
        "precision": str(precision),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_gpu_memory_mb": peak_mb,
        "warmup_batches": int(warmup_batches),
        "timed_batches": int(timed_batches),
        "note": "Tokenisation excluded; identical tokenizer/max_length/dtype for all four models.",
    }
    if extra:
        result["other_batch_sizes"] = extra
    LOGGER.info(
        "[bench] batch_size=%d  %.3f ms/batch  %.4f ms/sample  %.1f samples/s",
        batch_size,
        primary["latency_ms_per_batch"],
        primary["latency_ms_per_sample"],
        primary["throughput_samples_per_second"],
    )
    return result


__all__ = ["benchmark_model"]
