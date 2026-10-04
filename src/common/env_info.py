"""Runtime environment report (GPU, CUDA, library versions).

Used by ``model_summary.txt`` (plan section 27) and by the comparison tables
so that latency / FPS numbers are always attributable to a known hardware setup.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict

from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def _library_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "not installed"


def cuda_version() -> str:
    """CUDA runtime version actually used by torch (not the driver version)."""
    try:
        import torch

        if torch.version.cuda:
            return str(torch.version.cuda)
        return "not available"
    except Exception:
        return "unknown"


def driver_version() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "unknown"


def gpu_name(index: int = 0) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(index)
    except Exception:
        pass
    return "CPU"


def gpu_memory_gb(index: int = 0) -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 0.0


def collect_environment() -> Dict[str, Any]:
    """Everything needed for a reproducible hardware/software statement."""
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        torch_version, cuda_available = "unknown", False

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch": torch_version,
        "transformers": _library_version("transformers"),
        "tokenizers": _library_version("tokenizers"),
        "datasets": _library_version("datasets"),
        "sklearn": _library_version("sklearn"),
        "pandas": _library_version("pandas"),
        "numpy": _library_version("numpy"),
        "matplotlib": _library_version("matplotlib"),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version(),
        "cudnn_version": (
            str(__import__("torch").backends.cudnn.version())
            if cuda_available
            else "not available"
        ),
        "gpu": gpu_name(),
        "gpu_memory_gb": gpu_memory_gb(),
        "gpu_count": (
            __import__("torch").cuda.device_count() if cuda_available else 0
        ),
        "nvidia_driver": driver_version(),
    }
    if not cuda_available:
        LOGGER.warning("CUDA is not available - training will run on CPU (very slow).")
    return env


def format_environment(env: Dict[str, Any] | None = None) -> str:
    env = env or collect_environment()
    width = max(len(k) for k in env)
    lines = ["Hardware / software environment", "-" * (width + 24)]
    for key, value in env.items():
        lines.append(f"{key:<{width}} : {value}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(format_environment())
