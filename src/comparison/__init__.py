"""Cross-model comparison and paper-ready output."""

from .compare_models import build_comparison_table, collect_runs, run  # noqa: F401

__all__ = ["build_comparison_table", "collect_runs", "run"]
