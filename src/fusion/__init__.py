"""Hybrid fusion between the lexical rule engine and the AI classifier.

Implements :mod:`src.fusion.weighted_fusion` (score blending + a logistic-
regression meta-classifier) and :mod:`src.fusion.run_fusion` (the CLI entry
point).  See ``Rule_Engine_Plan.md`` sections 40-41.
"""

from .weighted_fusion import (
    COMPONENTS,
    MetaFusion,
    fuse_scores,
    fuse_scores_per_channel,
    meta_features,
    rule_pseudo_proba,
    score_alpha_grid,
    score_alpha_grid_per_channel,
    tune_thresholds_at_alpha,
    tune_thresholds_at_alphas,
)

__all__ = [
    "COMPONENTS",
    "MetaFusion",
    "fuse_scores",
    "fuse_scores_per_channel",
    "meta_features",
    "rule_pseudo_proba",
    "score_alpha_grid",
    "score_alpha_grid_per_channel",
    "tune_thresholds_at_alpha",
    "tune_thresholds_at_alphas",
]
