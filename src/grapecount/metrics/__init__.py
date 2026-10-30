"""Scoring: aggregate accuracy, Counting Fidelity, and cluster intervals.

Accuracy and fidelity can diverge, so a result is never quoted on MAE alone, and an interval
is never a fold SD.
"""

from .fidelity import aggregate_metrics, evaluate, fidelity_metrics, within_bunch_slope
from .interval import cluster_bootstrap, fold_spread, report

__all__ = ["aggregate_metrics", "evaluate", "fidelity_metrics", "within_bunch_slope",
           "cluster_bootstrap", "fold_spread", "report"]
