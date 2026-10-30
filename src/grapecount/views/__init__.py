"""View-set selection: one implementation, not four."""

from .sampling import (
    PARTIAL_ORBIT_MIXTURE, evaluation_view_sets, phase_indices, phase_indices_window,
    sample_training_views, tau_of,
)

__all__ = ["PARTIAL_ORBIT_MIXTURE", "evaluation_view_sets", "phase_indices", "phase_indices_window",
           "sample_training_views", "tau_of"]
