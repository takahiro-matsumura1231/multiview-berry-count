"""Model definitions.

Two stages, deliberately separate objects: `BackboneCounter` fine-tunes the CNN on grape
frames, and `ViewTokenCounter` trains on features cached from that backbone. See backbone.py
for why the backbone is "frozen" only after fine-tuning.

`aggregators.py` holds a registry of alternative stage-2 models behind one factory, sharing
width, depth and head so that a difference between them is a difference in how the K views
are pooled. `ViewTokenCounter` is wrapped, not edited.
"""

from .aggregators import (
    AGGREGATOR_SOURCE, AGGREGATORS, AggregatorConfig, AttentionMILCounter, DeepSetsCounter,
    SetTransformerCounter, ViewTokenAggregator, make_aggregator, n_parameters,
)
from .backbone import AttentionMIL, BackboneCounter, make_backbone, parameter_groups
from .viewtoken import FeatureScaler, ViewTokenCounter

__all__ = ["AttentionMIL", "BackboneCounter", "make_backbone", "parameter_groups",
           "FeatureScaler", "ViewTokenCounter",
           "AGGREGATORS", "AGGREGATOR_SOURCE", "AggregatorConfig", "AttentionMILCounter",
           "DeepSetsCounter", "SetTransformerCounter", "ViewTokenAggregator",
           "make_aggregator", "n_parameters"]
