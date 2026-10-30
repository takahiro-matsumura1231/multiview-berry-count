"""Fixed per-view feature extractors.

Every model here is external, pretrained, and has never seen a bunch-total count, so the
features it produces cannot leak the regression target. They are also fold-independent, so
they are computed once over the whole dataset rather than per fold.

Features are stored RAW. Standardisation happens inside the trainer using TRAIN-FOLD
statistics only; doing it here would leak across the split.
"""
