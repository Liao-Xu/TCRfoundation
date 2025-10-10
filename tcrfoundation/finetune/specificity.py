"""
Antigen specificity prediction.
This is essentially the same as classification with different data.
"""

from .classification import (
    MetatypeClassifier,
    SingleCellClassificationDataset,
    RepresentationDataset,
    load_foundation_model,
    train_classifier,
    train_epoch,
    evaluate,
    compute_metrics,
    extract_embeddings_and_predictions,
    extract_and_store_embeddings,
)

# Alias for specificity-specific naming
SpecificityClassifier = MetatypeClassifier
SpecificityDataset = SingleCellClassificationDataset

def train_specificity_classifier(*args, **kwargs):
    """Train specificity classifier. Wrapper around train_classifier."""
    return train_classifier(*args, **kwargs)

__all__ = [
    "SpecificityClassifier",
    "SpecificityDataset",
    "train_specificity_classifier",
    "load_foundation_model",
    "compute_metrics",
]
