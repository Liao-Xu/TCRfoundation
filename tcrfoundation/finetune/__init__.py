"""Fine-tuning tasks for TCRfoundation."""

from . import classification
from . import specificity
from . import avidity
from . import cross_modal

__all__ = [
    "classification",
    "specificity",
    "avidity",
    "cross_modal",
]
