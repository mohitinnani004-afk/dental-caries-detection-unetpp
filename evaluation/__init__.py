"""
Evaluation package for Dental Caries Detection.
"""
try:
    from .evaluator import Evaluator
except ImportError:
    Evaluator = None

__all__ = ["Evaluator"]
