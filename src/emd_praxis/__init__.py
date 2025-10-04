"""
EMD Praxis - IR&D Portfolio Optimization

A research codebase for defense R&D portfolio optimization using
machine learning and stochastic optimization.
"""

__version__ = "0.1.0"
__author__ = "Edward Danis"

# Expose key classes and functions at package level
from emd_praxis.data.synthetic_generator import generate_dataset
from emd_praxis.ml.model import train_and_eval, batch_annotate, MLResult
from emd_praxis.optimize.portfolio import OptimizeConfig, greedy_knapsack
from emd_praxis.validation import (
    validate_budget,
    validate_optimize_config,
    validate_analyzed_dataframe,
)

__all__ = [
    "__version__",
    "__author__",
    "generate_dataset",
    "train_and_eval",
    "batch_annotate",
    "MLResult",
    "OptimizeConfig",
    "greedy_knapsack",
    "validate_budget",
    "validate_optimize_config",
    "validate_analyzed_dataframe",
]
