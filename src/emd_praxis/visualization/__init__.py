"""
Visualization utilities for EMD Praxis IR&D Portfolio Optimization.

Provides plotting functions for model evaluation, portfolio analysis, and results interpretation.
"""

from emd_praxis.visualization.plots import (
    plot_calibration_curve,
    plot_feature_importances,
    plot_portfolio_distribution,
    plot_risk_vs_return,
)

__all__ = [
    "plot_calibration_curve",
    "plot_feature_importances",
    "plot_portfolio_distribution",
    "plot_risk_vs_return",
]
