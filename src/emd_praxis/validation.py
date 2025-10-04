"""
Input validation utilities for the EMD Praxis pipeline.
"""
from __future__ import annotations
from typing import Set
import pandas as pd
from emd_praxis.optimize.portfolio import OptimizeConfig


def validate_budget(budget: float, param_name: str = "budget") -> None:
    """Validate budget parameter."""
    if budget <= 0:
        raise ValueError(f"{param_name} must be positive, got {budget}")
    if budget > 1e12:  # $1 trillion sanity check
        raise ValueError(f"{param_name} exceeds reasonable bounds: {budget}")


def validate_horizon(horizon: int) -> None:
    """Validate horizon parameter."""
    if horizon not in {1, 2, 3}:
        raise ValueError(f"horizon must be 1, 2, or 3, got {horizon}")


def validate_dataframe_columns(df: pd.DataFrame, required_cols: Set[str]) -> None:
    """Validate that dataframe contains required columns."""
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def validate_generate_params(n: int, seed: int) -> None:
    """Validate parameters for dataset generation."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > 1_000_000:
        raise ValueError(f"n exceeds reasonable bounds: {n}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")


def validate_ml_params(n_estimators: int, learning_rate: float, max_depth: int, cv_folds: int) -> None:
    """Validate ML model hyperparameters."""
    if n_estimators <= 0:
        raise ValueError(f"n_estimators must be positive, got {n_estimators}")
    if learning_rate <= 0 or learning_rate > 1:
        raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
    if max_depth <= 0:
        raise ValueError(f"max_depth must be positive, got {max_depth}")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")


def validate_optimize_config(cfg: OptimizeConfig) -> None:
    """Validate optimization configuration."""
    # Budget validation
    validate_budget(cfg.total_budget, "total_budget")

    # Horizon percentages must sum to 1.0
    horizon_sum = sum(cfg.horizon_pct.values())
    if not (0.99 <= horizon_sum <= 1.01):  # Allow small floating point errors
        raise ValueError(f"horizon_pct values must sum to 1.0, got {horizon_sum}")

    # Each horizon percentage must be valid
    for key, val in cfg.horizon_pct.items():
        if not (0 <= val <= 1):
            raise ValueError(f"horizon_pct[{key}] must be in [0, 1], got {val}")

    # Risk limits validation
    for key, val in cfg.risk_limits.items():
        if not (0 <= val <= 1):
            raise ValueError(f"risk_limits[{key}] must be in [0, 1], got {val}")

    # Weights validation
    for key, val in cfg.weights.items():
        if val < 0:
            raise ValueError(f"weights[{key}] must be non-negative, got {val}")

    weight_sum = sum(cfg.weights.values())
    if not (0.99 <= weight_sum <= 1.01):
        raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

    # Scenarios validation
    if cfg.scenarios_n <= 0:
        raise ValueError(f"scenarios_n must be positive, got {cfg.scenarios_n}")

    # Budget volatility validation
    if cfg.budget_volatility < 0 or cfg.budget_volatility > 1:
        raise ValueError(f"budget_volatility must be in [0, 1], got {cfg.budget_volatility}")

    # Seed validation
    if cfg.seed < 0:
        raise ValueError(f"seed must be non-negative, got {cfg.seed}")


def validate_analyzed_dataframe(df: pd.DataFrame) -> None:
    """Validate dataframe for optimization (must have ML predictions)."""
    required_cols = {
        "project_name", "tech_area", "horizon", "budget_musd",
        "predicted_success_probability", "initial_trl", "target_trl"
    }
    validate_dataframe_columns(df, required_cols)

    # Check horizon values
    invalid_horizons = df[~df["horizon"].isin([1, 2, 3])]
    if len(invalid_horizons) > 0:
        raise ValueError(f"Found {len(invalid_horizons)} rows with invalid horizon values")

    # Check budget values
    if (df["budget_musd"] <= 0).any():
        raise ValueError("Found negative or zero budget values")

    # Check probability values
    if not df["predicted_success_probability"].between(0, 1).all():
        raise ValueError("predicted_success_probability must be in [0, 1]")
