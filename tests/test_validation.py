import pytest
import pandas as pd
from emd_praxis.validation import (
    validate_budget, validate_horizon, validate_dataframe_columns,
    validate_generate_params, validate_ml_params, validate_optimize_config,
    validate_analyzed_dataframe
)
from emd_praxis.optimize.portfolio import OptimizeConfig


def test_validate_budget_positive():
    validate_budget(1000.0)
    validate_budget(1e6)


def test_validate_budget_negative():
    with pytest.raises(ValueError, match="must be positive"):
        validate_budget(-100)


def test_validate_budget_zero():
    with pytest.raises(ValueError, match="must be positive"):
        validate_budget(0)


def test_validate_budget_too_large():
    with pytest.raises(ValueError, match="exceeds reasonable bounds"):
        validate_budget(2e12)


def test_validate_horizon_valid():
    validate_horizon(1)
    validate_horizon(2)
    validate_horizon(3)


def test_validate_horizon_invalid():
    with pytest.raises(ValueError, match="must be 1, 2, or 3"):
        validate_horizon(0)
    with pytest.raises(ValueError, match="must be 1, 2, or 3"):
        validate_horizon(4)


def test_validate_dataframe_columns_valid():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    validate_dataframe_columns(df, {"a", "b"})


def test_validate_dataframe_columns_missing():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe_columns(df, {"a", "b", "c"})


def test_validate_generate_params_valid():
    validate_generate_params(100, 42)


def test_validate_generate_params_negative_n():
    with pytest.raises(ValueError, match="n must be positive"):
        validate_generate_params(-10, 42)


def test_validate_generate_params_zero_n():
    with pytest.raises(ValueError, match="n must be positive"):
        validate_generate_params(0, 42)


def test_validate_generate_params_too_large():
    with pytest.raises(ValueError, match="exceeds reasonable bounds"):
        validate_generate_params(2_000_000, 42)


def test_validate_generate_params_negative_seed():
    with pytest.raises(ValueError, match="seed must be non-negative"):
        validate_generate_params(100, -1)


def test_validate_ml_params_valid():
    validate_ml_params(100, 0.05, 3, 5)


def test_validate_ml_params_negative_estimators():
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        validate_ml_params(-10, 0.05, 3, 5)


def test_validate_ml_params_invalid_learning_rate():
    with pytest.raises(ValueError, match="learning_rate must be"):
        validate_ml_params(100, 0, 3, 5)
    with pytest.raises(ValueError, match="learning_rate must be"):
        validate_ml_params(100, 1.5, 3, 5)


def test_validate_ml_params_negative_depth():
    with pytest.raises(ValueError, match="max_depth must be positive"):
        validate_ml_params(100, 0.05, -1, 5)


def test_validate_ml_params_invalid_cv():
    with pytest.raises(ValueError, match="cv_folds must be >= 2"):
        validate_ml_params(100, 0.05, 3, 1)


def test_validate_optimize_config_valid():
    cfg = OptimizeConfig(
        total_budget=1e6,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=100,
        budget_volatility=0.2,
        seed=42
    )
    validate_optimize_config(cfg)


def test_validate_optimize_config_negative_budget():
    cfg = OptimizeConfig(
        total_budget=-1000,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=100,
        budget_volatility=0.2,
        seed=42
    )
    with pytest.raises(ValueError, match="must be positive"):
        validate_optimize_config(cfg)


def test_validate_optimize_config_horizon_pct_not_sum_to_one():
    cfg = OptimizeConfig(
        total_budget=1e6,
        horizon_pct={"h1": 0.4, "h2": 0.4, "h3": 0.1},  # Sums to 0.9
        risk_limits={"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=100,
        budget_volatility=0.2,
        seed=42
    )
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_optimize_config(cfg)


def test_validate_optimize_config_invalid_risk_limit():
    cfg = OptimizeConfig(
        total_budget=1e6,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 1.5, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=100,
        budget_volatility=0.2,
        seed=42
    )
    with pytest.raises(ValueError, match="must be in"):
        validate_optimize_config(cfg)


def test_validate_optimize_config_weights_not_sum_to_one():
    cfg = OptimizeConfig(
        total_budget=1e6,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
        weights={"strategic": 0.5, "financial": 0.3},  # Sums to 0.8
        scenarios_n=100,
        budget_volatility=0.2,
        seed=42
    )
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        validate_optimize_config(cfg)


def test_validate_optimize_config_invalid_volatility():
    cfg = OptimizeConfig(
        total_budget=1e6,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=100,
        budget_volatility=1.5,
        seed=42
    )
    with pytest.raises(ValueError, match="budget_volatility must be in"):
        validate_optimize_config(cfg)


def test_validate_analyzed_dataframe_valid():
    df = pd.DataFrame({
        "project_name": ["A", "B"],
        "tech_area": ["AI", "Cyber"],
        "horizon": [1, 2],
        "budget_musd": [5.0, 3.0],
        "predicted_success_probability": [0.7, 0.5],
        "initial_trl": [2, 3],
        "target_trl": [5, 6]
    })
    validate_analyzed_dataframe(df)


def test_validate_analyzed_dataframe_missing_column():
    df = pd.DataFrame({
        "project_name": ["A", "B"],
        "tech_area": ["AI", "Cyber"],
        "horizon": [1, 2],
        "budget_musd": [5.0, 3.0]
    })
    with pytest.raises(ValueError, match="missing required columns"):
        validate_analyzed_dataframe(df)


def test_validate_analyzed_dataframe_invalid_horizon():
    df = pd.DataFrame({
        "project_name": ["A", "B"],
        "tech_area": ["AI", "Cyber"],
        "horizon": [1, 5],  # 5 is invalid
        "budget_musd": [5.0, 3.0],
        "predicted_success_probability": [0.7, 0.5],
        "initial_trl": [2, 3],
        "target_trl": [5, 6]
    })
    with pytest.raises(ValueError, match="invalid horizon values"):
        validate_analyzed_dataframe(df)


def test_validate_analyzed_dataframe_negative_budget():
    df = pd.DataFrame({
        "project_name": ["A", "B"],
        "tech_area": ["AI", "Cyber"],
        "horizon": [1, 2],
        "budget_musd": [5.0, -3.0],  # Negative budget
        "predicted_success_probability": [0.7, 0.5],
        "initial_trl": [2, 3],
        "target_trl": [5, 6]
    })
    with pytest.raises(ValueError, match="negative or zero budget"):
        validate_analyzed_dataframe(df)


def test_validate_analyzed_dataframe_invalid_probability():
    df = pd.DataFrame({
        "project_name": ["A", "B"],
        "tech_area": ["AI", "Cyber"],
        "horizon": [1, 2],
        "budget_musd": [5.0, 3.0],
        "predicted_success_probability": [0.7, 1.5],  # > 1.0
        "initial_trl": [2, 3],
        "target_trl": [5, 6]
    })
    with pytest.raises(ValueError, match="must be in"):
        validate_analyzed_dataframe(df)
