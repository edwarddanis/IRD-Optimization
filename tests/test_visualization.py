import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
from emd_praxis.data.synthetic_generator import generate_dataset
from emd_praxis.ml.model import train_and_eval, batch_annotate, generate_calibration_data
from emd_praxis.optimize.portfolio import OptimizeConfig, greedy_knapsack
from emd_praxis.visualization import (
    plot_calibration_curve,
    plot_feature_importances,
    plot_portfolio_distribution,
    plot_risk_vs_return
)


def test_plot_calibration_curve_saves_file():
    """Test that calibration curve plot is saved to file."""
    # Generate simple calibration data
    cal_data = {
        "prob_true": [0.1, 0.3, 0.5, 0.7, 0.9],
        "prob_pred": [0.15, 0.35, 0.5, 0.65, 0.85],
        "bin_counts": [10, 15, 20, 15, 10],
        "n_bins": 5
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "calibration.png")
        plot_calibration_curve(cal_data, output_path=output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_plot_calibration_curve_with_real_data():
    """Test calibration curve plotting with real ML output."""
    df = generate_dataset(100, seed=42)
    model, res = train_and_eval(df, seed=42, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "calibration.png")
        plot_calibration_curve(res.calibration_curve, output_path=output_path)
        assert os.path.exists(output_path)


def test_plot_feature_importances_saves_file():
    """Test that feature importance plot is saved to file."""
    importances = {
        "budget_musd": 0.25,
        "team_size": 0.20,
        "complexity": 0.15,
        "novelty": 0.10,
        "stakeholder_support": 0.08,
        "market_potential": 0.07,
        "duration_months": 0.06,
        "initial_trl": 0.05,
        "target_trl": 0.03,
        "horizon": 0.01
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "importance.png")
        plot_feature_importances(importances, output_path=output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_plot_feature_importances_with_real_data():
    """Test feature importance plotting with real ML output."""
    df = generate_dataset(100, seed=42)
    model, res = train_and_eval(df, seed=42, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "importance.png")
        plot_feature_importances(res.feature_importances, output_path=output_path)
        assert os.path.exists(output_path)


def test_plot_feature_importances_top_n():
    """Test feature importance plot with top_n parameter."""
    importances = {f"feature_{i}": 0.1 - i*0.01 for i in range(10)}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "importance_top5.png")
        plot_feature_importances(importances, output_path=output_path, top_n=5)
        assert os.path.exists(output_path)


def test_plot_portfolio_distribution_saves_file():
    """Test that portfolio distribution plot is saved to file."""
    df = generate_dataset(100, seed=42)
    model, _ = train_and_eval(df, seed=42, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)

    cfg = OptimizeConfig(
        total_budget=50_000_000,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 0.2, "h2_max_failure": 0.6, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=42
    )
    res = greedy_knapsack(analyzed, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "portfolio.png")
        plot_portfolio_distribution(res, output_path=output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_plot_portfolio_distribution_empty_portfolio():
    """Test portfolio plot handles empty portfolio gracefully."""
    res = {
        "selected_projects": 0,
        "total_budget_allocated": 0.0,
        "recommendations": [],
        "horizon_distribution": {}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "empty_portfolio.png")
        # Should handle empty portfolio without crashing
        plot_portfolio_distribution(res, output_path=output_path)
        # File should not be created for empty portfolio
        assert not os.path.exists(output_path)


def test_plot_risk_vs_return_saves_file():
    """Test that risk vs return plot is saved to file."""
    df = generate_dataset(50, seed=42)
    model, _ = train_and_eval(df, seed=42, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)

    cfg = OptimizeConfig(
        total_budget=30_000_000,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 0.2, "h2_max_failure": 0.6, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=42
    )
    res = greedy_knapsack(analyzed, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "risk_return.png")
        plot_risk_vs_return(res["recommendations"], output_path=output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_plot_risk_vs_return_empty_recommendations():
    """Test risk vs return plot handles empty recommendations gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "empty_risk_return.png")
        plot_risk_vs_return([], output_path=output_path)
        # Should not crash, file should not be created
        assert not os.path.exists(output_path)


def test_plot_directory_creation():
    """Test that plot functions create parent directories automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = os.path.join(tmpdir, "plots", "subdir", "calibration.png")

        cal_data = {
            "prob_true": [0.2, 0.5, 0.8],
            "prob_pred": [0.25, 0.5, 0.75],
            "bin_counts": [5, 10, 5],
            "n_bins": 3
        }

        plot_calibration_curve(cal_data, output_path=nested_path)
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))


def test_all_plots_together():
    """Integration test: generate all plots from a single workflow."""
    df = generate_dataset(150, seed=99)
    model, res = train_and_eval(df, seed=99, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)

    cfg = OptimizeConfig(
        total_budget=50_000_000,
        horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
        risk_limits={"h1_max_failure": 0.2, "h2_max_failure": 0.6, "h3_max_failure": 1.0},
        weights={"strategic": 0.7, "financial": 0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=99
    )
    opt_res = greedy_knapsack(analyzed, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate all plots
        plot_calibration_curve(res.calibration_curve, output_path=os.path.join(tmpdir, "calibration.png"))
        plot_feature_importances(res.feature_importances, output_path=os.path.join(tmpdir, "importance.png"))
        plot_portfolio_distribution(opt_res, output_path=os.path.join(tmpdir, "portfolio.png"))
        plot_risk_vs_return(opt_res["recommendations"], output_path=os.path.join(tmpdir, "risk_return.png"))

        # Verify all files created
        assert os.path.exists(os.path.join(tmpdir, "calibration.png"))
        assert os.path.exists(os.path.join(tmpdir, "importance.png"))
        assert os.path.exists(os.path.join(tmpdir, "portfolio.png"))
        assert os.path.exists(os.path.join(tmpdir, "risk_return.png"))
