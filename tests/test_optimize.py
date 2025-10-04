import pandas as pd
import pytest
from emd_praxis.data.synthetic_generator import generate_dataset
from emd_praxis.ml.model import train_and_eval, batch_annotate
from emd_praxis.optimize.portfolio import OptimizeConfig, greedy_knapsack


def test_optimizer_runs():
    df = generate_dataset(300, seed=1)
    model, _ = train_and_eval(df, seed=1, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)
    cfg = OptimizeConfig(
        total_budget=10_000_000,
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.2,"h2_max_failure":0.6,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=1
    )
    res = greedy_knapsack(analyzed, cfg)
    assert "selected_projects" in res and res["total_budget_allocated"] <= cfg.total_budget


def test_optimizer_respects_budget_constraint():
    """Verify total allocated budget never exceeds total_budget."""
    df = generate_dataset(100, seed=2)
    model, _ = train_and_eval(df, seed=2, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)
    cfg = OptimizeConfig(
        total_budget=5_000_000,
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.2,"h2_max_failure":0.6,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=2
    )
    res = greedy_knapsack(analyzed, cfg)
    assert res["total_budget_allocated"] <= cfg.total_budget


def test_optimizer_respects_horizon_allocation():
    """Verify horizon budget allocation respects configured percentages."""
    df = generate_dataset(200, seed=3)
    model, _ = train_and_eval(df, seed=3, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)
    cfg = OptimizeConfig(
        total_budget=20_000_000,
        horizon_pct={"h1":0.5,"h2":0.3,"h3":0.2},
        risk_limits={"h1_max_failure":0.1,"h2_max_failure":0.5,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=3
    )
    res = greedy_knapsack(analyzed, cfg)

    # Check each horizon doesn't exceed its allocation
    h1_budget = res["horizon_distribution"]["1"]["budget"]
    h2_budget = res["horizon_distribution"]["2"]["budget"]
    h3_budget = res["horizon_distribution"]["3"]["budget"]

    assert h1_budget <= cfg.total_budget * cfg.horizon_pct["h1"]
    assert h2_budget <= cfg.total_budget * cfg.horizon_pct["h2"]
    assert h3_budget <= cfg.total_budget * cfg.horizon_pct["h3"]


def test_optimizer_empty_dataset():
    """Verify optimizer handles empty dataset gracefully."""
    df = pd.DataFrame({
        "project_name": [],
        "tech_area": [],
        "horizon": [],
        "budget_musd": [],
        "predicted_success_probability": [],
        "initial_trl": [],
        "target_trl": [],
        "market_potential": []
    })
    cfg = OptimizeConfig(
        total_budget=10_000_000,
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.2,"h2_max_failure":0.6,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=4
    )
    res = greedy_knapsack(df, cfg)
    assert res["selected_projects"] == 0
    assert res["total_budget_allocated"] == 0.0


def test_optimizer_zero_budget():
    """Verify optimizer handles zero budget constraint."""
    df = generate_dataset(50, seed=5)
    model, _ = train_and_eval(df, seed=5, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)
    cfg = OptimizeConfig(
        total_budget=0,
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.2,"h2_max_failure":0.6,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=5
    )
    res = greedy_knapsack(analyzed, cfg)
    assert res["selected_projects"] == 0
    assert res["total_budget_allocated"] == 0.0


def test_optimizer_strict_risk_limits():
    """Verify optimizer respects strict risk limits."""
    df = generate_dataset(100, seed=6)
    model, _ = train_and_eval(df, seed=6, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)
    cfg = OptimizeConfig(
        total_budget=50_000_000,
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.0,"h2_max_failure":0.0,"h3_max_failure":0.0},  # No failures allowed
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=6
    )
    res = greedy_knapsack(analyzed, cfg)

    # All selected projects must have success probability >= 0.5
    for rec in res["recommendations"]:
        assert rec["predicted_success_probability"] >= 0.5


def test_optimizer_identical_scores():
    """Verify deterministic behavior when projects have identical scores."""
    # Create projects with identical attributes (same score)
    df = pd.DataFrame({
        "project_name": [f"Project_{i}" for i in range(5)],
        "tech_area": ["AI/ML"] * 5,
        "horizon": [1] * 5,
        "budget_musd": [5.0] * 5,
        "predicted_success_probability": [0.7] * 5,
        "initial_trl": [3] * 5,
        "target_trl": [6] * 5,
        "market_potential": [2] * 5,
    })
    cfg = OptimizeConfig(
        total_budget=20_000_000,
        horizon_pct={"h1":0.8,"h2":0.15,"h3":0.05},
        risk_limits={"h1_max_failure":0.5,"h2_max_failure":0.5,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=42
    )

    # Run twice with same seed - should get identical results
    res1 = greedy_knapsack(df, cfg)
    res2 = greedy_knapsack(df, cfg)

    assert res1["selected_projects"] == res2["selected_projects"]
    assert res1["total_budget_allocated"] == res2["total_budget_allocated"]
    assert len(res1["recommendations"]) == len(res2["recommendations"])


def test_optimizer_large_dataset():
    """Stress test with large dataset."""
    df = generate_dataset(2000, seed=7)
    model, _ = train_and_eval(df, seed=7, n_estimators=20, learning_rate=0.1, max_depth=2, cv_folds=3)
    analyzed = batch_annotate(df, model)
    cfg = OptimizeConfig(
        total_budget=100_000_000,
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.2,"h2_max_failure":0.6,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=7
    )
    res = greedy_knapsack(analyzed, cfg)

    # Should complete without error
    assert res["selected_projects"] >= 0
    assert res["total_budget_allocated"] <= cfg.total_budget


def test_optimizer_all_projects_fail_risk_limits():
    """Test when all projects fail risk limit constraints."""
    # Create projects with very low success probability
    df = pd.DataFrame({
        "project_name": [f"Risky_Project_{i}" for i in range(10)],
        "tech_area": ["Hypersonics"] * 10,
        "horizon": [1] * 10,
        "budget_musd": [2.0] * 10,
        "predicted_success_probability": [0.1] * 10,  # All very risky
        "initial_trl": [2] * 10,
        "target_trl": [5] * 10,
        "market_potential": [3] * 10,
    })
    cfg = OptimizeConfig(
        total_budget=50_000_000,
        horizon_pct={"h1":1.0,"h2":0.0,"h3":0.0},
        risk_limits={"h1_max_failure":0.05,"h2_max_failure":0.5,"h3_max_failure":1.0},  # Very strict for h1
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=8
    )
    res = greedy_knapsack(df, cfg)

    # Should select very few or no projects due to strict risk limit
    assert res["selected_projects"] <= 1  # At most 1 project (5% of 1 = 0.05)


def test_optimizer_single_project():
    """Test with exactly one project."""
    df = pd.DataFrame({
        "project_name": ["Solo_Project"],
        "tech_area": ["Quantum Science"],
        "horizon": [2],
        "budget_musd": [3.0],
        "predicted_success_probability": [0.8],
        "initial_trl": [4],
        "target_trl": [7],
        "market_potential": [3],
    })
    cfg = OptimizeConfig(
        total_budget=10_000_000,
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.2,"h2_max_failure":0.6,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=9
    )
    res = greedy_knapsack(df, cfg)

    # Should select the single project
    assert res["selected_projects"] == 1
    assert res["recommendations"][0]["project_name"] == "Solo_Project"


def test_optimizer_all_projects_too_expensive():
    """Test when all projects exceed total budget."""
    df = pd.DataFrame({
        "project_name": [f"Expensive_Project_{i}" for i in range(5)],
        "tech_area": ["Space Technology"] * 5,
        "horizon": [2] * 5,
        "budget_musd": [20.0] * 5,  # Each project costs $20M
        "predicted_success_probability": [0.8] * 5,
        "initial_trl": [3] * 5,
        "target_trl": [6] * 5,
        "market_potential": [4] * 5,
    })
    cfg = OptimizeConfig(
        total_budget=10_000_000,  # Only $10M available
        horizon_pct={"h1":0.4,"h2":0.5,"h3":0.1},
        risk_limits={"h1_max_failure":0.2,"h2_max_failure":0.6,"h3_max_failure":1.0},
        weights={"strategic":0.7,"financial":0.3},
        scenarios_n=20,
        budget_volatility=0.2,
        seed=10
    )
    res = greedy_knapsack(df, cfg)

    # Should select no projects
    assert res["selected_projects"] == 0
    assert res["total_budget_allocated"] == 0.0
