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
