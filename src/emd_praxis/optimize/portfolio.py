"""
Portfolio optimization under budget and policy constraints.
Implements a transparent greedy baseline suitable for academic reporting.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

@dataclass
class OptimizeConfig:
    total_budget: float
    horizon_pct: Dict[str, float]
    risk_limits: Dict[str, float]
    weights: Dict[str, float]  # {"strategic":0.7,"financial":0.3}
    scenarios_n: int = 100
    budget_volatility: float = 0.20
    seed: int = 42

def expected_pcv(row: pd.Series, *, rng: np.random.Generator, cfg: OptimizeConfig) -> float:
    # Simple, explainable proxy for expected post-competition value
    base_value = (row["target_trl"] - row["initial_trl"]) * 1_000_000.0
    strategic = float(row.get("market_potential", 2)) / 4.0  # 0..1
    success = float(row.get("predicted_success_probability", 0.5))
    # Random budget/mkt multipliers to mimic scenarios
    mults = rng.normal(1.0, cfg.budget_volatility, size=cfg.scenarios_n).clip(0.6, 1.4)
    expected = base_value * (cfg.weights["strategic"] * strategic + cfg.weights["financial"] * success) * mults.mean()
    return float(expected)

def greedy_knapsack(df: pd.DataFrame, cfg: OptimizeConfig) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    df = df.copy()
    df["expected_pcv"] = df.apply(lambda r: expected_pcv(r, rng=rng, cfg=cfg), axis=1)
    # Score = EV per cost with risk adjustment
    eps = 1e-9
    df["score"] = (0.7 * df["expected_pcv"] + 0.3 * df["predicted_success_probability"]) / (df["budget_musd"] * 1_000_000 + eps)

    # Sort
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Constraints tracking
    budget_used = 0.0
    sel = []
    horizon_spend = {"1":0.0,"2":0.0,"3":0.0}
    horizon_counts = {"1":0,"2":0,"3":0}
    horizon_fail = {"1":0,"2":0,"3":0}

    for _, r in df.iterrows():
        cost = float(r["budget_musd"]) * 1_000_000.0
        h = str(int(r["horizon"]))

        # Check budget
        if budget_used + cost > cfg.total_budget:
            continue
        # Horizon budget limit
        if horizon_spend[h] + cost > cfg.total_budget * cfg.horizon_pct[f"h{h}"]:
            continue
        # Risk limit by horizon
        if r["predicted_success_probability"] < 0.5:
            prospective_fail = horizon_fail[h] + 1
            prospective_cnt  = horizon_counts[h] + 1
            if (prospective_fail / max(1, prospective_cnt)) > cfg.risk_limits[f"h{h}_max_failure"]:
                continue

        # Accept
        sel.append(r)
        budget_used += cost
        horizon_spend[h] += cost
        horizon_counts[h] += 1
        if r["predicted_success_probability"] < 0.5:
            horizon_fail[h] += 1

    selected = pd.DataFrame(sel)
    result = {
        "selected_projects": int(len(selected)),
        "total_budget_allocated": float(budget_used),
        "budget_utilization": float(budget_used / cfg.total_budget) if cfg.total_budget else 0.0,
        "expected_portfolio_pcv": float(selected["expected_pcv"].sum()) if len(selected) else 0.0,
        "horizon_distribution": {
            "1": {"count": int((selected["horizon"]==1).sum()), "budget": float(selected.loc[selected["horizon"]==1,"budget_musd"].sum()*1_000_000)},
            "2": {"count": int((selected["horizon"]==2).sum()), "budget": float(selected.loc[selected["horizon"]==2,"budget_musd"].sum()*1_000_000)},
            "3": {"count": int((selected["horizon"]==3).sum()), "budget": float(selected.loc[selected["horizon"]==3,"budget_musd"].sum()*1_000_000)},
        },
        "recommendations": selected[["project_name","tech_area","horizon","budget_musd","predicted_success_probability","expected_pcv"]].to_dict(orient="records")
    }
    return result
