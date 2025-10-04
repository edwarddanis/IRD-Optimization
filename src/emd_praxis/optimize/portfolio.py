"""
Portfolio optimization under budget and policy constraints.
Implements a transparent greedy baseline suitable for academic reporting.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Portfolio optimization constants
TRL_VALUE_PER_LEVEL = 1_000_000.0  # Base value per TRL advancement ($)
EPSILON = 1e-9  # Small constant to prevent division by zero
MARKET_POTENTIAL_MAX = 4.0  # Maximum market potential rating
BUDGET_MULTIPLIER_MIN = 0.6  # Minimum budget scenario multiplier
BUDGET_MULTIPLIER_MAX = 1.4  # Maximum budget scenario multiplier

@dataclass
class OptimizeConfig:
    total_budget: float
    horizon_pct: Dict[str, float]
    risk_limits: Dict[str, float]
    weights: Dict[str, float]  # {"strategic":0.7,"financial":0.3}
    scenarios_n: int = 100
    budget_volatility: float = 0.20
    seed: int = 42
    # Scoring weights for portfolio optimization
    score_weights: Dict[str, float] = None  # {"pcv": 0.7, "success_prob": 0.3}
    # Threshold for considering a project as high-risk/likely to fail
    failure_threshold: float = 0.5

    def __post_init__(self):
        """Set default score weights if not provided."""
        if self.score_weights is None:
            self.score_weights = {"pcv": 0.7, "success_prob": 0.3}

def expected_pcv(row: pd.Series, *, rng: np.random.Generator, cfg: OptimizeConfig) -> float:
    """
    Calculate expected post-competition value (PCV) for an IR&D project.

    This function implements a stochastic valuation model that combines:
    1. Technical advancement value (TRL progression)
    2. Strategic value (market potential)
    3. Financial value (ML-predicted success probability)
    4. Uncertainty modeling (budget/market volatility via Monte Carlo scenarios)

    The model uses Monte Carlo simulation to account for budget and market
    uncertainties by sampling multiple scenarios and averaging the results.

    Parameters
    ----------
    row : pd.Series
        Project attributes including:
        - target_trl : int (1-9)
            Target Technology Readiness Level
        - initial_trl : int (1-9)
            Starting Technology Readiness Level
        - market_potential : int (1-4)
            Strategic importance rating (defaults to 2 if missing)
        - predicted_success_probability : float (0-1)
            ML-predicted success rate (defaults to 0.5 if missing)
    rng : np.random.Generator
        Random number generator for reproducible scenario sampling.
        Use np.random.default_rng(seed) for reproducibility.
    cfg : OptimizeConfig
        Configuration containing:
        - weights : Dict[str, float]
            Strategic vs financial weights (e.g., {"strategic": 0.7, "financial": 0.3})
        - scenarios_n : int
            Number of Monte Carlo scenarios to simulate
        - budget_volatility : float
            Standard deviation for budget multiplier scenarios

    Returns
    -------
    float
        Expected post-competition value in dollars. Higher values indicate
        projects with greater expected strategic and financial returns.

    Notes
    -----
    The base value assumes $1M per TRL level advancement. This is a simplified
    proxy for academic demonstration purposes. Real applications should calibrate
    against historical program data and organizational cost models.

    Budget multipliers are clipped to [0.6, 1.4] to represent realistic
    budget uncertainty ranges (±40% variation).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> cfg = OptimizeConfig(
    ...     total_budget=10e6,
    ...     horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
    ...     risk_limits={"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
    ...     weights={"strategic": 0.7, "financial": 0.3}
    ... )
    >>> project = pd.Series({
    ...     "initial_trl": 3,
    ...     "target_trl": 6,
    ...     "market_potential": 3,
    ...     "predicted_success_probability": 0.75
    ... })
    >>> pcv = expected_pcv(project, rng=rng, cfg=cfg)
    >>> print(f"Expected PCV: ${pcv:,.0f}")
    Expected PCV: $2,737,500
    """
    # Base value: $1M per TRL advancement
    base_value = (row["target_trl"] - row["initial_trl"]) * TRL_VALUE_PER_LEVEL

    # Strategic component: market potential normalized to [0, 1]
    strategic = float(row.get("market_potential", 2)) / MARKET_POTENTIAL_MAX

    # Financial component: ML-predicted success probability
    success = float(row.get("predicted_success_probability", 0.5))

    # Monte Carlo simulation: sample budget/market multipliers
    mults = rng.normal(1.0, cfg.budget_volatility, size=cfg.scenarios_n).clip(
        BUDGET_MULTIPLIER_MIN, BUDGET_MULTIPLIER_MAX
    )

    # Expected value: weighted combination with scenario averaging
    expected = base_value * (
        cfg.weights["strategic"] * strategic +
        cfg.weights["financial"] * success
    ) * mults.mean()

    return float(expected)

def greedy_knapsack(df: pd.DataFrame, cfg: OptimizeConfig) -> Dict[str, float]:
    """
    Optimize IR&D portfolio selection using a greedy knapsack algorithm.

    This algorithm implements a transparent, explainable approach to portfolio
    optimization suitable for academic research and stakeholder communication.
    Projects are ranked by value-per-cost and selected sequentially while
    respecting budget, horizon allocation, and risk constraints.

    Algorithm Overview
    ------------------
    1. Calculate expected post-competition value (PCV) for each project
    2. Compute efficiency score: (weighted PCV + success prob) / cost
    3. Sort projects by efficiency score (descending)
    4. Greedily select projects that satisfy all constraints:
       - Total budget constraint
       - Per-horizon budget allocation limits
       - Per-horizon risk/failure rate limits

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed project dataset containing:
        - project_name : str
        - tech_area : str
        - horizon : int (1, 2, or 3)
        - budget_musd : float
        - predicted_success_probability : float (0-1)
        - initial_trl, target_trl : int (1-9)
        - market_potential : int (1-4)
    cfg : OptimizeConfig
        Optimization configuration including:
        - total_budget : float
            Total available budget in dollars
        - horizon_pct : Dict[str, float]
            Budget allocation percentages per horizon (e.g., {"h1": 0.4, "h2": 0.5, "h3": 0.1})
        - risk_limits : Dict[str, float]
            Maximum failure rate per horizon (e.g., {"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0})
        - weights : Dict[str, float]
            Strategic vs financial weights for PCV calculation
        - score_weights : Dict[str, float]
            Weights for efficiency scoring (PCV vs success probability)
        - failure_threshold : float
            Success probability threshold below which a project is considered high-risk
        - scenarios_n : int
            Number of Monte Carlo scenarios for PCV estimation
        - seed : int
            Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Optimization results containing:
        - selected_projects : int
            Number of projects selected
        - total_budget_allocated : float
            Total budget allocated in dollars
        - budget_utilization : float
            Fraction of total budget used (0-1)
        - expected_portfolio_pcv : float
            Sum of expected PCV across selected projects
        - horizon_distribution : Dict[str, Dict[str, int|float]]
            Projects and budget allocated per horizon
        - recommendations : List[Dict]
            Detailed list of selected projects with attributes

    Notes
    -----
    Greedy algorithms do not guarantee global optimality but provide:
    - Transparency: Easy to explain to stakeholders
    - Speed: O(n log n) complexity from sorting
    - Reasonableness: Good approximation for most portfolio problems

    For true global optimality, consider integer programming approaches
    (e.g., Gurobi, CPLEX), though they sacrifice explainability.

    The algorithm enforces hard constraints. If constraints are too restrictive,
    the portfolio may be underfunded. Consider relaxing risk limits or horizon
    allocations if budget utilization is consistently low.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("data/projects_analyzed.csv")
    >>> cfg = OptimizeConfig(
    ...     total_budget=10_000_000,
    ...     horizon_pct={"h1": 0.4, "h2": 0.5, "h3": 0.1},
    ...     risk_limits={"h1_max_failure": 0.1, "h2_max_failure": 0.5, "h3_max_failure": 1.0},
    ...     weights={"strategic": 0.7, "financial": 0.3},
    ...     seed=42
    ... )
    >>> result = greedy_knapsack(df, cfg)
    >>> print(f"Selected {result['selected_projects']} projects")
    >>> print(f"Budget utilization: {result['budget_utilization']:.1%}")
    >>> print(f"Expected portfolio value: ${result['expected_portfolio_pcv']:,.0f}")

    See Also
    --------
    expected_pcv : Function for calculating project post-competition value
    OptimizeConfig : Configuration dataclass for portfolio optimization
    """
    rng = np.random.default_rng(cfg.seed)
    df = df.copy()

    # Step 1: Calculate expected PCV for each project
    df["expected_pcv"] = df.apply(lambda r: expected_pcv(r, rng=rng, cfg=cfg), axis=1)

    # Step 2: Calculate efficiency score (value per dollar)
    df["score"] = (
        cfg.score_weights["pcv"] * df["expected_pcv"] +
        cfg.score_weights["success_prob"] * df["predicted_success_probability"]
    ) / (df["budget_musd"] * 1_000_000 + EPSILON)

    # Step 3: Sort by efficiency (highest first)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Step 4: Greedy selection with constraint checking
    budget_used = 0.0
    sel = []
    horizon_spend = {"1": 0.0, "2": 0.0, "3": 0.0}
    horizon_counts = {"1": 0, "2": 0, "3": 0}
    horizon_fail = {"1": 0, "2": 0, "3": 0}

    for _, r in df.iterrows():
        cost = float(r["budget_musd"]) * 1_000_000.0
        h = str(int(r["horizon"]))

        # Constraint 1: Total budget
        if budget_used + cost > cfg.total_budget:
            continue

        # Constraint 2: Horizon budget allocation limit
        if horizon_spend[h] + cost > cfg.total_budget * cfg.horizon_pct[f"h{h}"]:
            continue

        # Constraint 3: Risk/failure rate limit by horizon
        if r["predicted_success_probability"] < cfg.failure_threshold:
            prospective_fail = horizon_fail[h] + 1
            prospective_cnt = horizon_counts[h] + 1
            if (prospective_fail / max(1, prospective_cnt)) > cfg.risk_limits[f"h{h}_max_failure"]:
                continue

        # All constraints satisfied - accept project
        sel.append(r)
        budget_used += cost
        horizon_spend[h] += cost
        horizon_counts[h] += 1
        if r["predicted_success_probability"] < cfg.failure_threshold:
            horizon_fail[h] += 1

    # Step 5: Compile results
    selected = pd.DataFrame(sel)

    # Handle empty selection case
    if len(selected) == 0:
        result = {
            "selected_projects": 0,
            "total_budget_allocated": 0.0,
            "budget_utilization": 0.0,
            "expected_portfolio_pcv": 0.0,
            "horizon_distribution": {
                "1": {"count": 0, "budget": 0.0},
                "2": {"count": 0, "budget": 0.0},
                "3": {"count": 0, "budget": 0.0},
            },
            "recommendations": []
        }
    else:
        result = {
            "selected_projects": int(len(selected)),
            "total_budget_allocated": float(budget_used),
            "budget_utilization": float(budget_used / cfg.total_budget) if cfg.total_budget else 0.0,
            "expected_portfolio_pcv": float(selected["expected_pcv"].sum()),
            "horizon_distribution": {
                "1": {
                    "count": int((selected["horizon"] == 1).sum()),
                    "budget": float(selected.loc[selected["horizon"] == 1, "budget_musd"].sum() * 1_000_000)
                },
                "2": {
                    "count": int((selected["horizon"] == 2).sum()),
                    "budget": float(selected.loc[selected["horizon"] == 2, "budget_musd"].sum() * 1_000_000)
                },
                "3": {
                    "count": int((selected["horizon"] == 3).sum()),
                    "budget": float(selected.loc[selected["horizon"] == 3, "budget_musd"].sum() * 1_000_000)
                },
            },
            "recommendations": selected[[
                "project_name", "tech_area", "horizon", "budget_musd",
                "predicted_success_probability", "expected_pcv"
            ]].to_dict(orient="records")
        }
    return result
