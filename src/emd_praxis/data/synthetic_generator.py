"""
Synthetic data generator for IR&D portfolio optimization testing.

Generates realistic IR&D project datasets with:
- Project metadata (name, tech area, horizon)
- Budget and resource constraints
- Technical readiness levels (TRL)
- Market potential and stakeholder support
- Success outcomes based on realistic correlations
"""
import numpy as np
import pandas as pd
from typing import Optional

# Tech areas typical in defense R&D
TECH_AREAS = [
    "AI/ML",
    "Autonomy",
    "Cyber Security",
    "Directed Energy",
    "Hypersonics",
    "Quantum Science",
    "Space Technology",
    "Advanced Materials",
    "Biotechnology",
    "Communications",
]

# Time horizons (years to impact)
HORIZONS = [1, 2, 3]

# Budget ranges in millions USD
BUDGET_MIN = 0.5
BUDGET_MAX = 12.0

# Duration in months
DURATION_MIN = 6
DURATION_MAX = 48

# Team size ranges
TEAM_MIN = 2
TEAM_MAX = 25

# Complexity and novelty scales (1-5)
COMPLEXITY_MIN = 1
COMPLEXITY_MAX = 5
NOVELTY_MIN = 1
NOVELTY_MAX = 5

# Stakeholder support and market potential (1-5)
SUPPORT_MIN = 1
SUPPORT_MAX = 5
MARKET_MIN = 1
MARKET_MAX = 5

# TRL ranges
TRL_MIN = 1
TRL_MAX = 9


def generate_dataset(n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic IR&D project dataset for testing and development.

    Creates realistic project data with correlated features that influence success
    probability. Success outcomes are generated based on weighted combinations of:
    - Higher budget increases success probability
    - Higher stakeholder support increases success
    - Higher complexity and novelty decrease success
    - Higher TRL advancement (target - initial) increases difficulty
    - Longer duration projects have more uncertainty

    Parameters
    ----------
    n_samples : int
        Number of projects to generate. Must be positive.
    seed : int, optional
        Random seed for reproducibility. If None, uses random initialization.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - project_name : str
            Unique project identifier
        - tech_area : str
            Technology domain (e.g., "AI/ML", "Hypersonics")
        - horizon : int
            Time horizon category (1=near-term, 2=mid-term, 3=far-term)
        - budget_musd : float
            Budget in millions USD (0.5-12.0)
        - duration_months : int
            Project duration in months (6-48)
        - team_size : int
            Number of team members (2-25)
        - complexity : int
            Technical complexity rating (1-5)
        - novelty : int
            Innovation novelty rating (1-5)
        - stakeholder_support : int
            Stakeholder support level (1-5)
        - market_potential : int
            Market/mission impact potential (1-5)
        - initial_trl : int
            Starting technology readiness level (1-9)
        - target_trl : int
            Target technology readiness level (initial_trl to 9)
        - success : int
            Binary outcome (0=failure, 1=success)

    Examples
    --------
    >>> from emd_praxis.data.synthetic_generator import generate_dataset
    >>> df = generate_dataset(100, seed=42)
    >>> print(f"Generated {len(df)} projects")
    >>> print(f"Success rate: {df['success'].mean():.2%}")

    >>> # Use in ML pipeline
    >>> from emd_praxis.ml.model import train_and_eval
    >>> model, results = train_and_eval(df, seed=42)

    Notes
    -----
    The generated data is synthetic and intended for testing, development,
    and demonstration purposes only. Real-world project data will have
    different distributions and correlations.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.RandomState(seed)

    # Generate base features
    data = {
        "project_name": [f"Project_{i:04d}" for i in range(n_samples)],
        "tech_area": rng.choice(TECH_AREAS, size=n_samples),
        "horizon": rng.choice(HORIZONS, size=n_samples),
        "budget_musd": rng.uniform(BUDGET_MIN, BUDGET_MAX, size=n_samples),
        "duration_months": rng.randint(DURATION_MIN, DURATION_MAX + 1, size=n_samples),
        "team_size": rng.randint(TEAM_MIN, TEAM_MAX + 1, size=n_samples),
        "complexity": rng.randint(COMPLEXITY_MIN, COMPLEXITY_MAX + 1, size=n_samples),
        "novelty": rng.randint(NOVELTY_MIN, NOVELTY_MAX + 1, size=n_samples),
        "stakeholder_support": rng.randint(SUPPORT_MIN, SUPPORT_MAX + 1, size=n_samples),
        "market_potential": rng.randint(MARKET_MIN, MARKET_MAX + 1, size=n_samples),
    }

    # Generate TRL progression (initial < target)
    initial_trl = rng.randint(TRL_MIN, TRL_MAX, size=n_samples)
    # Ensure target TRL is at least initial TRL, up to max
    target_trl = np.array([
        rng.randint(init_trl, TRL_MAX + 1) if init_trl < TRL_MAX else TRL_MAX
        for init_trl in initial_trl
    ])
    data["initial_trl"] = initial_trl
    data["target_trl"] = target_trl

    # Generate success outcomes based on feature correlations
    # Create a success probability for each project
    success_prob = _calculate_success_probability(
        budget=data["budget_musd"],
        complexity=data["complexity"],
        novelty=data["novelty"],
        stakeholder_support=data["stakeholder_support"],
        trl_advancement=target_trl - initial_trl,
        duration=data["duration_months"],
        rng=rng
    )

    # Sample binary outcomes
    data["success"] = (rng.random(n_samples) < success_prob).astype(int)

    df = pd.DataFrame(data)

    # Round numeric columns appropriately
    df["budget_musd"] = df["budget_musd"].round(2)

    return df


def _calculate_success_probability(
    budget: np.ndarray,
    complexity: np.ndarray,
    novelty: np.ndarray,
    stakeholder_support: np.ndarray,
    trl_advancement: np.ndarray,
    duration: np.ndarray,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Calculate project success probability based on features.

    Uses a weighted combination of factors with realistic correlations:
    - Higher budget and support increase success
    - Higher complexity, novelty, and TRL advancement increase risk
    - Adds realistic noise to prevent perfect prediction

    Parameters
    ----------
    budget : np.ndarray
        Project budgets in millions USD
    complexity : np.ndarray
        Complexity ratings (1-5)
    novelty : np.ndarray
        Novelty ratings (1-5)
    stakeholder_support : np.ndarray
        Support levels (1-5)
    trl_advancement : np.ndarray
        TRL progression distance (target - initial)
    duration : np.ndarray
        Project duration in months
    rng : np.random.RandomState
        Random number generator for noise

    Returns
    -------
    np.ndarray
        Success probabilities for each project (0-1 range)
    """
    # Normalize features to 0-1 range
    budget_norm = (budget - BUDGET_MIN) / (BUDGET_MAX - BUDGET_MIN)
    complexity_norm = (complexity - COMPLEXITY_MIN) / (COMPLEXITY_MAX - COMPLEXITY_MIN)
    novelty_norm = (novelty - NOVELTY_MIN) / (NOVELTY_MAX - NOVELTY_MIN)
    support_norm = (stakeholder_support - SUPPORT_MIN) / (SUPPORT_MAX - SUPPORT_MIN)
    trl_norm = trl_advancement / (TRL_MAX - TRL_MIN)
    duration_norm = (duration - DURATION_MIN) / (DURATION_MAX - DURATION_MIN)

    # Weighted combination (positive factors - negative factors)
    # Positive: budget, support
    # Negative: complexity, novelty, TRL advancement, long duration
    score = (
        0.3 * budget_norm +
        0.3 * support_norm -
        0.2 * complexity_norm -
        0.15 * novelty_norm -
        0.1 * trl_norm -
        0.05 * duration_norm
    )

    # Convert to probability using sigmoid-like transformation
    # Add baseline to ensure reasonable success rates (not too extreme)
    prob = 0.5 + 0.4 * score

    # Add noise for realism (projects have inherent uncertainty)
    noise = rng.normal(0, 0.1, size=len(prob))
    prob = prob + noise

    # Clip to valid probability range
    prob = np.clip(prob, 0.05, 0.95)

    return prob
