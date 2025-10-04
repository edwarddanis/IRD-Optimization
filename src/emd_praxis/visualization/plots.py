"""
Plotting functions for model evaluation and portfolio visualization.
"""
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_calibration_curve(
    calibration_data: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "Model Calibration Curve",
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot calibration curve showing predicted vs actual success probabilities.

    A well-calibrated model has points close to the diagonal (perfect calibration).
    Points above the diagonal indicate overconfidence; points below indicate underconfidence.

    Parameters
    ----------
    calibration_data : Dict[str, Any]
        Output from generate_calibration_data() containing:
        - prob_true: List of actual success frequencies per bin
        - prob_pred: List of mean predicted probabilities per bin
        - bin_counts: Number of samples in each bin
        - n_bins: Number of bins used
    output_path : Optional[str]
        File path to save plot (e.g., "plots/calibration.png").
        If None, displays plot interactively.
    title : str, optional
        Plot title (default: "Model Calibration Curve")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (8, 6))

    Examples
    --------
    >>> from emd_praxis.ml.model import train_and_eval
    >>> from emd_praxis.visualization import plot_calibration_curve
    >>> model, res = train_and_eval(df)
    >>> plot_calibration_curve(res.calibration_curve, output_path="calibration.png")
    """
    prob_pred = np.array(calibration_data["prob_pred"])
    prob_true = np.array(calibration_data["prob_true"])
    bin_counts = calibration_data.get("bin_counts", [])

    fig, ax = plt.subplots(figsize=figsize)

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', label='Model', markersize=8, linewidth=2)

    # Plot perfect calibration reference
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5, alpha=0.7)

    # Add bin count annotations if available
    if bin_counts:
        for x, y, count in zip(prob_pred, prob_true, bin_counts):
            if count > 0:
                ax.annotate(f'n={count}', (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8, alpha=0.7)

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importances(
    feature_importances: Dict[str, float],
    output_path: Optional[str] = None,
    title: str = "Feature Importances",
    figsize: tuple = (10, 6),
    top_n: Optional[int] = None
) -> None:
    """
    Plot horizontal bar chart of feature importances from trained model.

    Parameters
    ----------
    feature_importances : Dict[str, float]
        Dictionary mapping feature names to importance scores.
        Typically from MLResult.feature_importances.
    output_path : Optional[str]
        File path to save plot (e.g., "plots/feature_importance.png").
        If None, displays plot interactively.
    title : str, optional
        Plot title (default: "Feature Importances")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 6))
    top_n : Optional[int]
        Show only top N most important features. If None, shows all.

    Examples
    --------
    >>> from emd_praxis.ml.model import train_and_eval
    >>> from emd_praxis.visualization import plot_feature_importances
    >>> model, res = train_and_eval(df)
    >>> plot_feature_importances(res.feature_importances, output_path="importance.png")
    """
    # Sort by importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_features = sorted_features[:top_n]

    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(features))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))

    bars = ax.barh(y_pos, importances, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importances)):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()  # Highest importance on top
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_portfolio_distribution(
    optimization_result: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Create comprehensive portfolio visualization with multiple subplots:
    - Budget allocation by horizon
    - Project count by horizon
    - Technology area distribution
    - Risk level distribution

    Parameters
    ----------
    optimization_result : Dict[str, Any]
        Output from greedy_knapsack() containing:
        - recommendations: List of selected projects
        - horizon_distribution: Budget/count by horizon
        - total_budget_allocated: Total budget used
    output_path : Optional[str]
        File path to save plot (e.g., "plots/portfolio.png").
        If None, displays plot interactively.
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (12, 8))

    Examples
    --------
    >>> from emd_praxis.optimize.portfolio import greedy_knapsack
    >>> from emd_praxis.visualization import plot_portfolio_distribution
    >>> result = greedy_knapsack(analyzed_df, config)
    >>> plot_portfolio_distribution(result, output_path="portfolio.png")
    """
    recommendations = optimization_result.get("recommendations", [])
    horizon_dist = optimization_result.get("horizon_distribution", {})

    if not recommendations:
        print("No projects selected - cannot create visualization")
        return

    df = pd.DataFrame(recommendations)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Budget by Horizon
    ax1 = fig.add_subplot(gs[0, 0])
    horizons = sorted([int(h) for h in horizon_dist.keys()])
    budgets = [horizon_dist[str(h)]["budget"] / 1_000_000 for h in horizons]
    colors_horizon = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(horizons)]

    bars1 = ax1.bar(horizons, budgets, color=colors_horizon, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Horizon', fontsize=11)
    ax1.set_ylabel('Budget Allocated ($M)', fontsize=11)
    ax1.set_title('Budget Distribution by Horizon', fontsize=12, fontweight='bold')
    ax1.set_xticks(horizons)
    ax1.set_xticklabels([f'H{h}' for h in horizons])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, budgets):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.1f}M', ha='center', va='bottom', fontsize=9)

    # 2. Project Count by Horizon
    ax2 = fig.add_subplot(gs[0, 1])
    counts = [horizon_dist[str(h)]["count"] for h in horizons]

    bars2 = ax2.bar(horizons, counts, color=colors_horizon, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Horizon', fontsize=11)
    ax2.set_ylabel('Number of Projects', fontsize=11)
    ax2.set_title('Project Count by Horizon', fontsize=12, fontweight='bold')
    ax2.set_xticks(horizons)
    ax2.set_xticklabels([f'H{h}' for h in horizons])
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=9)

    # 3. Technology Area Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    tech_counts = df['tech_area'].value_counts()
    colors_tech = plt.cm.Set3(np.linspace(0, 1, len(tech_counts)))

    wedges, texts, autotexts = ax3.pie(
        tech_counts.values,
        labels=tech_counts.index,
        autopct='%1.1f%%',
        colors=colors_tech,
        startangle=90,
        textprops={'fontsize': 9}
    )
    ax3.set_title('Technology Area Distribution', fontsize=12, fontweight='bold')

    # 4. Risk Level Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        risk_order = ['LOW', 'MEDIUM', 'HIGH']
        risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index])
        colors_risk = {'LOW': '#2ca02c', 'MEDIUM': '#ff7f0e', 'HIGH': '#d62728'}
        bar_colors = [colors_risk.get(r, '#gray') for r in risk_counts.index]

        bars4 = ax4.bar(range(len(risk_counts)), risk_counts.values,
                       color=bar_colors, edgecolor='black', linewidth=1)
        ax4.set_xlabel('Risk Level', fontsize=11)
        ax4.set_ylabel('Number of Projects', fontsize=11)
        ax4.set_title('Risk Level Distribution', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(risk_counts)))
        ax4.set_xticklabels(risk_counts.index)
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars4, risk_counts.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Risk level data not available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.set_xticks([])
        ax4.set_yticks([])

    # Overall title
    total_budget = optimization_result.get("total_budget_allocated", 0) / 1_000_000
    total_projects = optimization_result.get("selected_projects", 0)
    fig.suptitle(f'Portfolio Overview: {total_projects} Projects, ${total_budget:.1f}M Allocated',
                fontsize=14, fontweight='bold', y=0.98)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_risk_vs_return(
    recommendations: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Portfolio Projects: Risk vs Return",
    figsize: tuple = (10, 7)
) -> None:
    """
    Scatter plot showing success probability vs expected value for portfolio projects.

    Bubble size represents project budget. Color indicates horizon.

    Parameters
    ----------
    recommendations : List[Dict[str, Any]]
        List of selected projects from optimization result.
        Each project dict should contain: predicted_success_probability,
        budget_musd, horizon, project_name.
    output_path : Optional[str]
        File path to save plot (e.g., "plots/risk_return.png").
        If None, displays plot interactively.
    title : str, optional
        Plot title (default: "Portfolio Projects: Risk vs Return")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 7))

    Examples
    --------
    >>> from emd_praxis.visualization import plot_risk_vs_return
    >>> plot_risk_vs_return(result["recommendations"], output_path="risk_return.png")
    """
    if not recommendations:
        print("No projects to visualize")
        return

    df = pd.DataFrame(recommendations)

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate expected value proxy (success_prob * market_potential * trl_advancement)
    if 'market_potential' in df.columns and 'target_trl' in df.columns and 'initial_trl' in df.columns:
        df['expected_value'] = (df['predicted_success_probability'] *
                               df['market_potential'] *
                               (df['target_trl'] - df['initial_trl']))
    else:
        # Fallback: use success probability as proxy
        df['expected_value'] = df['predicted_success_probability']

    # Plot by horizon with different colors
    horizons = sorted(df['horizon'].unique())
    colors_map = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}

    for h in horizons:
        subset = df[df['horizon'] == h]
        scatter = ax.scatter(
            subset['predicted_success_probability'],
            subset['expected_value'],
            s=subset['budget_musd'] * 50,  # Bubble size proportional to budget
            c=colors_map.get(h, '#gray'),
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
            label=f'Horizon {h}'
        )

    ax.set_xlabel('Success Probability', fontsize=12)
    ax.set_ylabel('Expected Value (proxy)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)

    # Add quadrant lines
    ax.axhline(y=df['expected_value'].median(), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
