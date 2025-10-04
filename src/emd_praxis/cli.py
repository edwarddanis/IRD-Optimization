"""
Command-line interface for the EMD Praxis research codebase.
"""
from __future__ import annotations
import argparse, json, os, sys, logging
import yaml
import pandas as pd
from pathlib import Path

from emd_praxis.data.synthetic_generator import generate_dataset, save_csv
from emd_praxis.ml.model import train_and_eval, batch_annotate, save_model, load_model
from emd_praxis.optimize.portfolio import OptimizeConfig, greedy_knapsack
from emd_praxis.validation import (
    validate_generate_params, validate_ml_params, validate_optimize_config,
    validate_analyzed_dataframe, validate_dataframe_columns
)
from emd_praxis.visualization import (
    plot_calibration_curve, plot_feature_importances,
    plot_portfolio_distribution, plot_risk_vs_return
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def _ensure_dirs(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        logger.debug(f"Created directory: {d}")

def cmd_generate(args):
    logger.info(f"Generating dataset: n={args.n}, seed={args.seed}")
    try:
        validate_generate_params(args.n, args.seed)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    try:
        df = generate_dataset(n=args.n, seed=args.seed)
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        sys.exit(1)

    try:
        _ensure_dirs(args.out)
        save_csv(df, args.out)
        logger.info(f"Generated dataset written to {args.out} with {len(df)} rows")
    except PermissionError:
        logger.error(f"Permission denied when writing to {args.out}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Failed to write file {args.out}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error writing output: {e}")
        sys.exit(1)

def cmd_analyze(args):
    logger.info(f"Analyzing dataset from {args.inp}")

    # Read input file
    try:
        df = pd.read_csv(args.inp)
        logger.info(f"Loaded {len(df)} projects")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.inp}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty: {args.inp}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file {args.inp}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error reading {args.inp}: {e}")
        sys.exit(1)

    # Validate data
    try:
        required_cols = {"project_name", "tech_area", "horizon", "budget_musd", "success",
                         "initial_trl", "target_trl", "duration_months", "team_size",
                         "complexity", "novelty", "stakeholder_support", "market_potential"}
        validate_dataframe_columns(df, required_cols)
        validate_ml_params(args.n_estimators, args.learning_rate, args.max_depth, args.cv)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    # Train model
    try:
        logger.info(f"Training model: n_estimators={args.n_estimators}, lr={args.learning_rate}, max_depth={args.max_depth}")
        model, res = train_and_eval(
            df,
            seed=args.seed,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            cv_folds=args.cv
        )
        logger.info(f"Model performance - AUC: {res.auc:.3f}, Accuracy: {res.acc:.3f}, Brier: {res.brier:.3f}")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)

    # Annotate and save
    try:
        analyzed = batch_annotate(df, model)
        _ensure_dirs(args.out)
        analyzed.to_csv(args.out, index=False)
        logger.info(f"Analyzed dataset written to {args.out}")
    except PermissionError:
        logger.error(f"Permission denied when writing to {args.out}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Failed to write file {args.out}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error writing output: {e}")
        sys.exit(1)

    # Save model if requested
    if hasattr(args, 'save_model_path') and args.save_model_path:
        try:
            save_model(model, args.save_model_path)
            logger.info(f"Model saved to {args.save_model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            sys.exit(1)

    # Generate visualizations if requested
    if hasattr(args, 'plot_calibration') and args.plot_calibration:
        try:
            plot_calibration_curve(res.calibration_curve, output_path=args.plot_calibration)
            logger.info(f"Calibration curve saved to {args.plot_calibration}")
        except Exception as e:
            logger.error(f"Failed to generate calibration plot: {e}")
            sys.exit(1)

    if hasattr(args, 'plot_importance') and args.plot_importance:
        try:
            plot_feature_importances(res.feature_importances, output_path=args.plot_importance)
            logger.info(f"Feature importance plot saved to {args.plot_importance}")
        except Exception as e:
            logger.error(f"Failed to generate feature importance plot: {e}")
            sys.exit(1)

    report = {
        "auc": res.auc,
        "acc": res.acc,
        "brier": res.brier,
        "feature_importances": res.feature_importances,
        "calibration_curve": res.calibration_curve
    }
    print(json.dumps(report, indent=2))

def cmd_optimize(args):
    logger.info(f"Optimizing portfolio from {args.inp}")

    # Read input file
    try:
        df = pd.read_csv(args.inp)
        logger.info(f"Loaded {len(df)} analyzed projects")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.inp}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty: {args.inp}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file {args.inp}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error reading {args.inp}: {e}")
        sys.exit(1)

    # Validate data
    try:
        validate_analyzed_dataframe(df)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    # Build and validate configuration
    try:
        cfg = OptimizeConfig(
            total_budget=float(args.budget),
            horizon_pct={"h1":args.h1, "h2":args.h2, "h3":args.h3},
            risk_limits={"h1_max_failure":args.h1f, "h2_max_failure":args.h2f, "h3_max_failure":args.h3f},
            weights={"strategic":args.w_strat, "financial":args.w_fin},
            scenarios_n=args.scenarios,
            budget_volatility=args.volatility,
            seed=args.seed
        )
        validate_optimize_config(cfg)
        logger.info(f"Configuration: budget=${cfg.total_budget:,.0f}, scenarios={cfg.scenarios_n}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in configuration: {e}")
        sys.exit(1)

    # Run optimization
    try:
        res = greedy_knapsack(df, cfg)
        logger.info(f"Optimization complete: selected {res['selected_projects']} projects, "
                    f"allocated ${res['total_budget_allocated']:,.0f} ({res['budget_utilization']:.1%} utilization)")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

    # Write results
    try:
        _ensure_dirs(args.out)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info(f"Optimization results written to {args.out}")
    except PermissionError:
        logger.error(f"Permission denied when writing to {args.out}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Failed to write file {args.out}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error writing output: {e}")
        sys.exit(1)

    # Generate visualizations if requested
    if hasattr(args, 'plot_portfolio') and args.plot_portfolio:
        try:
            plot_portfolio_distribution(res, output_path=args.plot_portfolio)
            logger.info(f"Portfolio distribution plot saved to {args.plot_portfolio}")
        except Exception as e:
            logger.error(f"Failed to generate portfolio plot: {e}")
            sys.exit(1)

    if hasattr(args, 'plot_risk_return') and args.plot_risk_return:
        try:
            plot_risk_vs_return(res["recommendations"], output_path=args.plot_risk_return)
            logger.info(f"Risk vs return plot saved to {args.plot_risk_return}")
        except Exception as e:
            logger.error(f"Failed to generate risk-return plot: {e}")
            sys.exit(1)

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="EMD Praxis Research CLI")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI arguments override config file values."
    )
    sub = parser.add_subparsers()

    p1 = sub.add_parser("generate", help="Generate synthetic IR&D dataset")
    p1.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of synthetic projects to generate (default: 1000). "
             "Larger datasets provide better ML training but increase computation time."
    )
    p1.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). "
             "Use the same seed to reproduce identical datasets across runs."
    )
    p1.add_argument(
        "--out",
        type=str,
        default="data/projects.csv",
        help="Output CSV file path (default: data/projects.csv). "
             "Parent directories will be created if they don't exist."
    )
    p1.set_defaults(func=cmd_generate)

    p2 = sub.add_parser("analyze", help="Train ML model and annotate dataset")
    p2.add_argument(
        "--inp",
        type=str,
        required=True,
        help="Input CSV file path containing project data. "
             "Must include columns: project_name, tech_area, horizon, budget_musd, success, "
             "initial_trl, target_trl, duration_months, team_size, complexity, novelty, "
             "stakeholder_support, market_potential."
    )
    p2.add_argument(
        "--out",
        type=str,
        default="data/projects_analyzed.csv",
        help="Output CSV file path for analyzed dataset (default: data/projects_analyzed.csv). "
             "Will contain original data plus predicted_success_probability and risk_level."
    )
    p2.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and model initialization (default: 42)."
    )
    p2.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting stages (trees) in the GradientBoosting model (default: 200). "
             "More estimators improve accuracy but increase training time."
    )
    p2.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for gradient boosting (default: 0.05). "
             "Lower values require more estimators but may improve generalization."
    )
    p2.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth of individual trees (default: 3). "
             "Deeper trees capture more complex patterns but risk overfitting."
    )
    p2.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds for model evaluation (default: 5)."
    )
    p2.add_argument(
        "--save-model",
        dest="save_model_path",
        type=str,
        default=None,
        help="Optional path to save the trained model (e.g., models/predictor.joblib). "
             "If not provided, model will not be saved."
    )
    p2.add_argument(
        "--plot-calibration",
        type=str,
        default=None,
        help="Optional path to save calibration curve plot (e.g., plots/calibration.png). "
             "Visualizes model calibration quality."
    )
    p2.add_argument(
        "--plot-importance",
        type=str,
        default=None,
        help="Optional path to save feature importance plot (e.g., plots/importance.png). "
             "Shows which features most influence predictions."
    )
    p2.set_defaults(func=cmd_analyze)

    p3 = sub.add_parser("optimize", help="Optimize portfolio under constraints")
    p3.add_argument(
        "--inp",
        type=str,
        required=True,
        help="Input CSV file path from 'analyze' command. "
             "Must contain predicted_success_probability column."
    )
    p3.add_argument(
        "--budget",
        type=float,
        required=True,
        help="Total budget available for portfolio allocation (in dollars). "
             "Example: 10000000 for $10M."
    )
    p3.add_argument(
        "--h1",
        type=float,
        default=0.40,
        help="Budget allocation percentage for Horizon 1 (near-term) projects (default: 0.40). "
             "Must sum to 1.0 with --h2 and --h3."
    )
    p3.add_argument(
        "--h2",
        type=float,
        default=0.50,
        help="Budget allocation percentage for Horizon 2 (mid-term) projects (default: 0.50)."
    )
    p3.add_argument(
        "--h3",
        type=float,
        default=0.10,
        help="Budget allocation percentage for Horizon 3 (far-term) projects (default: 0.10)."
    )
    p3.add_argument(
        "--h1f",
        type=float,
        default=0.10,
        help="Maximum failure rate allowed for Horizon 1 projects (default: 0.10). "
             "Projects with success_probability < 0.5 are considered failures."
    )
    p3.add_argument(
        "--h2f",
        type=float,
        default=0.50,
        help="Maximum failure rate allowed for Horizon 2 projects (default: 0.50)."
    )
    p3.add_argument(
        "--h3f",
        type=float,
        default=1.00,
        help="Maximum failure rate allowed for Horizon 3 projects (default: 1.00, no limit)."
    )
    p3.add_argument(
        "--w-strat",
        type=float,
        default=0.70,
        help="Weight for strategic value in PCV calculation (default: 0.70). "
             "Must sum to 1.0 with --w-fin."
    )
    p3.add_argument(
        "--w-fin",
        type=float,
        default=0.30,
        help="Weight for financial value (success probability) in PCV calculation (default: 0.30)."
    )
    p3.add_argument(
        "--scenarios",
        type=int,
        default=100,
        help="Number of Monte Carlo scenarios for PCV uncertainty modeling (default: 100). "
             "More scenarios increase accuracy but slow computation."
    )
    p3.add_argument(
        "--volatility",
        type=float,
        default=0.20,
        help="Budget/market volatility standard deviation for scenario generation (default: 0.20)."
    )
    p3.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility in Monte Carlo simulations (default: 42)."
    )
    p3.add_argument(
        "--out",
        type=str,
        default="results/opt_result.json",
        help="Output JSON file path for optimization results (default: results/opt_result.json)."
    )
    p3.add_argument(
        "--plot-portfolio",
        type=str,
        default=None,
        help="Optional path to save portfolio distribution plot (e.g., plots/portfolio.png). "
             "Visualizes budget allocation, project counts, tech areas, and risk levels."
    )
    p3.add_argument(
        "--plot-risk-return",
        type=str,
        default=None,
        help="Optional path to save risk vs return scatter plot (e.g., plots/risk_return.png). "
             "Shows relationship between success probability and expected value."
    )
    p3.set_defaults(func=cmd_optimize)

    args = parser.parse_args()

    # Load config file if specified and merge with CLI arguments
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Merge config with args (CLI args take precedence)
        # Only set values that weren't explicitly provided on CLI
        if hasattr(args, 'func'):
            cmd_name = args.func.__name__.replace('cmd_', '')

            # Common seed parameter
            if 'seed' in config and not any(f'--seed' in arg for arg in sys.argv):
                args.seed = config['seed']

            # Generate command
            if cmd_name == 'generate' and 'data' in config:
                if not any(f'--n' in arg for arg in sys.argv):
                    args.n = config['data'].get('n_projects', args.n)
                if not any(f'--out' in arg for arg in sys.argv):
                    args.out = config['data'].get('out_csv', args.out)

            # Analyze command
            elif cmd_name == 'analyze' and 'ml' in config:
                ml_config = config['ml']
                if not any('--n-estimators' in arg for arg in sys.argv):
                    args.n_estimators = ml_config.get('n_estimators', args.n_estimators)
                if not any('--learning-rate' in arg for arg in sys.argv):
                    args.learning_rate = ml_config.get('learning_rate', args.learning_rate)
                if not any('--max-depth' in arg for arg in sys.argv):
                    args.max_depth = ml_config.get('max_depth', args.max_depth)
                if not any('--cv' in arg for arg in sys.argv):
                    args.cv = ml_config.get('cv_folds', args.cv)

            # Optimize command
            elif cmd_name == 'optimize' and 'optimize' in config:
                opt_config = config['optimize']
                if not any('--budget' in arg for arg in sys.argv):
                    args.budget = opt_config.get('total_budget', args.budget if hasattr(args, 'budget') else None)
                if 'horizon_pct' in opt_config:
                    if not any('--h1' in arg for arg in sys.argv):
                        args.h1 = opt_config['horizon_pct'].get('h1', args.h1)
                    if not any('--h2' in arg for arg in sys.argv):
                        args.h2 = opt_config['horizon_pct'].get('h2', args.h2)
                    if not any('--h3' in arg for arg in sys.argv):
                        args.h3 = opt_config['horizon_pct'].get('h3', args.h3)
                if 'risk_limits' in opt_config:
                    if not any('--h1f' in arg for arg in sys.argv):
                        args.h1f = opt_config['risk_limits'].get('h1_max_failure', args.h1f)
                    if not any('--h2f' in arg for arg in sys.argv):
                        args.h2f = opt_config['risk_limits'].get('h2_max_failure', args.h2f)
                    if not any('--h3f' in arg for arg in sys.argv):
                        args.h3f = opt_config['risk_limits'].get('h3_max_failure', args.h3f)
                if 'weights' in opt_config:
                    if not any('--w-strat' in arg for arg in sys.argv):
                        args.w_strat = opt_config['weights'].get('strategic', args.w_strat)
                    if not any('--w-fin' in arg for arg in sys.argv):
                        args.w_fin = opt_config['weights'].get('financial', args.w_fin)
                if 'scenarios' in opt_config:
                    if not any('--scenarios' in arg for arg in sys.argv):
                        args.scenarios = opt_config['scenarios'].get('n', args.scenarios)
                    if not any('--volatility' in arg for arg in sys.argv):
                        args.volatility = opt_config['scenarios'].get('budget_volatility', args.volatility)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
