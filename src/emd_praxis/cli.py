"""
Command-line interface for the EMD Praxis research codebase.
"""
from __future__ import annotations
import argparse, json, os, sys, logging
import yaml
import pandas as pd
from pathlib import Path

from emd_praxis.data.synthetic_generator import generate_dataset, save_csv
from emd_praxis.ml.model import train_and_eval, batch_annotate
from emd_praxis.optimize.portfolio import OptimizeConfig, greedy_knapsack
from emd_praxis.validation import (
    validate_generate_params, validate_ml_params, validate_optimize_config,
    validate_analyzed_dataframe, validate_dataframe_columns
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

    report = {
        "auc": res.auc, "acc": res.acc, "brier": res.brier,
        "feature_importances": res.feature_importances
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

def main():
    parser = argparse.ArgumentParser(description="EMD Praxis Research CLI")
    sub = parser.add_subparsers()

    p1 = sub.add_parser("generate", help="Generate synthetic IR&D dataset")
    p1.add_argument("--n", type=int, default=1000)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--out", type=str, default="data/projects.csv")
    p1.set_defaults(func=cmd_generate)

    p2 = sub.add_parser("analyze", help="Train ML model and annotate dataset")
    p2.add_argument("--inp", type=str, required=True)
    p2.add_argument("--out", type=str, default="data/projects_analyzed.csv")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--n-estimators", type=int, default=200)
    p2.add_argument("--learning-rate", type=float, default=0.05)
    p2.add_argument("--max-depth", type=int, default=3)
    p2.add_argument("--cv", type=int, default=5)
    p2.set_defaults(func=cmd_analyze)

    p3 = sub.add_parser("optimize", help="Optimize portfolio under constraints")
    p3.add_argument("--inp", type=str, required=True)
    p3.add_argument("--budget", type=float, required=True)
    p3.add_argument("--h1", type=float, default=0.40)
    p3.add_argument("--h2", type=float, default=0.50)
    p3.add_argument("--h3", type=float, default=0.10)
    p3.add_argument("--h1f", type=float, default=0.10)
    p3.add_argument("--h2f", type=float, default=0.50)
    p3.add_argument("--h3f", type=float, default=1.00)
    p3.add_argument("--w-strat", type=float, default=0.70)
    p3.add_argument("--w-fin", type=float, default=0.30)
    p3.add_argument("--scenarios", type=int, default=100)
    p3.add_argument("--volatility", type=float, default=0.20)
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--out", type=str, default="results/opt_result.json")
    p3.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
