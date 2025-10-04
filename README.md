# EMD Praxis — IR&D Portfolio Optimization (Research Codebase)

This repository contains **reproducible, academic-quality code** for a Doctor of Engineering (D.Eng.) praxis in **Engineering Management**.
It operationalizes an **IR&D portfolio optimization** workflow with three phases:

1. **Synthetic Data Generation** (defense R&D project attributes and outcomes)
2. **ML Modeling** (success probability prediction with cross-validated Gradient Boosting)
3. **Stochastic/Agnostic Portfolio Optimization** (knapsack-style selection under constraints; competition-aware expected value)

The project emphasizes **reproducibility, transparency, and evaluation**:
- Deterministic seeds, typed code, unit tests (pytest), and clear configs (YAML).
- CSV/Parquet inputs/outputs for auditability.
- CLI entry-points for end-to-end experiments.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt

# Run unit tests
pytest -q

# Generate synthetic data (CSV)
python -m emd_praxis.cli generate --n 1000 --out data/projects.csv

# Train model + evaluate + write analyzed CSV
python -m emd_praxis.cli analyze --inp data/projects.csv --out data/projects_analyzed.csv

# Optimize portfolio
python -m emd_praxis.cli optimize --inp data/projects_analyzed.csv --budget 100000000 --out results/opt_result.json
```

## Structure

```
src/emd_praxis/
  data/synthetic_generator.py
  ml/model.py
  optimize/portfolio.py
  evaluation/metrics.py
  cli.py
configs/
  default.yaml
tests/
  test_generator.py
  test_model.py
  test_optimize.py
```

## Academic Notes

- Includes **k-fold CV**, **calibration curves**, and **confusion metrics**.
- Provides **ablation flags** in the config for sensitivity analyses (e.g., change budget volatility).
- Logs parameters and seeds for each run for traceability.
