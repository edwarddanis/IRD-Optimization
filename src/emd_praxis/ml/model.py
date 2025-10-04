"""
ML modeling: success probability prediction for IR&D projects.
Includes train/validation split, k-fold CV metrics, and calibration curve data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier

FEATURES = [
    "budget_musd","duration_months","team_size","complexity","novelty",
    "stakeholder_support","market_potential","initial_trl","target_trl","horizon"
]

# Model training constants
TEST_SIZE_DEFAULT = 0.2
PREDICTION_THRESHOLD = 0.5

# Risk level binning thresholds
RISK_LOW_THRESHOLD = 0.66
RISK_MEDIUM_THRESHOLD = 0.33

@dataclass
class MLResult:
    auc: float
    acc: float
    brier: float
    y_true: np.ndarray
    y_pred_proba: np.ndarray
    feature_importances: Dict[str, float]
    calibration_curve: Dict[str, Any] = None  # Optional calibration data

def prepare(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["success"].astype(int).to_numpy()
    return X, y

def train_and_eval(df: pd.DataFrame, *, seed: int = 42, n_estimators: int = 200,
                   learning_rate: float = 0.05, max_depth: int = 3, cv_folds: int = 5) -> Tuple[GradientBoostingClassifier, MLResult]:
    X, y = prepare(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE_DEFAULT, random_state=seed, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:,1]
    preds = (proba >= PREDICTION_THRESHOLD).astype(int)

    auc = float(roc_auc_score(y_te, proba))
    acc = float(accuracy_score(y_te, preds))
    brier = float(brier_score_loss(y_te, proba))

    # CV AUC (for reporting)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(model, X, y, cv=kf, scoring="roc_auc")
    # For brevity we don't return cv series, but could log it

    importances = {f: float(w) for f, w in zip(FEATURES, getattr(model, "feature_importances_", np.zeros(len(FEATURES))))}

    # Generate calibration curve data
    calibration_data = generate_calibration_data(y_te, proba, n_bins=10)

    res = MLResult(
        auc=auc,
        acc=acc,
        brier=brier,
        y_true=y_te,
        y_pred_proba=proba,
        feature_importances=importances,
        calibration_curve=calibration_data
    )
    return model, res

def batch_annotate(df: pd.DataFrame, model: GradientBoostingClassifier) -> pd.DataFrame:
    X, _ = prepare(df)
    proba = model.predict_proba(X)[:,1]
    out = df.copy()
    out["predicted_success_probability"] = proba
    out["risk_level"] = pd.cut(
        proba,
        bins=[-1, RISK_MEDIUM_THRESHOLD, RISK_LOW_THRESHOLD, 1.01],
        labels=["HIGH", "MEDIUM", "LOW"]
    )
    return out


def save_model(model: GradientBoostingClassifier, path: str) -> None:
    """
    Save trained model to disk using joblib.

    Parameters
    ----------
    model : GradientBoostingClassifier
        Trained scikit-learn model to save.
    path : str
        File path where model should be saved (typically .joblib or .pkl extension).
        Parent directories will be created if they don't exist.

    Examples
    --------
    >>> model, res = train_and_eval(df)
    >>> save_model(model, "models/success_predictor.joblib")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> GradientBoostingClassifier:
    """
    Load trained model from disk.

    Parameters
    ----------
    path : str
        File path to the saved model file.

    Returns
    -------
    GradientBoostingClassifier
        Loaded scikit-learn model ready for prediction.

    Raises
    ------
    FileNotFoundError
        If the model file doesn't exist at the specified path.

    Examples
    --------
    >>> model = load_model("models/success_predictor.joblib")
    >>> predictions = batch_annotate(df, model)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def generate_calibration_data(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Generate calibration curve data for model evaluation.

    Calibration curves show how well predicted probabilities match actual outcomes.
    A well-calibrated model's predicted probabilities should align with observed frequencies.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities from the model (values between 0 and 1).
    n_bins : int, optional
        Number of bins to use for calibration curve (default: 10).
        More bins provide finer granularity but may be noisier with small datasets.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - prob_true : List[float]
            Fraction of positives in each bin (observed probability)
        - prob_pred : List[float]
            Mean predicted probability in each bin
        - bin_counts : List[int]
            Number of samples in each bin
        - n_bins : int
            Number of bins used

    Notes
    -----
    Perfect calibration occurs when prob_true == prob_pred for all bins.
    Use this data to plot calibration curves or assess model reliability.

    Examples
    --------
    >>> from emd_praxis.ml.model import train_and_eval, generate_calibration_data
    >>> model, res = train_and_eval(df)
    >>> cal_data = generate_calibration_data(res.y_true, res.y_pred_proba)
    >>> print(f"Calibration bins: {cal_data['n_bins']}")
    >>> # Plot: plt.plot(cal_data['prob_pred'], cal_data['prob_true'])
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')

    # Calculate bin counts for each predicted probability bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = []
    for i in range(n_bins):
        count = ((y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])).sum()
        bin_counts.append(int(count))

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }
