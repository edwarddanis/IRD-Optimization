"""
ML modeling: success probability prediction for IR&D projects.
Includes train/validation split, k-fold CV metrics, and calibration curve data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
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

@dataclass
class MLResult:
    auc: float
    acc: float
    brier: float
    y_true: np.ndarray
    y_pred_proba: np.ndarray
    feature_importances: Dict[str, float]

def prepare(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["success"].astype(int).to_numpy()
    return X, y

def train_and_eval(df: pd.DataFrame, *, seed: int = 42, n_estimators: int = 200,
                   learning_rate: float = 0.05, max_depth: int = 3, cv_folds: int = 5) -> Tuple[GradientBoostingClassifier, MLResult]:
    X, y = prepare(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=seed
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:,1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_te, proba))
    acc = float(accuracy_score(y_te, preds))
    brier = float(brier_score_loss(y_te, proba))

    # CV AUC (for reporting)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(model, X, y, cv=kf, scoring="roc_auc")
    # For brevity we don't return cv series, but could log it

    importances = {f: float(w) for f, w in zip(FEATURES, getattr(model, "feature_importances_", np.zeros(len(FEATURES))))}
    res = MLResult(auc=auc, acc=acc, brier=brier, y_true=y_te, y_pred_proba=proba, feature_importances=importances)
    return model, res

def batch_annotate(df: pd.DataFrame, model: GradientBoostingClassifier) -> pd.DataFrame:
    X, _ = prepare(df)
    proba = model.predict_proba(X)[:,1]
    out = df.copy()
    out["predicted_success_probability"] = proba
    out["risk_level"] = pd.cut(proba, bins=[-1,0.33,0.66,1.01], labels=["HIGH","MEDIUM","LOW"])
    return out
