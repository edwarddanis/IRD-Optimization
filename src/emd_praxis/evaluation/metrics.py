"""
Evaluation utilities for model performance and reporting.
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

def summarize_classification(y_true, y_proba, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
