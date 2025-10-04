"""
Tests for evaluation metrics module.
"""
import pytest
import numpy as np
from emd_praxis.evaluation.metrics import summarize_classification


def test_summarize_classification_perfect_predictions():
    """Test with perfect predictions (AUC=1.0, accuracy=1.0)."""
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.9, 0.8]
    result = summarize_classification(y_true, y_proba)

    assert result["auc"] == 1.0
    assert result["accuracy"] == 1.0
    assert "confusion_matrix" in result
    assert "classification_report" in result

    # Perfect predictions: [[2, 0], [0, 2]]
    cm = result["confusion_matrix"]
    assert cm[0][1] == 0  # No false positives
    assert cm[1][0] == 0  # No false negatives


def test_summarize_classification_worst_predictions():
    """Test with completely wrong predictions."""
    y_true = [0, 0, 1, 1]
    y_proba = [0.9, 0.8, 0.1, 0.2]
    result = summarize_classification(y_true, y_proba)

    assert result["auc"] == 0.0  # Worst possible AUC
    assert result["accuracy"] == 0.0  # All predictions wrong

    # Completely wrong: [[0, 2], [2, 0]]
    cm = result["confusion_matrix"]
    assert cm[0][0] == 0  # No true negatives
    assert cm[1][1] == 0  # No true positives


def test_summarize_classification_random_predictions():
    """Test with random predictions around 0.5."""
    y_true = [0, 1, 0, 1, 0, 1]
    y_proba = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    result = summarize_classification(y_true, y_proba)

    # AUC for random predictions should be 0.5
    assert 0.4 <= result["auc"] <= 0.6  # Allow some tolerance
    assert 0.0 <= result["accuracy"] <= 1.0
    assert isinstance(result["confusion_matrix"], list)
    assert isinstance(result["classification_report"], dict)


def test_summarize_classification_custom_threshold():
    """Test with different threshold values."""
    y_true = [0, 0, 1, 1]
    y_proba = [0.3, 0.4, 0.6, 0.7]

    # With low threshold (0.35), more predictions are positive
    result_low = summarize_classification(y_true, y_proba, threshold=0.35)

    # With high threshold (0.65), fewer predictions are positive
    result_high = summarize_classification(y_true, y_proba, threshold=0.65)

    # AUC should be same regardless of threshold
    assert result_low["auc"] == result_high["auc"]

    # Accuracy should be 0.75 for both in this case (3 out of 4 correct)
    # With threshold 0.35: [0,1,1,1] -> wrong on second 0, correct on rest = 0.75
    # With threshold 0.65: [0,0,0,1] -> correct on 0s, wrong on first 1, correct on last = 0.75
    # So they happen to be equal, not different
    assert result_low["accuracy"] == 0.75
    assert result_high["accuracy"] == 0.75


def test_summarize_classification_with_numpy_arrays():
    """Test that numpy arrays work as input."""
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.1, 0.9])
    result = summarize_classification(y_true, y_proba)

    assert result["auc"] == 1.0
    assert result["accuracy"] == 1.0
    assert isinstance(result["confusion_matrix"], list)


def test_summarize_classification_edge_case_single_class():
    """Test behavior when only one class is present in predictions."""
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.4]  # All below 0.5 threshold

    result = summarize_classification(y_true, y_proba)

    # Should still return valid structure
    assert "auc" in result
    assert "accuracy" in result
    assert "confusion_matrix" in result
    assert "classification_report" in result

    # Accuracy should be 0.5 (got all 0s correct, all 1s wrong)
    assert result["accuracy"] == 0.5


def test_summarize_classification_threshold_boundaries():
    """Test edge cases at threshold boundaries."""
    y_true = [0, 1]

    # Test exactly at threshold
    y_proba = [0.5, 0.5]
    result = summarize_classification(y_true, y_proba, threshold=0.5)

    # At threshold, prediction is 1 (>=)
    assert result["accuracy"] == 0.5  # One correct, one wrong


def test_summarize_classification_output_structure():
    """Test that output has correct structure and types."""
    y_true = [0, 1, 0, 1]
    y_proba = [0.3, 0.7, 0.2, 0.8]
    result = summarize_classification(y_true, y_proba)

    # Check all required keys exist
    assert "auc" in result
    assert "accuracy" in result
    assert "confusion_matrix" in result
    assert "classification_report" in result

    # Check types
    assert isinstance(result["auc"], float)
    assert isinstance(result["accuracy"], float)
    assert isinstance(result["confusion_matrix"], list)
    assert isinstance(result["classification_report"], dict)

    # Check confusion matrix shape
    cm = result["confusion_matrix"]
    assert len(cm) == 2  # 2x2 matrix
    assert len(cm[0]) == 2
    assert len(cm[1]) == 2

    # Check classification report has expected keys
    report = result["classification_report"]
    assert "0" in report or 0 in report
    assert "1" in report or 1 in report
    assert "accuracy" in report


def test_summarize_classification_large_dataset():
    """Test with larger dataset to ensure scalability."""
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_proba = np.random.rand(n)

    result = summarize_classification(y_true, y_proba)

    # Should complete without error
    assert 0.0 <= result["auc"] <= 1.0
    assert 0.0 <= result["accuracy"] <= 1.0

    # Confusion matrix should sum to n
    cm = result["confusion_matrix"]
    total = sum(sum(row) for row in cm)
    assert total == n
