"""
Constants for the Fall Prediction Project.

This module provides centralized definitions for metric names, display names,
and other constants used throughout the codebase to ensure consistency.
"""

from typing import List, Dict

# Metric names (lowercase for internal use in dictionaries)
METRIC_AUC_ROC = 'auc_roc'
METRIC_AUC_PR = 'auc_pr'
METRIC_ACCURACY = 'accuracy'
METRIC_SENSITIVITY = 'sensitivity'
METRIC_SPECIFICITY = 'specificity'
METRIC_PRECISION = 'precision'
METRIC_F1 = 'f1'

# All classification metrics
ALL_METRICS = [
    METRIC_AUC_ROC,
    METRIC_AUC_PR,
    METRIC_ACCURACY,
    METRIC_SENSITIVITY,
    METRIC_SPECIFICITY,
    METRIC_PRECISION,
    METRIC_F1,
]

# Core metrics for most visualizations (excluding AUC-PR for backward compatibility)
CORE_METRICS = [
    METRIC_AUC_ROC,
    METRIC_ACCURACY,
    METRIC_SENSITIVITY,
    METRIC_SPECIFICITY,
    METRIC_PRECISION,
    METRIC_F1,
]

# Metric display names (for plots and tables)
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    METRIC_AUC_ROC: 'AUC-ROC',
    METRIC_AUC_PR: 'AUC-PR',
    METRIC_ACCURACY: 'Accuracy',
    METRIC_SENSITIVITY: 'Sensitivity',
    METRIC_SPECIFICITY: 'Specificity',
    METRIC_PRECISION: 'Precision',
    METRIC_F1: 'F1 Score',
}

# Uppercase metric names (for DataFrame columns)
METRIC_NAMES_UPPER: Dict[str, str] = {
    METRIC_AUC_ROC: 'AUC_ROC',
    METRIC_AUC_PR: 'AUC_PR',
    METRIC_ACCURACY: 'ACCURACY',
    METRIC_SENSITIVITY: 'SENSITIVITY',
    METRIC_SPECIFICITY: 'SPECIFICITY',
    METRIC_PRECISION: 'PRECISION',
    METRIC_F1: 'F1',
}

# Reverse mapping: uppercase to lowercase
METRIC_NAMES_LOWER: Dict[str, str] = {v: k for k, v in METRIC_NAMES_UPPER.items()}


def get_display_name(metric: str) -> str:
    """
    Get the display name for a metric.

    Parameters
    ----------
    metric : str
        Metric name (lowercase or uppercase)

    Returns
    -------
    str
        Display name for the metric
    """
    # Handle uppercase metric names
    if metric.upper() in METRIC_NAMES_LOWER:
        metric = METRIC_NAMES_LOWER[metric.upper()]

    return METRIC_DISPLAY_NAMES.get(metric, metric.replace('_', ' ').title())


def get_upper_name(metric: str) -> str:
    """
    Get the uppercase name for a metric (for DataFrame columns).

    Parameters
    ----------
    metric : str
        Metric name (lowercase)

    Returns
    -------
    str
        Uppercase metric name
    """
    return METRIC_NAMES_UPPER.get(metric, metric.upper())


def get_lower_name(metric: str) -> str:
    """
    Get the lowercase name for a metric (for internal dictionaries).

    Parameters
    ----------
    metric : str
        Metric name (uppercase or lowercase)

    Returns
    -------
    str
        Lowercase metric name
    """
    if metric in METRIC_NAMES_LOWER:
        return METRIC_NAMES_LOWER[metric]
    return metric.lower()
