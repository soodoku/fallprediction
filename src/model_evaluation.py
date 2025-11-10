"""
Model evaluation module with bootstrap standard error calculations.

This module provides comprehensive model evaluation metrics including:
- AUC-ROC (Area Under the Receiver Operating Characteristic curve)
- AUC-PR (Area Under the Precision-Recall curve)
- Accuracy, Sensitivity, Specificity, Precision, F1 Score
- Bootstrap-based standard errors and confidence intervals
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score
)
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import optimal cutoffs package
from optimal_cutoffs import get_optimal_threshold

# Import constants for consistent metric naming
from .constants import (
    METRIC_AUC_ROC, METRIC_AUC_PR, METRIC_ACCURACY, METRIC_SENSITIVITY,
    METRIC_SPECIFICITY, METRIC_PRECISION, METRIC_F1, CORE_METRICS,
    get_upper_name, get_display_name
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary: 0 or 1)
    y_pred : np.ndarray
        Predicted labels (binary: 0 or 1)
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class (between 0 and 1)

    Returns
    -------
    Dict[str, float]
        Dictionary with the following keys:
        - accuracy: Overall classification accuracy
        - sensitivity: True positive rate (recall)
        - specificity: True negative rate
        - precision: Positive predictive value
        - f1: F1 score (harmonic mean of precision and recall)
        - auc_roc: Area under ROC curve (if y_proba provided)
        - auc_pr: Area under Precision-Recall curve (if y_proba provided)
        - tp, tn, fp, fn: Confusion matrix elements
    """
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate core metrics using constants
    metrics = {
        METRIC_ACCURACY: accuracy_score(y_true, y_pred),
        METRIC_SENSITIVITY: tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Recall / TPR
        METRIC_SPECIFICITY: tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # TNR
        METRIC_PRECISION: precision_score(y_true, y_pred, zero_division=0),
        METRIC_F1: f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

    # Add AUC metrics if probabilities provided
    if y_proba is not None:
        try:
            # AUC-ROC: Area under Receiver Operating Characteristic curve
            metrics[METRIC_AUC_ROC] = roc_auc_score(y_true, y_proba)
        except (ValueError, RuntimeWarning):
            metrics[METRIC_AUC_ROC] = np.nan

        try:
            # AUC-PR: Area under Precision-Recall curve
            # Using average_precision_score which is equivalent to AUC-PR
            metrics[METRIC_AUC_PR] = average_precision_score(y_true, y_proba)
        except (ValueError, RuntimeWarning):
            metrics[METRIC_AUC_PR] = np.nan

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Find the optimal classification threshold that maximizes F1 score.

    Uses the optimal_classification_cutoffs package which implements an
    efficient O(n log n) algorithm for finding the exact threshold that
    maximizes F1 score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_proba : np.ndarray
        Predicted probabilities for the positive class

    Returns
    -------
    Tuple[float, float]
        - optimal_threshold: The threshold that maximizes F1 score
        - max_f1: The maximum F1 score achieved at this threshold

    Notes
    -----
    This implementation uses the optimize_f1_binary function from the
    optimal_classification_cutoffs package, which finds the exact optimal
    threshold through efficient piecewise optimization.

    For imbalanced datasets (like fall prediction with ~20% positive class),
    the optimal threshold is typically lower than the default 0.5, allowing
    for better sensitivity while maintaining reasonable specificity.
    """
    # Use the package to find optimal F1 threshold
    result = get_optimal_threshold(y_true, y_proba, metric='f1')

    # The result object contains the optimal threshold and can generate predictions
    # For binary classification, result.thresholds is an array with one element
    optimal_threshold = result.thresholds[0] if hasattr(result.thresholds, '__len__') else result.thresholds

    # Calculate the F1 score at this threshold
    y_pred_optimal = result.predict(y_proba)
    max_f1 = f1_score(y_true, y_pred_optimal)

    return optimal_threshold, max_f1


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics with bootstrap standard errors and confidence intervals.

    This function performs bootstrap resampling to estimate the uncertainty
    in classification metrics. For each bootstrap sample, all metrics are
    recalculated, and the distribution is used to compute standard errors
    and 95% confidence intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with metric names as keys, each containing:
        - 'mean': Mean value across bootstrap samples
        - 'se': Standard error (SD of bootstrap distribution)
        - 'ci_lower': Lower bound of 95% CI (2.5th percentile)
        - 'ci_upper': Upper bound of 95% CI (97.5th percentile)

    Notes
    -----
    Bootstrap samples with only one class are skipped. If too many samples
    are skipped, this may indicate an issue with the dataset or model.
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    np.random.seed(random_state)
    n_samples = len(y_true)

    # Storage for bootstrap samples using constants
    bootstrap_results = {
        METRIC_ACCURACY: [],
        METRIC_SENSITIVITY: [],
        METRIC_SPECIFICITY: [],
        METRIC_PRECISION: [],
        METRIC_F1: []
    }
    if y_proba is not None:
        bootstrap_results[METRIC_AUC_ROC] = []
        bootstrap_results[METRIC_AUC_PR] = []

    # Track skipped samples
    skipped_samples = 0

    # Perform bootstrap
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_proba_boot = y_proba[indices] if y_proba is not None else None

        # Calculate metrics for this bootstrap sample
        try:
            metrics = calculate_metrics(y_true_boot, y_pred_boot, y_proba_boot)

            # Append core metrics
            bootstrap_results[METRIC_ACCURACY].append(metrics[METRIC_ACCURACY])
            bootstrap_results[METRIC_SENSITIVITY].append(metrics[METRIC_SENSITIVITY])
            bootstrap_results[METRIC_SPECIFICITY].append(metrics[METRIC_SPECIFICITY])
            bootstrap_results[METRIC_PRECISION].append(metrics[METRIC_PRECISION])
            bootstrap_results[METRIC_F1].append(metrics[METRIC_F1])

            # Append AUC metrics if available and valid
            if y_proba is not None:
                if not np.isnan(metrics[METRIC_AUC_ROC]):
                    bootstrap_results[METRIC_AUC_ROC].append(metrics[METRIC_AUC_ROC])
                if not np.isnan(metrics[METRIC_AUC_PR]):
                    bootstrap_results[METRIC_AUC_PR].append(metrics[METRIC_AUC_PR])
        except (ValueError, RuntimeWarning):
            # Skip bootstrap samples with only one class
            skipped_samples += 1
            continue

    # Warn if too many samples were skipped
    if skipped_samples > 0:
        skip_pct = 100 * skipped_samples / n_bootstrap
        if skip_pct > 10:
            warnings.warn(
                f"Skipped {skipped_samples}/{n_bootstrap} ({skip_pct:.1f}%) "
                f"bootstrap samples due to single-class resamples. "
                f"This may indicate dataset imbalance issues."
            )

    # Calculate statistics
    results = {}
    for metric_name, values in bootstrap_results.items():
        if len(values) > 0:
            values_array = np.array(values)
            results[metric_name] = {
                'mean': np.mean(values_array),
                'se': np.std(values_array),  # Standard error (SD of bootstrap distribution)
                'ci_lower': np.percentile(values_array, 2.5),
                'ci_upper': np.percentile(values_array, 97.5)
            }
        else:
            results[metric_name] = {
                'mean': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }

    return results


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    use_oob: bool = False,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Dict:
    """
    Comprehensive model evaluation with bootstrap standard errors.

    Evaluates model performance using both default (0.5) and optimal F1-based
    thresholds. The optimal threshold maximizes F1 score, which often provides
    better sensitivity-specificity balance for imbalanced datasets.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    use_oob : bool
        Whether to extract OOB score (for Random Forest)
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed

    Returns:
    --------
    dict with comprehensive evaluation results including:
        - point_estimates: Metrics at default 0.5 threshold
        - bootstrap_stats: Bootstrap statistics at default threshold
        - optimal_threshold: Best threshold for F1 score
        - optimal_metrics: Point estimates at optimal threshold
        - optimal_bootstrap_stats: Bootstrap statistics at optimal threshold
    """
    results = {'model_name': model_name}

    # Get predictions with default 0.5 threshold
    y_pred_default = model.predict(X_test)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None

    # Calculate point estimates at default threshold (0.5)
    point_metrics = calculate_metrics(y_test, y_pred_default, y_proba)
    results['point_estimates'] = point_metrics

    # Calculate bootstrap statistics at default threshold
    bootstrap_stats = bootstrap_metrics(
        y_test, y_pred_default, y_proba,
        n_bootstrap=n_bootstrap,
        random_state=random_state
    )
    results['bootstrap_stats'] = bootstrap_stats

    # Find optimal threshold and evaluate at that threshold
    if y_proba is not None:
        # Find optimal F1 threshold
        optimal_threshold, max_f1 = find_optimal_threshold(y_test, y_proba)
        results['optimal_threshold'] = optimal_threshold

        # Generate predictions at optimal threshold
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

        # Calculate point estimates at optimal threshold
        optimal_metrics = calculate_metrics(y_test, y_pred_optimal, y_proba)
        results['optimal_point_estimates'] = optimal_metrics

        # Calculate bootstrap statistics at optimal threshold
        optimal_bootstrap_stats = bootstrap_metrics(
            y_test, y_pred_optimal, y_proba,
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        results['optimal_bootstrap_stats'] = optimal_bootstrap_stats
    else:
        results['optimal_threshold'] = None
        results['optimal_point_estimates'] = None
        results['optimal_bootstrap_stats'] = None

    # Get OOB score if available
    if use_oob and hasattr(model, 'oob_score_'):
        results['oob_score'] = model.oob_score_
    else:
        results['oob_score'] = None

    # Store predictions and probabilities
    results['y_pred'] = y_pred_default  # Default threshold predictions
    results['y_proba'] = y_proba
    results['y_true'] = y_test

    return results


def format_results_table(results_list: List[Dict]) -> pd.DataFrame:
    """
    Format evaluation results into a clean table with metrics and standard errors.

    Parameters
    ----------
    results_list : List[Dict]
        List of results dictionaries from evaluate_model()

    Returns
    -------
    pd.DataFrame
        Formatted results table with columns:
        - Model: Model name
        - OOB_Score: Out-of-bag score (if available)
        - TP, TN, FP, FN: Confusion matrix elements
        - For each metric: [METRIC] and [METRIC]_SE columns

    Notes
    -----
    The table includes both AUC-ROC and AUC-PR metrics when probabilities
    are available. Metrics are ordered by importance: AUC metrics first,
    followed by accuracy, sensitivity, specificity, precision, and F1.
    """
    rows = []

    for result in results_list:
        model_name = result['model_name']
        bootstrap_stats = result['bootstrap_stats']

        row = {'Model': model_name}

        # Add metrics with mean ± SE format using constants
        # Include both AUC-ROC and AUC-PR
        metrics_to_add = [
            METRIC_AUC_ROC, METRIC_AUC_PR, METRIC_ACCURACY,
            METRIC_SENSITIVITY, METRIC_SPECIFICITY, METRIC_PRECISION, METRIC_F1
        ]

        for metric in metrics_to_add:
            metric_upper = get_upper_name(metric)
            if metric in bootstrap_stats:
                mean = bootstrap_stats[metric]['mean']
                se = bootstrap_stats[metric]['se']
                row[metric_upper] = mean
                row[f'{metric_upper}_SE'] = se
            else:
                row[metric_upper] = np.nan
                row[f'{metric_upper}_SE'] = np.nan

        # Add OOB score if available
        if result['oob_score'] is not None:
            row['OOB_Score'] = result['oob_score']

        # Add confusion matrix elements
        pe = result['point_estimates']
        row['TP'] = pe['tp']
        row['TN'] = pe['tn']
        row['FP'] = pe['fp']
        row['FN'] = pe['fn']

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns for better presentation
    metric_cols = []
    metric_order = [
        'AUC_ROC', 'AUC_PR', 'ACCURACY', 'SENSITIVITY',
        'SPECIFICITY', 'PRECISION', 'F1'
    ]
    for metric in metric_order:
        if metric in df.columns:
            metric_cols.extend([metric, f'{metric}_SE'])

    other_cols = ['Model']
    if 'OOB_Score' in df.columns:
        other_cols.append('OOB_Score')
    other_cols.extend(['TP', 'TN', 'FP', 'FN'])

    final_cols = other_cols + metric_cols
    final_cols = [c for c in final_cols if c in df.columns]

    return df[final_cols]


def format_threshold_comparison(results_list: List[Dict]) -> pd.DataFrame:
    """
    Format a comparison table showing metrics at both default (0.5) and optimal thresholds.

    Parameters
    ----------
    results_list : List[Dict]
        List of results dictionaries from evaluate_model()

    Returns
    -------
    pd.DataFrame
        Comparison table with columns for model name, threshold, optimal threshold value,
        and all metrics at both thresholds

    Notes
    -----
    This table is useful for understanding the trade-offs between using the default
    0.5 threshold versus the optimal F1-maximizing threshold. For imbalanced datasets,
    the optimal threshold is often lower than 0.5, which typically increases sensitivity
    at the cost of some specificity.
    """
    rows = []

    for result in results_list:
        model_name = result['model_name']

        # Add row for default threshold (0.5)
        default_stats = result['bootstrap_stats']
        default_point = result['point_estimates']

        default_row = {
            'Model': model_name,
            'Threshold': 'Default (0.5)',
            'Threshold_Value': 0.5
        }

        # Add metrics
        for metric in [METRIC_SENSITIVITY, METRIC_SPECIFICITY, METRIC_PRECISION, METRIC_F1, METRIC_ACCURACY]:
            metric_upper = get_upper_name(metric)
            if metric in default_stats:
                mean = default_stats[metric]['mean']
                se = default_stats[metric]['se']
                default_row[metric_upper] = mean
                default_row[f'{metric_upper}_SE'] = se

        # Add confusion matrix
        default_row['TP'] = default_point['tp']
        default_row['TN'] = default_point['tn']
        default_row['FP'] = default_point['fp']
        default_row['FN'] = default_point['fn']

        rows.append(default_row)

        # Add row for optimal threshold (if available)
        if result['optimal_threshold'] is not None:
            optimal_stats = result['optimal_bootstrap_stats']
            optimal_point = result['optimal_point_estimates']
            optimal_threshold = result['optimal_threshold']

            optimal_row = {
                'Model': model_name,
                'Threshold': 'Optimal (F1)',
                'Threshold_Value': optimal_threshold
            }

            # Add metrics
            for metric in [METRIC_SENSITIVITY, METRIC_SPECIFICITY, METRIC_PRECISION, METRIC_F1, METRIC_ACCURACY]:
                metric_upper = get_upper_name(metric)
                if metric in optimal_stats:
                    mean = optimal_stats[metric]['mean']
                    se = optimal_stats[metric]['se']
                    optimal_row[metric_upper] = mean
                    optimal_row[f'{metric_upper}_SE'] = se

            # Add confusion matrix
            optimal_row['TP'] = optimal_point['tp']
            optimal_row['TN'] = optimal_point['tn']
            optimal_row['FP'] = optimal_point['fp']
            optimal_row['FN'] = optimal_point['fn']

            rows.append(optimal_row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns
    base_cols = ['Model', 'Threshold', 'Threshold_Value']
    metric_cols = []
    for metric in ['SENSITIVITY', 'SPECIFICITY', 'PRECISION', 'F1', 'ACCURACY']:
        if metric in df.columns:
            metric_cols.extend([metric, f'{metric}_SE'])

    confusion_cols = ['TP', 'TN', 'FP', 'FN']

    final_cols = base_cols + metric_cols + confusion_cols
    final_cols = [c for c in final_cols if c in df.columns]

    return df[final_cols]


def print_results_summary(results_df: pd.DataFrame, title: str = "Model Comparison Results"):
    """
    Print a formatted summary of results with key metrics highlighted.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from format_results_table()
    title : str, default="Model Comparison Results"
        Title for the summary output

    Notes
    -----
    This function highlights the best-performing models for AUC-ROC, AUC-PR,
    accuracy, and sensitivity metrics.
    """
    print("\n" + "="*100)
    print(f"{title:^100}")
    print("="*100)

    # Print with nice formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 4)

    print(results_df.to_string(index=False))
    print("="*100)

    # Find and display best models for key metrics
    key_metrics = [
        ('AUC_ROC', 'AUC-ROC'),
        ('AUC_PR', 'AUC-PR'),
        ('ACCURACY', 'Accuracy'),
        ('SENSITIVITY', 'Sensitivity')
    ]

    for metric_col, metric_display in key_metrics:
        if metric_col in results_df.columns and not results_df[metric_col].isna().all():
            best_idx = results_df[metric_col].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_value = results_df.loc[best_idx, metric_col]
            best_se = results_df.loc[best_idx, f'{metric_col}_SE']
            print(f"\nBest {metric_display}: {best_model} "
                  f"({best_value:.4f} ± {best_se:.4f})")

    print("\n")


def get_roc_data(results: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Extract ROC curve data from results.

    Parameters
    ----------
    results : Dict
        Results from evaluate_model()

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]
        - fpr: False positive rates
        - tpr: True positive rates
        - auc_score: AUC-ROC score (bootstrap mean)
        Returns (None, None, None) if probabilities not available
    """
    if results['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
        auc_score = results['bootstrap_stats'][METRIC_AUC_ROC]['mean']
        return fpr, tpr, auc_score
    else:
        return None, None, None


def get_pr_data(results: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Extract Precision-Recall curve data from results.

    Parameters
    ----------
    results : Dict
        Results from evaluate_model()

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]
        - precision: Precision values
        - recall: Recall values
        - auc_score: AUC-PR score (bootstrap mean)
        Returns (None, None, None) if probabilities not available
    """
    if results['y_proba'] is not None:
        precision, recall, _ = precision_recall_curve(results['y_true'], results['y_proba'])
        auc_score = results['bootstrap_stats'][METRIC_AUC_PR]['mean']
        return precision, recall, auc_score
    else:
        return None, None, None
