"""
Model evaluation module with bootstrap standard error calculations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, f1_score, precision_score, recall_score
)
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> dict:
    """
    Calculate comprehensive classification metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class

    Returns:
    --------
    dict with metrics:
        accuracy, sensitivity, specificity, precision, f1, auc_roc
    """
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Recall / True Positive Rate
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # True Negative Rate
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

    # Add AUC-ROC if probabilities provided
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = np.nan

    return metrics


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics with bootstrap standard errors.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed

    Returns:
    --------
    dict with keys being metric names and values being:
        {'mean': float, 'se': float, 'ci_lower': float, 'ci_upper': float}
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    np.random.seed(random_state)
    n_samples = len(y_true)

    # Storage for bootstrap samples
    bootstrap_results = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': []
    }
    if y_proba is not None:
        bootstrap_results['auc_roc'] = []

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

            bootstrap_results['accuracy'].append(metrics['accuracy'])
            bootstrap_results['sensitivity'].append(metrics['sensitivity'])
            bootstrap_results['specificity'].append(metrics['specificity'])
            bootstrap_results['precision'].append(metrics['precision'])
            bootstrap_results['f1'].append(metrics['f1'])

            if y_proba is not None and not np.isnan(metrics['auc_roc']):
                bootstrap_results['auc_roc'].append(metrics['auc_roc'])
        except Exception:
            # Skip problematic bootstrap samples
            continue

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
    dict with comprehensive evaluation results
    """
    results = {'model_name': model_name}

    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None

    # Calculate point estimates
    point_metrics = calculate_metrics(y_test, y_pred, y_proba)
    results['point_estimates'] = point_metrics

    # Calculate bootstrap statistics
    bootstrap_stats = bootstrap_metrics(
        y_test, y_pred, y_proba,
        n_bootstrap=n_bootstrap,
        random_state=random_state
    )
    results['bootstrap_stats'] = bootstrap_stats

    # Get OOB score if available
    if use_oob and hasattr(model, 'oob_score_'):
        results['oob_score'] = model.oob_score_
    else:
        results['oob_score'] = None

    # Store predictions and probabilities
    results['y_pred'] = y_pred
    results['y_proba'] = y_proba
    results['y_true'] = y_test

    return results


def format_results_table(results_list: list) -> pd.DataFrame:
    """
    Format evaluation results into a clean table.

    Parameters:
    -----------
    results_list : list
        List of results dictionaries from evaluate_model()

    Returns:
    --------
    pd.DataFrame with formatted results
    """
    rows = []

    for result in results_list:
        model_name = result['model_name']
        bootstrap_stats = result['bootstrap_stats']

        row = {'Model': model_name}

        # Add metrics with mean ± SE format
        for metric in ['auc_roc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']:
            if metric in bootstrap_stats:
                mean = bootstrap_stats[metric]['mean']
                se = bootstrap_stats[metric]['se']
                row[metric.upper()] = mean
                row[f'{metric.upper()}_SE'] = se
            else:
                row[metric.upper()] = np.nan
                row[f'{metric.upper()}_SE'] = np.nan

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
    for metric in ['AUC_ROC', 'ACCURACY', 'SENSITIVITY', 'SPECIFICITY', 'PRECISION', 'F1']:
        if metric in df.columns:
            metric_cols.extend([metric, f'{metric}_SE'])

    other_cols = ['Model']
    if 'OOB_Score' in df.columns:
        other_cols.append('OOB_Score')
    other_cols.extend(['TP', 'TN', 'FP', 'FN'])

    final_cols = other_cols + metric_cols
    final_cols = [c for c in final_cols if c in df.columns]

    return df[final_cols]


def print_results_summary(results_df: pd.DataFrame, title: str = "Model Comparison Results"):
    """
    Print a formatted summary of results.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results table from format_results_table()
    title : str
        Title for the summary
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

    # Find best models
    if 'AUC_ROC' in results_df.columns:
        best_auc_idx = results_df['AUC_ROC'].idxmax()
        print(f"\nBest AUC-ROC: {results_df.loc[best_auc_idx, 'Model']} "
              f"({results_df.loc[best_auc_idx, 'AUC_ROC']:.4f} ± {results_df.loc[best_auc_idx, 'AUC_ROC_SE']:.4f})")

    if 'ACCURACY' in results_df.columns:
        best_acc_idx = results_df['ACCURACY'].idxmax()
        print(f"Best Accuracy: {results_df.loc[best_acc_idx, 'Model']} "
              f"({results_df.loc[best_acc_idx, 'ACCURACY']:.4f} ± {results_df.loc[best_acc_idx, 'ACCURACY_SE']:.4f})")

    if 'SENSITIVITY' in results_df.columns:
        best_sens_idx = results_df['SENSITIVITY'].idxmax()
        print(f"Best Sensitivity: {results_df.loc[best_sens_idx, 'Model']} "
              f"({results_df.loc[best_sens_idx, 'SENSITIVITY']:.4f} ± {results_df.loc[best_sens_idx, 'SENSITIVITY_SE']:.4f})")

    print("\n")


def get_roc_data(results: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract ROC curve data from results.

    Parameters:
    -----------
    results : dict
        Results from evaluate_model()

    Returns:
    --------
    fpr, tpr, auc_score
    """
    if results['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
        auc_score = results['bootstrap_stats']['auc_roc']['mean']
        return fpr, tpr, auc_score
    else:
        return None, None, None
