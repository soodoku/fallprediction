"""
Visualization module for fall prediction experiments.

This module provides comprehensive visualization functions for model evaluation:
- ROC curves (Receiver Operating Characteristic)
- Precision-Recall curves
- Metrics comparison with error bars
- Confusion matrices
- Multi-panel comprehensive comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import constants for consistent metric naming
from .constants import (
    METRIC_AUC_ROC, METRIC_AUC_PR, METRIC_ACCURACY, METRIC_SENSITIVITY,
    METRIC_SPECIFICITY, METRIC_PRECISION, METRIC_F1,
    get_upper_name, get_display_name
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_roc_curves(
    results_list: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot ROC (Receiver Operating Characteristic) curves for multiple models.

    Parameters
    ----------
    results_list : List[Dict]
        List of results dictionaries from evaluate_model()
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height)

    Notes
    -----
    The ROC curve shows the trade-off between True Positive Rate (sensitivity)
    and False Positive Rate (1 - specificity). AUC-ROC values are displayed
    with bootstrap standard errors.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))

    for i, result in enumerate(results_list):
        if result['y_proba'] is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_proba'])

            # Use constants for accessing metrics
            auc_mean = result['bootstrap_stats'][METRIC_AUC_ROC]['mean']
            auc_se = result['bootstrap_stats'][METRIC_AUC_ROC]['se']

            label = f"{result['model_name']} (AUC-ROC={auc_mean:.3f}±{auc_se:.3f})"
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=label)

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")

    plt.close()


def plot_pr_curves(
    results_list: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot Precision-Recall curves for multiple models.

    Parameters
    ----------
    results_list : List[Dict]
        List of results dictionaries from evaluate_model()
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height)

    Notes
    -----
    The Precision-Recall curve shows the trade-off between precision and recall
    at different classification thresholds. AUC-PR (Average Precision) values
    are displayed with bootstrap standard errors. This metric is particularly
    useful for imbalanced datasets.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))

    for i, result in enumerate(results_list):
        if result['y_proba'] is not None:
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(
                result['y_true'], result['y_proba']
            )

            # Use constants for accessing metrics
            if METRIC_AUC_PR in result['bootstrap_stats']:
                auc_mean = result['bootstrap_stats'][METRIC_AUC_PR]['mean']
                auc_se = result['bootstrap_stats'][METRIC_AUC_PR]['se']

                label = f"{result['model_name']} (AUC-PR={auc_mean:.3f}±{auc_se:.3f})"
                ax.plot(recall, precision, color=colors[i], lw=2, label=label)

    # Calculate baseline (random classifier for imbalanced data)
    # For binary classification, baseline is the proportion of positive samples
    if results_list and results_list[0]['y_true'] is not None:
        baseline = np.mean(results_list[0]['y_true'])
        ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
                  label=f'Random Classifier (baseline={baseline:.3f})')

    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Precision-Recall curves saved to: {save_path}")

    plt.close()


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
):
    """
    Plot bar charts comparing metrics across models with error bars.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from format_results_table()
    metrics : List[str], optional
        List of metrics to plot. If None, defaults to:
        ['AUC_ROC', 'AUC_PR', 'ACCURACY', 'SENSITIVITY', 'SPECIFICITY', 'PRECISION', 'F1']
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(14, 10)
        Figure size in inches (width, height)

    Notes
    -----
    Each metric is displayed as a bar chart with bootstrap standard error bars.
    Values are labeled on top of each bar for easy comparison.
    """
    if metrics is None:
        # Use constants to build default metric list
        metrics = [
            get_upper_name(METRIC_AUC_ROC),
            get_upper_name(METRIC_AUC_PR),
            get_upper_name(METRIC_ACCURACY),
            get_upper_name(METRIC_SENSITIVITY),
            get_upper_name(METRIC_SPECIFICITY),
            get_upper_name(METRIC_PRECISION),
            get_upper_name(METRIC_F1)
        ]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    n_metrics = len(available_metrics)

    if n_metrics == 0:
        print("No metrics available to plot")
        return

    # Create subplots
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]

        # Get values and errors
        values = results_df[metric].values
        se_col = f'{metric}_SE'
        errors = results_df[se_col].values if se_col in results_df.columns else None

        # Create bar plot
        x_pos = np.arange(len(results_df))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add error bars
        if errors is not None:
            ax.errorbar(x_pos, values, yerr=errors, fmt='none', ecolor='black',
                       capsize=5, capthick=2, alpha=0.7)

        # Customize
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ")} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Model Performance Comparison with Bootstrap Standard Errors',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")

    plt.close()


def plot_confusion_matrices(
    results_list: List[Dict],
    save_path: str = None,
    figsize: tuple = (16, 12)
):
    """
    Plot confusion matrices for all models.

    Parameters:
    -----------
    results_list : list
        List of results from evaluate_model()
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    n_models = len(results_list)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, result in enumerate(results_list):
        ax = axes[idx]
        pe = result['point_estimates']

        # Create confusion matrix
        cm = np.array([[pe['tn'], pe['fp']],
                       [pe['fn'], pe['tp']]])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   ax=ax, annot_kws={'size': 14, 'weight': 'bold'},
                   square=True, linewidths=2, linecolor='black')

        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(result['model_name'], fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Non-Faller', 'Faller'])
        ax.set_yticklabels(['Non-Faller', 'Faller'])

    # Remove extra subplots
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Confusion Matrices - All Models',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrices saved to: {save_path}")

    plt.close()


def plot_comprehensive_comparison(
    results_df: pd.DataFrame,
    results_list: List[Dict],
    save_path: str = None,
    figsize: tuple = (18, 12)
):
    """
    Create a comprehensive 4-panel comparison figure.

    Panels:
    1. ROC curves
    2. AUC-ROC comparison bars
    3. Sensitivity & Specificity comparison
    4. All metrics grouped bar chart

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results table
    results_list : list
        List of results from evaluate_model()
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: ROC Curves
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))

    for i, result in enumerate(results_list):
        if result['y_proba'] is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_proba'])

            # Use constants for accessing metrics
            auc_mean = result['bootstrap_stats'][METRIC_AUC_ROC]['mean']
            auc_se = result['bootstrap_stats'][METRIC_AUC_ROC]['se']
            label = f"{result['model_name']} ({auc_mean:.3f}±{auc_se:.3f})"
            ax1.plot(fpr, tpr, color=colors[i], lw=2.5, label=label)

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_title('ROC Curves', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: AUC Comparison (both ROC and PR)
    ax2 = fig.add_subplot(gs[0, 1])
    auc_roc_col = get_upper_name(METRIC_AUC_ROC)
    auc_pr_col = get_upper_name(METRIC_AUC_PR)

    if auc_roc_col in results_df.columns or auc_pr_col in results_df.columns:
        x_pos = np.arange(len(results_df))
        width = 0.35

        # Plot AUC-ROC if available
        if auc_roc_col in results_df.columns:
            values_roc = results_df[auc_roc_col].values
            errors_roc = results_df[f'{auc_roc_col}_SE'].values
            ax2.barh(x_pos - width/2, values_roc, width, label='AUC-ROC',
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.errorbar(values_roc, x_pos - width/2, xerr=errors_roc, fmt='none',
                        ecolor='black', capsize=4, capthick=1.5, alpha=0.7)

        # Plot AUC-PR if available
        if auc_pr_col in results_df.columns:
            values_pr = results_df[auc_pr_col].values
            errors_pr = results_df[f'{auc_pr_col}_SE'].values
            ax2.barh(x_pos + width/2, values_pr, width, label='AUC-PR',
                    color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.errorbar(values_pr, x_pos + width/2, xerr=errors_pr, fmt='none',
                        ecolor='black', capsize=4, capthick=1.5, alpha=0.7)

        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(results_df['Model'])
        ax2.set_xlabel('AUC Score', fontsize=11, fontweight='bold')
        ax2.set_title('AUC Comparison', fontsize=13, fontweight='bold')
        ax2.set_xlim([0, 1.1])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')

    # Panel 3: Sensitivity & Specificity
    ax3 = fig.add_subplot(gs[1, 0])
    if 'SENSITIVITY' in results_df.columns and 'SPECIFICITY' in results_df.columns:
        x_pos = np.arange(len(results_df))
        width = 0.35

        sens_vals = results_df['SENSITIVITY'].values
        spec_vals = results_df['SPECIFICITY'].values
        sens_err = results_df['SENSITIVITY_SE'].values
        spec_err = results_df['SPECIFICITY_SE'].values

        ax3.bar(x_pos - width/2, sens_vals, width, label='Sensitivity',
               color='skyblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.bar(x_pos + width/2, spec_vals, width, label='Specificity',
               color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax3.errorbar(x_pos - width/2, sens_vals, yerr=sens_err, fmt='none',
                    ecolor='black', capsize=3, capthick=1.5, alpha=0.7)
        ax3.errorbar(x_pos + width/2, spec_vals, yerr=spec_err, fmt='none',
                    ecolor='black', capsize=3, capthick=1.5, alpha=0.7)

        ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax3.set_title('Sensitivity vs Specificity', fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax3.set_ylim([0, 1.1])
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: All Metrics Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    metrics_to_plot = ['ACCURACY', 'SENSITIVITY', 'SPECIFICITY', 'PRECISION', 'F1']
    available = [m for m in metrics_to_plot if m in results_df.columns]

    if len(available) > 0:
        x_pos = np.arange(len(results_df))
        width = 0.15
        offsets = np.linspace(-width*len(available)/2, width*len(available)/2, len(available))

        metric_colors = {'ACCURACY': 'steelblue', 'SENSITIVITY': 'forestgreen',
                        'SPECIFICITY': 'firebrick', 'PRECISION': 'darkorange', 'F1': 'purple'}

        for i, metric in enumerate(available):
            values = results_df[metric].values
            ax4.bar(x_pos + offsets[i], values, width,
                   label=metric, color=metric_colors.get(metric, 'gray'),
                   alpha=0.8, edgecolor='black', linewidth=1)

        ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax4.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax4.set_ylim([0, 1.1])
        ax4.legend(fontsize=9, ncol=2)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Comprehensive Model Performance Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Comprehensive comparison saved to: {save_path}")

    plt.close()


def save_all_visualizations(
    results_df: pd.DataFrame,
    results_list: List[Dict],
    output_dir: str = 'outputs/figures'
):
    """
    Generate and save all visualizations including ROC, PR curves, and comparisons.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from format_results_table()
    results_list : List[Dict]
        List of results dictionaries from evaluate_model()
    output_dir : str, default='outputs/figures'
        Directory to save figures (will be created if it doesn't exist)

    Notes
    -----
    This function generates:
    - ROC curves (roc_curves.png)
    - Precision-Recall curves (pr_curves.png)
    - Metrics comparison bar charts (metrics_comparison.png)
    - Confusion matrices (confusion_matrices.png)
    - Comprehensive 4-panel comparison (comprehensive_comparison.png)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # ROC Curves
    plot_roc_curves(results_list, save_path=f'{output_dir}/roc_curves.png')

    # Precision-Recall Curves
    plot_pr_curves(results_list, save_path=f'{output_dir}/pr_curves.png')

    # Metrics comparison
    plot_metrics_comparison(results_df, save_path=f'{output_dir}/metrics_comparison.png')

    # Confusion matrices
    plot_confusion_matrices(results_list, save_path=f'{output_dir}/confusion_matrices.png')

    # Comprehensive comparison
    plot_comprehensive_comparison(results_df, results_list,
                                  save_path=f'{output_dir}/comprehensive_comparison.png')

    print("All visualizations generated successfully!")
