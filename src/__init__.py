"""
Fall prediction package - machine learning models for fall risk prediction.

This package provides tools for:
- Loading and preprocessing fall prediction data
- Evaluating machine learning models with bootstrap confidence intervals
- Calculating comprehensive metrics including AUC-ROC and AUC-PR
- Generating publication-quality visualizations
"""

__version__ = "0.1.0"

# Import key functions and classes for easier access
from .data_loader import FallDataLoader
from .model_evaluation import (
    calculate_metrics,
    bootstrap_metrics,
    evaluate_model,
    format_results_table,
    print_results_summary,
    get_roc_data,
    get_pr_data
)
from .visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_metrics_comparison,
    plot_confusion_matrices,
    plot_comprehensive_comparison,
    save_all_visualizations
)
from .pca_features import GaitFeaturePCA

# Import constants for metric names
from .constants import (
    METRIC_AUC_ROC,
    METRIC_AUC_PR,
    METRIC_ACCURACY,
    METRIC_SENSITIVITY,
    METRIC_SPECIFICITY,
    METRIC_PRECISION,
    METRIC_F1,
    ALL_METRICS,
    CORE_METRICS,
    get_display_name,
    get_upper_name,
    get_lower_name
)

__all__ = [
    # Data loading
    'FallDataLoader',
    'GaitFeaturePCA',

    # Model evaluation
    'calculate_metrics',
    'bootstrap_metrics',
    'evaluate_model',
    'format_results_table',
    'print_results_summary',
    'get_roc_data',
    'get_pr_data',

    # Visualization
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_metrics_comparison',
    'plot_confusion_matrices',
    'plot_comprehensive_comparison',
    'save_all_visualizations',

    # Constants
    'METRIC_AUC_ROC',
    'METRIC_AUC_PR',
    'METRIC_ACCURACY',
    'METRIC_SENSITIVITY',
    'METRIC_SPECIFICITY',
    'METRIC_PRECISION',
    'METRIC_F1',
    'ALL_METRICS',
    'CORE_METRICS',
    'get_display_name',
    'get_upper_name',
    'get_lower_name',
]
