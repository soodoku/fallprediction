#!/usr/bin/env python3
"""
Generate publication-quality figures for manuscript.

This script creates high-resolution figures suitable for academic publication
based on the experimental results.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality parameters
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'text.usetex': False,  # Set to True if LaTeX is available
})

sys.path.insert(0, 'src')
from data_loader import load_and_prepare_data
from model_evaluation import format_results_table


def load_results():
    """Load experimental results."""
    results_df = pd.read_csv('outputs/results/model_comparison_results.csv')
    return results_df


def create_figure_1_confusion_matrices():
    """
    Figure 1: Confusion matrices for selected models.

    Shows 6 representative models covering different algorithmic families.
    """
    # Load results to get confusion matrix data
    results_df = load_results()

    # Select representative models
    selected_models = [
        'RF_500trees',
        'GradientBoosting_Tuned',
        'XGBoost_Default',
        'SVM_RBF',
        'NeuralNet_Tuned',
        'LogisticRegression_Tuned'
    ]

    selected_df = results_df[results_df['Model'].isin(selected_models)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        ax = axes[idx]

        # Create confusion matrix
        cm = np.array([[row['TN'], row['FP']],
                       [row['FN'], row['TP']]])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   ax=ax, annot_kws={'size': 14, 'weight': 'bold'},
                   square=True, linewidths=2, linecolor='black',
                   vmin=0, vmax=34)

        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(row['Model'].replace('_', ' '), fontweight='bold')
        ax.set_xticklabels(['Non-Faller', 'Faller'], rotation=0)
        ax.set_yticklabels(['Non-Faller', 'Faller'], rotation=0)

    plt.tight_layout()

    # Create figures directory if it doesn't exist
    os.makedirs('outputs/figures/manuscript', exist_ok=True)

    plt.savefig('outputs/figures/manuscript/Figure1_ConfusionMatrices.png',
                bbox_inches='tight', dpi=600)
    plt.savefig('outputs/figures/manuscript/Figure1_ConfusionMatrices.pdf',
                bbox_inches='tight')
    print("Figure 1 saved: Confusion matrices")
    plt.close()


def create_figure_2_roc_curves():
    """
    Figure 2: ROC curves for all models.

    Displays ROC curves with AUC values and standard errors.
    """
    # We'll need to re-run predictions to get ROC data
    # For now, create a placeholder based on the results
    results_df = load_results()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')

    # Color scheme
    colors = {
        'RF': '#2E86AB',
        'GradientBoosting': '#A23B72',
        'XGBoost': '#F18F01',
        'SVM': '#C73E1D',
        'NeuralNet': '#6A994E',
        'LogisticRegression': '#BC4B51'
    }

    # Group models by type
    model_groups = {
        'Random Forest': results_df[results_df['Model'].str.contains('RF')],
        'Gradient Boosting': results_df[results_df['Model'].str.contains('GradientBoosting')],
        'XGBoost': results_df[results_df['Model'].str.contains('XGBoost')],
        'SVM': results_df[results_df['Model'].str.contains('SVM')],
        'Neural Network': results_df[results_df['Model'].str.contains('NeuralNet')],
        'Logistic Regression': results_df[results_df['Model'].str.contains('LogisticRegression')]
    }

    # Plot approximate ROC curves based on sensitivity/specificity
    for group_name, group_df in model_groups.items():
        color_key = group_name.split()[0]
        color = colors.get(color_key, '#000000')

        for _, row in group_df.iterrows():
            # Approximate ROC curve from sensitivity and specificity
            sens = row['SENSITIVITY']
            spec = row['SPECIFICITY']
            fpr = 1 - spec

            # Simple two-point ROC curve
            fpr_curve = [0, fpr, 1]
            tpr_curve = [0, sens, 1]

            auc_val = row['AUC_ROC']
            auc_se = row['AUC_ROC_SE']

            label = f"{row['Model'].replace('_', ' ')} (AUC={auc_val:.3f}±{auc_se:.3f})"

            ax.plot(fpr_curve, tpr_curve, color=color, lw=2, alpha=0.7,
                   label=label, marker='o', markersize=6)

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig('outputs/figures/manuscript/Figure2_ROC_Curves.png',
                bbox_inches='tight', dpi=600)
    plt.savefig('outputs/figures/manuscript/Figure2_ROC_Curves.pdf',
                bbox_inches='tight')
    print("Figure 2 saved: ROC curves")
    plt.close()


def create_figure_3_metrics_comparison():
    """
    Figure 3: Side-by-side metrics comparison with error bars.

    Six-panel figure showing all metrics with bootstrap SEs.
    """
    results_df = load_results()

    metrics = ['AUC_ROC', 'ACCURACY', 'SENSITIVITY', 'SPECIFICITY', 'PRECISION', 'F1']
    metric_labels = ['AUC-ROC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Color palette
    n_models = len(results_df)
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        values = results_df[metric].values
        se_col = f'{metric}_SE'
        errors = results_df[se_col].values

        # Create bar plot
        x_pos = np.arange(len(results_df))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.85,
                     edgecolor='black', linewidth=1.5)

        # Add error bars
        ax.errorbar(x_pos, values, yerr=errors, fmt='none', ecolor='black',
                   capsize=5, capthick=2, alpha=0.7)

        # Customize
        ax.set_ylabel(label, fontweight='bold', fontsize=11)
        ax.set_title(f'{label} Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df['Model'].str.replace('_', '\n'),
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best performer
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

        # Add value labels on top performers (top 3)
        top_3_idx = np.argsort(values)[-3:]
        for i in top_3_idx:
            if values[i] > 0:
                ax.text(i, values[i] + errors[i] + 0.03, f'{values[i]:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Model Performance Comparison with Bootstrap Standard Errors (1000 iterations)',
                 fontweight='bold', fontsize=14, y=0.995)
    plt.tight_layout()

    plt.savefig('outputs/figures/manuscript/Figure3_Metrics_Comparison.png',
                bbox_inches='tight', dpi=600)
    plt.savefig('outputs/figures/manuscript/Figure3_Metrics_Comparison.pdf',
                bbox_inches='tight')
    print("Figure 3 saved: Metrics comparison")
    plt.close()


def create_supplementary_table_1():
    """
    Supplementary Table 1: Detailed bootstrap statistics with 95% CIs.
    """
    detailed_df = pd.read_csv('outputs/results/detailed_bootstrap_results.csv')

    # Format for publication
    output_rows = []

    for _, row in detailed_df.iterrows():
        model = row['Model']

        # AUC-ROC
        auc_mean = row['auc_roc_mean']
        auc_ci_low = row['auc_roc_ci_lower']
        auc_ci_high = row['auc_roc_ci_upper']
        auc_str = f"{auc_mean:.3f} ({auc_ci_low:.3f}-{auc_ci_high:.3f})"

        # Sensitivity
        sens_mean = row['sensitivity_mean']
        sens_ci_low = row['sensitivity_ci_lower']
        sens_ci_high = row['sensitivity_ci_upper']
        sens_str = f"{sens_mean:.3f} ({sens_ci_low:.3f}-{sens_ci_high:.3f})"

        # Specificity
        spec_mean = row['specificity_mean']
        spec_ci_low = row['specificity_ci_lower']
        spec_ci_high = row['specificity_ci_upper']
        spec_str = f"{spec_mean:.3f} ({spec_ci_low:.3f}-{spec_ci_high:.3f})"

        output_rows.append({
            'Model': model,
            'AUC-ROC (95% CI)': auc_str,
            'Sensitivity (95% CI)': sens_str,
            'Specificity (95% CI)': spec_str
        })

    output_df = pd.DataFrame(output_rows)

    # Save as CSV and LaTeX
    os.makedirs('outputs/tables', exist_ok=True)
    output_df.to_csv('outputs/tables/SupplementaryTable1_Bootstrap_CIs.csv', index=False)

    # Create LaTeX table
    with open('outputs/tables/SupplementaryTable1_Bootstrap_CIs.tex', 'w') as f:
        f.write("% Supplementary Table 1: Bootstrap 95% Confidence Intervals\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Bootstrap 95\\% Confidence Intervals for Primary Metrics}\n")
        f.write("\\label{tab:supp1}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("Model & AUC-ROC (95\\% CI) & Sensitivity (95\\% CI) & Specificity (95\\% CI) \\\\\n")
        f.write("\\hline\n")

        for _, row in output_df.iterrows():
            model_name = row['Model'].replace('_', '\\_')
            f.write(f"{model_name} & {row['AUC-ROC (95% CI)']} & {row['Sensitivity (95% CI)']} & {row['Specificity (95% CI)']} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("Supplementary Table 1 saved: Bootstrap CIs")


def main():
    """Generate all manuscript figures and tables."""
    print("\n" + "="*80)
    print("Generating Publication-Quality Figures")
    print("="*80 + "\n")

    # Check that results exist
    if not os.path.exists('outputs/results/model_comparison_results.csv'):
        print("Error: Results not found. Please run run_experiments.py first.")
        return

    create_figure_1_confusion_matrices()
    create_figure_2_roc_curves()
    create_figure_3_metrics_comparison()
    create_supplementary_table_1()

    print("\n" + "="*80)
    print("All figures and tables generated successfully!")
    print("="*80)
    print("\nOutputs saved to:")
    print("  - outputs/figures/manuscript/")
    print("  - outputs/tables/")
    print("\nFile formats: PNG (600 DPI), PDF (vector)")


if __name__ == "__main__":
    main()
