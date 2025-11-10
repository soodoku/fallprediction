#!/usr/bin/env python3
"""
Comprehensive Fall Prediction Comparison: Baseline vs. Extensions

This script:
1. Replicates Soangra et al. (Nature 2021) baseline exactly
2. Runs our extensions (more models, better tuning, class imbalance handling)
3. Compares results side-by-side

Usage:
    python run_baseline_comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_and_prepare_data
from src.pca_features import GaitFeaturePCA
from src.model_evaluation import evaluate_model, format_results_table, print_results_summary
from src.visualization import save_all_visualizations

# Import sklearn models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")


def create_output_dirs():
    """Create output directories."""
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/baseline_comparison', exist_ok=True)


def run_soangra_baseline(X_train_raw, y_train, X_test_raw, y_test, random_state=42):
    """
    Replicate Soangra et al. (Nature 2021) Experiment III exactly.

    Their approach:
    - PCA: 4 linear + 26 nonlinear PCs (99% variance)
    - Random Forest: 365 trees, 1 feature per split
    - 10 runs with different random seeds
    - Report mean ± SE across runs

    Returns:
    --------
    dict with results from 10 runs
    """
    print("\n" + "="*80)
    print("BASELINE: Replicating Soangra et al. (Nature 2021) - Experiment III")
    print("="*80)

    # Step 1: PCA feature engineering
    print("\nStep 1: PCA Feature Engineering (99% variance threshold)")
    pca_transformer = GaitFeaturePCA(variance_threshold=0.99, n_iterations=1000)

    # Fit PCA on training data
    X_train_pca_full = pca_transformer.fit_transform(X_train_raw, y_train)
    X_test_pca_full = pca_transformer.transform(X_test_raw)

    optimal_pcs = pca_transformer.get_optimal_n_pcs()
    print(f"\nPCA results:")
    print(f"  Linear PCs: {optimal_pcs['linear']}")
    print(f"  Nonlinear PCs: {optimal_pcs['nonlinear']}")

    # Step 2: Select specific number of PCs (Soangra used 4 linear + 26 nonlinear)
    # Note: Their paper used an "elbow method" to select 4 linear PCs
    # We'll use their final configuration directly
    n_linear_pcs = min(4, optimal_pcs['linear'])
    n_nonlinear_pcs = min(26, optimal_pcs['nonlinear'])

    print(f"\nUsing Soangra et al.'s configuration:")
    print(f"  {n_linear_pcs} linear PCs + {n_nonlinear_pcs} nonlinear PCs")

    X_train_pca = pca_transformer.transform(X_train_raw, n_linear_pcs, n_nonlinear_pcs)
    X_test_pca = pca_transformer.transform(X_test_raw, n_linear_pcs, n_nonlinear_pcs)

    print(f"  Final feature dimensions: {X_train_pca.shape[1]} PCs")

    # Step 3: Train RF with their exact configuration
    print("\nStep 2: Training Random Forest (Soangra et al. configuration)")
    print("  Architecture: 365 trees, max_features=1 (1 feature per split)")
    print("  Running 10 iterations with different random seeds...")

    # Their exact RF configuration
    rf_config = {
        'n_estimators': 365,
        'max_features': 1,  # Only 1 feature per split
        'oob_score': True,
        'n_jobs': -1,
        'random_state': None  # Will be set in loop
    }

    # Run 10 times with different seeds (as they did)
    results_per_seed = []
    oob_scores = []

    for seed in range(10):
        rf_config['random_state'] = seed + random_state

        rf_model = RandomForestClassifier(**rf_config)
        rf_model.fit(X_train_pca, y_train)

        # Evaluate on test set
        y_pred = rf_model.predict(X_test_pca)
        y_proba = rf_model.predict_proba(X_test_pca)[:, 1]

        # Calculate metrics (without bootstrap, just point estimates)
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'auc_roc': roc_auc_score(y_test, y_proba),
            'oob_score': rf_model.oob_score_
        }

        results_per_seed.append(metrics)
        oob_scores.append(rf_model.oob_score_)

        print(f"    Seed {seed}: Acc={metrics['accuracy']:.3f}, "
              f"Sens={metrics['sensitivity']:.3f}, "
              f"Spec={metrics['specificity']:.3f}, "
              f"AUC={metrics['auc_roc']:.3f}, "
              f"OOB={metrics['oob_score']:.3f}")

    # Aggregate across seeds (mean ± SE)
    print("\n" + "-"*80)
    print("BASELINE RESULTS (Mean ± SE from 10 runs):")
    print("-"*80)

    baseline_results = {}
    for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'auc_roc', 'oob_score']:
        values = [r[metric] for r in results_per_seed]
        baseline_results[metric] = {
            'mean': np.mean(values),
            'se': np.std(values),  # Standard error (SD of means across runs)
            'values': values
        }

        print(f"{metric.upper():15s}: {baseline_results[metric]['mean']:.3f} ± "
              f"{baseline_results[metric]['se']:.3f}")

    print("-"*80)

    # Store additional info
    baseline_results['model_name'] = 'Soangra_Baseline_RF'
    baseline_results['n_linear_pcs'] = n_linear_pcs
    baseline_results['n_nonlinear_pcs'] = n_nonlinear_pcs
    baseline_results['pca_transformer'] = pca_transformer

    return baseline_results


def run_pca_with_tuned_rf(X_train_raw, y_train, X_test_raw, y_test, pca_transformer,
                          n_linear_pcs, n_nonlinear_pcs, random_state=42):
    """
    Use PCA (like baseline) but with optimized RF hyperparameters.

    This tests: Is their RF configuration (365 trees, 1 feature/split) optimal?
    """
    print("\n" + "="*80)
    print("EXTENSION 1: PCA + Optimized Random Forest")
    print("="*80)

    X_train_pca = pca_transformer.transform(X_train_raw, n_linear_pcs, n_nonlinear_pcs)
    X_test_pca = pca_transformer.transform(X_test_raw, n_linear_pcs, n_nonlinear_pcs)

    print(f"\nUsing PCA features: {n_linear_pcs} linear + {n_nonlinear_pcs} nonlinear = {X_train_pca.shape[1]} PCs")

    # Comprehensive hyperparameter search
    print("\nOptimizing RF hyperparameters via RandomizedSearchCV...")

    param_dist = {
        'n_estimators': [100, 200, 365, 500],  # Include their 365
        'max_features': [1, 'sqrt', 'log2', None],  # Include their max_features=1
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, oob_score=True, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    rf_search.fit(X_train_pca, y_train)

    print(f"\nBest hyperparameters: {rf_search.best_params_}")
    print(f"Best CV AUC-ROC: {rf_search.best_score_:.4f}")

    # Evaluate on test set
    result = evaluate_model(
        model=rf_search.best_estimator_,
        X_train=X_train_pca.values,
        y_train=y_train,
        X_test=X_test_pca.values,
        y_test=y_test,
        model_name="PCA_RF_Tuned",
        use_oob=True,
        n_bootstrap=1000,
        random_state=random_state
    )

    return result, rf_search.best_estimator_


def run_raw_features_comparison(X_train_raw, y_train, X_test_raw, y_test,
                                random_state=42):
    """
    Compare models using raw features (our original approach).

    Tests: How much does PCA help vs. using raw features?
    """
    print("\n" + "="*80)
    print("EXTENSION 2: Raw Features + Multiple Algorithms")
    print("="*80)

    from sklearn.preprocessing import StandardScaler

    # Scale raw features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    results = []

    # 1. Random Forest (tuned)
    print("\n1. Random Forest (Raw Features, Tuned)...")
    param_dist_rf = {
        'n_estimators': [100, 200, 300, 500],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, oob_score=True, n_jobs=-1),
        param_dist_rf, n_iter=30, cv=5, scoring='roc_auc',
        random_state=random_state, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train_scaled, y_train)

    rf_result = evaluate_model(
        rf_search.best_estimator_, X_train_scaled, y_train,
        X_test_scaled, y_test, "RF_Raw_Tuned", use_oob=True,
        n_bootstrap=1000, random_state=random_state
    )
    results.append(rf_result)
    print(f"   Best params: {rf_search.best_params_}")

    # 2. Gradient Boosting
    print("\n2. Gradient Boosting...")
    param_dist_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        param_dist_gb, n_iter=20, cv=5, scoring='roc_auc',
        random_state=random_state, n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_scaled, y_train)

    gb_result = evaluate_model(
        gb_search.best_estimator_, X_train_scaled, y_train,
        X_test_scaled, y_test, "GradientBoosting_Raw",
        n_bootstrap=1000, random_state=random_state
    )
    results.append(gb_result)

    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n3. XGBoost...")
        param_dist_xgb = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'scale_pos_weight': [1, 3, 5]
        }

        xgb_search = RandomizedSearchCV(
            XGBClassifier(random_state=random_state, eval_metric='logloss',
                         use_label_encoder=False),
            param_dist_xgb, n_iter=20, cv=5, scoring='roc_auc',
            random_state=random_state, n_jobs=-1, verbose=0
        )
        xgb_search.fit(X_train_scaled, y_train)

        xgb_result = evaluate_model(
            xgb_search.best_estimator_, X_train_scaled, y_train,
            X_test_scaled, y_test, "XGBoost_Raw",
            n_bootstrap=1000, random_state=random_state
        )
        results.append(xgb_result)

    return results


def run_smote_experiments(X_train_raw, y_train, X_test_raw, y_test, pca_transformer,
                          n_linear_pcs, n_nonlinear_pcs, random_state=42):
    """
    Test class imbalance handling with SMOTE.

    Tests: Does SMOTE improve sensitivity?
    """
    print("\n" + "="*80)
    print("EXTENSION 3: PCA + SMOTE + Random Forest")
    print("="*80)

    X_train_pca = pca_transformer.transform(X_train_raw, n_linear_pcs, n_nonlinear_pcs)
    X_test_pca = pca_transformer.transform(X_test_raw, n_linear_pcs, n_nonlinear_pcs)

    print(f"\nApplying SMOTE to balance training data...")
    print(f"  Original: Fallers={y_train.sum()}, Non-fallers={len(y_train)-y_train.sum()}")

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)

    print(f"  After SMOTE: Fallers={y_train_smote.sum()}, Non-fallers={len(y_train_smote)-y_train_smote.sum()}")

    # Train RF on SMOTE data
    print("\nTraining Random Forest on SMOTE-balanced data...")

    rf_smote = RandomForestClassifier(
        n_estimators=365,  # Use baseline config
        max_features=1,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )

    rf_smote.fit(X_train_smote, y_train_smote)

    # Evaluate on original (non-SMOTE) test set
    result = evaluate_model(
        model=rf_smote,
        X_train=X_train_smote,
        y_train=y_train_smote,
        X_test=X_test_pca.values,
        y_test=y_test,
        model_name="PCA_RF_SMOTE",
        use_oob=True,
        n_bootstrap=1000,
        random_state=random_state
    )

    return result


def main():
    """Main execution function."""
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: Baseline (Soangra et al.) vs. Extensions".center(100))
    print("="*100)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    create_output_dirs()

    # Load data (raw features, no PCA yet)
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)

    data = load_and_prepare_data(
        data_path='data/processed/combined_output.csv',
        test_size=0.25,
        random_state=42,
        scale=False,  # We'll handle scaling separately for PCA vs raw
        stratify=True
    )

    X_train_raw = data['X_train_raw']
    X_test_raw = data['X_test_raw']
    y_train = data['y_train']
    y_test = data['y_test']

    print(f"\nData loaded: Train={len(y_train)}, Test={len(y_test)}")
    print(f"Features: {X_train_raw.shape[1]}")

    # =========================================================================
    # BASELINE: Soangra et al. exact replication
    # =========================================================================

    baseline_results = run_soangra_baseline(
        X_train_raw, y_train, X_test_raw, y_test, random_state=42
    )

    # =========================================================================
    # EXTENSIONS
    # =========================================================================

    all_results = []

    # Extension 1: PCA + Tuned RF
    pca_rf_result, _ = run_pca_with_tuned_rf(
        X_train_raw, y_train, X_test_raw, y_test,
        baseline_results['pca_transformer'],
        baseline_results['n_linear_pcs'],
        baseline_results['n_nonlinear_pcs'],
        random_state=42
    )
    all_results.append(pca_rf_result)

    # Extension 2: Raw features + multiple algorithms
    raw_results = run_raw_features_comparison(
        X_train_raw, y_train, X_test_raw, y_test, random_state=42
    )
    all_results.extend(raw_results)

    # Extension 3: SMOTE
    smote_result = run_smote_experiments(
        X_train_raw, y_train, X_test_raw, y_test,
        baseline_results['pca_transformer'],
        baseline_results['n_linear_pcs'],
        baseline_results['n_nonlinear_pcs'],
        random_state=42
    )
    all_results.append(smote_result)

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================

    print("\n" + "="*100)
    print("FINAL COMPARISON: Baseline vs. Extensions".center(100))
    print("="*100)

    # Create comparison table
    comparison_data = []

    # Add baseline
    comparison_data.append({
        'Model': 'Soangra_Baseline',
        'Approach': 'PCA + Fixed RF (365trees, max_feat=1)',
        'Accuracy': f"{baseline_results['accuracy']['mean']:.3f} ± {baseline_results['accuracy']['se']:.3f}",
        'Sensitivity': f"{baseline_results['sensitivity']['mean']:.3f} ± {baseline_results['sensitivity']['se']:.3f}",
        'Specificity': f"{baseline_results['specificity']['mean']:.3f} ± {baseline_results['specificity']['se']:.3f}",
        'AUC-ROC': f"{baseline_results['auc_roc']['mean']:.3f} ± {baseline_results['auc_roc']['se']:.3f}",
        'OOB': f"{baseline_results['oob_score']['mean']:.3f}",
        'Features': f"{baseline_results['n_linear_pcs']}L + {baseline_results['n_nonlinear_pcs']}NL PCs"
    })

    # Add extensions
    for result in all_results:
        bs = result['bootstrap_stats']
        comparison_data.append({
            'Model': result['model_name'],
            'Approach': 'Our Extension',
            'Accuracy': f"{bs['accuracy']['mean']:.3f} ± {bs['accuracy']['se']:.3f}",
            'Sensitivity': f"{bs['sensitivity']['mean']:.3f} ± {bs['sensitivity']['se']:.3f}",
            'Specificity': f"{bs['specificity']['mean']:.3f} ± {bs['specificity']['se']:.3f}",
            'AUC-ROC': f"{bs['auc_roc']['mean']:.3f} ± {bs['auc_roc']['se']:.3f}",
            'OOB': f"{result['oob_score']:.3f}" if result['oob_score'] else 'N/A',
            'Features': 'PCA' if 'PCA' in result['model_name'] else 'Raw (61)'
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # Save results
    output_path = 'outputs/baseline_comparison/comparison_results.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "="*100)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    return baseline_results, all_results


if __name__ == "__main__":
    baseline, extensions = main()
