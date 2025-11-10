#!/usr/bin/env python3
"""
Comprehensive Fall Prediction Experimentation Script

This script runs multiple machine learning models with various hyperparameters,
computes comprehensive metrics with bootstrap standard errors, and generates
detailed comparison reports and visualizations.

Usage:
    python run_experiments.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
sys.path.insert(0, 'src')
from data_loader import load_and_prepare_data
from model_evaluation import evaluate_model, format_results_table, print_results_summary
from visualization import save_all_visualizations

# Import sklearn models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

# Import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Skipping XGBoost models.")


def create_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    print("Output directories created/verified.")


def train_random_forest_models(X_train, y_train, random_state=42):
    """
    Train Random Forest models with different configurations.

    Returns:
    --------
    list of (model, model_name, use_oob) tuples
    """
    print("\n" + "="*80)
    print("Training Random Forest Models")
    print("="*80)

    models = []

    # Configuration 1: Default RF
    print("\n1. Training Random Forest (Default)...")
    rf_default = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        oob_score=True,
        n_jobs=-1
    )
    rf_default.fit(X_train, y_train)
    models.append((rf_default, "RF_Default", True))
    print(f"   OOB Score: {rf_default.oob_score_:.4f}")

    # Configuration 2: High trees with OOB
    print("\n2. Training Random Forest (500 trees)...")
    rf_high = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        oob_score=True,
        n_jobs=-1
    )
    rf_high.fit(X_train, y_train)
    models.append((rf_high, "RF_500trees", True))
    print(f"   OOB Score: {rf_high.oob_score_:.4f}")

    # Configuration 3: Tuned RF with RandomizedSearchCV
    print("\n3. Training Random Forest (RandomizedSearchCV)...")
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf_random = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, oob_score=True, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    rf_random.fit(X_train, y_train)
    print(f"   Best params: {rf_random.best_params_}")
    print(f"   Best CV score: {rf_random.best_score_:.4f}")
    models.append((rf_random.best_estimator_, "RF_Tuned", True))

    return models


def train_gradient_boosting_models(X_train, y_train, X_test, y_test, random_state=42):
    """Train Gradient Boosting models."""
    print("\n" + "="*80)
    print("Training Gradient Boosting Models")
    print("="*80)

    models = []

    # Configuration 1: Default GB
    print("\n1. Training Gradient Boosting (Default)...")
    gb_default = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=random_state
    )
    gb_default.fit(X_train, y_train)
    models.append((gb_default, "GradientBoosting_Default", False))

    # Configuration 2: Tuned GB
    print("\n2. Training Gradient Boosting (RandomizedSearchCV)...")
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.8, 1.0]
    }

    gb_random = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    gb_random.fit(X_train, y_train)
    print(f"   Best params: {gb_random.best_params_}")
    print(f"   Best CV score: {gb_random.best_score_:.4f}")
    models.append((gb_random.best_estimator_, "GradientBoosting_Tuned", False))

    return models


def train_xgboost_models(X_train, y_train, X_test, y_test, random_state=42):
    """Train XGBoost models."""
    if not XGBOOST_AVAILABLE:
        return []

    print("\n" + "="*80)
    print("Training XGBoost Models")
    print("="*80)

    models = []

    # Configuration 1: Default XGBoost
    print("\n1. Training XGBoost (Default)...")
    xgb_default = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_default.fit(X_train, y_train)
    models.append((xgb_default, "XGBoost_Default", False))

    # Configuration 2: Tuned XGBoost
    print("\n2. Training XGBoost (RandomizedSearchCV)...")
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 2, 5]
    }

    xgb_random = RandomizedSearchCV(
        XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False),
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    xgb_random.fit(X_train, y_train)
    print(f"   Best params: {xgb_random.best_params_}")
    print(f"   Best CV score: {xgb_random.best_score_:.4f}")
    models.append((xgb_random.best_estimator_, "XGBoost_Tuned", False))

    return models


def train_svm_models(X_train, y_train, X_test, y_test, random_state=42):
    """Train SVM models with different kernels."""
    print("\n" + "="*80)
    print("Training Support Vector Machine Models")
    print("="*80)

    models = []

    # Configuration 1: SVM RBF
    print("\n1. Training SVM (RBF kernel, GridSearchCV)...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'class_weight': ['balanced', None]
    }

    svm_rbf = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    svm_rbf.fit(X_train, y_train)
    print(f"   Best params: {svm_rbf.best_params_}")
    print(f"   Best CV score: {svm_rbf.best_score_:.4f}")
    models.append((svm_rbf.best_estimator_, "SVM_RBF", False))

    # Configuration 2: SVM Linear
    print("\n2. Training SVM (Linear kernel)...")
    param_grid_linear = {
        'C': [0.1, 1, 10, 100],
        'class_weight': ['balanced', None]
    }

    svm_linear = GridSearchCV(
        SVC(kernel='linear', probability=True, random_state=random_state),
        param_grid=param_grid_linear,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    svm_linear.fit(X_train, y_train)
    print(f"   Best params: {svm_linear.best_params_}")
    print(f"   Best CV score: {svm_linear.best_score_:.4f}")
    models.append((svm_linear.best_estimator_, "SVM_Linear", False))

    return models


def train_neural_network_models(X_train, y_train, X_test, y_test, random_state=42):
    """Train Neural Network models."""
    print("\n" + "="*80)
    print("Training Neural Network Models")
    print("="*80)

    models = []

    # Configuration 1: Simple MLP
    print("\n1. Training Neural Network (Simple)...")
    mlp_simple = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=random_state,
        early_stopping=True
    )
    mlp_simple.fit(X_train, y_train)
    models.append((mlp_simple, "NeuralNet_Simple", False))

    # Configuration 2: Tuned MLP
    print("\n2. Training Neural Network (GridSearchCV)...")
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }

    mlp_tuned = GridSearchCV(
        MLPClassifier(solver='adam', max_iter=500, random_state=random_state, early_stopping=True),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    mlp_tuned.fit(X_train, y_train)
    print(f"   Best params: {mlp_tuned.best_params_}")
    print(f"   Best CV score: {mlp_tuned.best_score_:.4f}")
    models.append((mlp_tuned.best_estimator_, "NeuralNet_Tuned", False))

    return models


def train_logistic_regression_models(X_train, y_train, X_test, y_test, random_state=42):
    """Train Logistic Regression models."""
    print("\n" + "="*80)
    print("Training Logistic Regression Models")
    print("="*80)

    models = []

    # Configuration 1: L2 regularization
    print("\n1. Training Logistic Regression (L2)...")
    lr_l2 = LogisticRegression(
        penalty='l2',
        C=1.0,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    lr_l2.fit(X_train, y_train)
    models.append((lr_l2, "LogisticRegression_L2", False))

    # Configuration 2: Tuned with GridSearchCV
    print("\n2. Training Logistic Regression (GridSearchCV)...")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'class_weight': ['balanced', None]
    }

    lr_tuned = GridSearchCV(
        LogisticRegression(random_state=random_state, max_iter=1000, solver='lbfgs'),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    lr_tuned.fit(X_train, y_train)
    print(f"   Best params: {lr_tuned.best_params_}")
    print(f"   Best CV score: {lr_tuned.best_score_:.4f}")
    models.append((lr_tuned.best_estimator_, "LogisticRegression_Tuned", False))

    return models


def main():
    """Main execution function."""
    print("\n" + "="*100)
    print("FALL PREDICTION - COMPREHENSIVE MODEL COMPARISON".center(100))
    print("="*100)
    print(f"\nExperiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories
    create_output_dirs()

    # Load and prepare data
    print("\n" + "="*80)
    print("Loading and Preparing Data")
    print("="*80)

    data = load_and_prepare_data(
        data_path='data/combined_output.csv',
        test_size=0.25,
        random_state=42,
        scale=True,
        stratify=True
    )

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    print(f"\nClass distribution: {data['class_distribution']}")

    # Train all models
    all_models = []

    all_models.extend(train_random_forest_models(X_train, y_train, X_test, y_test))
    all_models.extend(train_gradient_boosting_models(X_train, y_train, X_test, y_test))
    all_models.extend(train_xgboost_models(X_train, y_train, X_test, y_test))
    all_models.extend(train_svm_models(X_train, y_train, X_test, y_test))
    all_models.extend(train_neural_network_models(X_train, y_train, X_test, y_test))
    all_models.extend(train_logistic_regression_models(X_train, y_train, X_test, y_test))

    # Evaluate all models
    print("\n" + "="*80)
    print("Evaluating Models with Bootstrap Standard Errors")
    print("="*80)

    results_list = []
    for model, model_name, use_oob in all_models:
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            use_oob=use_oob,
            n_bootstrap=1000,
            random_state=42
        )
        results_list.append(result)

    # Format results table
    results_df = format_results_table(results_list)

    # Print results summary
    print_results_summary(results_df, title="COMPREHENSIVE MODEL COMPARISON RESULTS")

    # Save results to CSV
    results_path = 'outputs/results/model_comparison_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Generate visualizations
    save_all_visualizations(results_df, results_list, output_dir='outputs/figures')

    # Save detailed results with bootstrap CIs
    detailed_results = []
    for result in results_list:
        row = {'Model': result['model_name']}
        for metric, stats in result['bootstrap_stats'].items():
            row[f'{metric}_mean'] = stats['mean']
            row[f'{metric}_se'] = stats['se']
            row[f'{metric}_ci_lower'] = stats['ci_lower']
            row[f'{metric}_ci_upper'] = stats['ci_upper']
        detailed_results.append(row)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = 'outputs/results/detailed_bootstrap_results.csv'
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed bootstrap results saved to: {detailed_path}")

    print("\n" + "="*100)
    print(f"Experiment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    return results_df, results_list


if __name__ == "__main__":
    results_df, results_list = main()
