# Fall Prediction Using Gait Analysis

A comprehensive machine learning framework for predicting fall risk in elderly individuals using gait variability and anthropometric data.

## Overview

This repository implements a systematic comparison of multiple machine learning models for fall risk prediction, with rigorous evaluation using bootstrap standard errors and comprehensive performance metrics.

**Dataset**: 171 participants (34 Fallers, 137 Non-Fallers)
**Features**: 61 gait parameters + anthropometry (Age, Height, Weight)
**Challenge**: Imbalanced dataset (~20% positive class)

## Repository Structure

```
fallprediction/
├── data/                           # Dataset files
│   └── combined_output.csv         # Main dataset (171 samples, 63 features)
│
├── src/                            # Source code modules
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── model_evaluation.py         # Model evaluation with bootstrap SE
│   └── visualization.py            # Results visualization
│
├── experiments/                    # Jupyter notebooks (exploratory work)
│   ├── fall_prediction_analysis.ipynb
│   ├── random_forest_experiments.ipynb
│   ├── ml_models_comparison.ipynb
│   └── [experiment outputs]
│
├── outputs/                        # Generated results (gitignored)
│   ├── results/                    # CSV result tables
│   │   ├── model_comparison_results.csv
│   │   └── detailed_bootstrap_results.csv
│   └── figures/                    # Visualizations
│       ├── roc_curves.png
│       ├── metrics_comparison.png
│       ├── confusion_matrices.png
│       └── comprehensive_comparison.png
│
├── run_experiments.py              # Main experimentation script
├── README.md                       # This file
└── EXPERIMENTS_README.md           # Detailed methodology documentation
```

## Quick Start

### Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

### Running Experiments

```bash
# Run comprehensive model comparison
python run_experiments.py
```

This will:
1. Load and preprocess the data
2. Train 13 different model configurations
3. Evaluate with bootstrap standard errors (1000 iterations)
4. Generate comparison tables and visualizations
5. Save results to `outputs/`

## Models Evaluated

### Ensemble Methods
- **Random Forest** (3 configurations)
  - Default (100 trees) - OOB Score: 0.8203
  - High trees (500 trees) - OOB Score: 0.8047
  - Tuned (RandomizedSearchCV)

- **Gradient Boosting** (2 configurations)
  - Default
  - Tuned (RandomizedSearchCV)

- **XGBoost** (2 configurations)
  - Default
  - Tuned (RandomizedSearchCV)

### Support Vector Machines
- **SVM** (2 configurations)
  - RBF kernel (GridSearchCV)
  - Linear kernel (GridSearchCV)

### Neural Networks
- **MLP Classifier** (2 configurations)
  - Simple (single hidden layer)
  - Tuned (GridSearchCV)

### Linear Models
- **Logistic Regression** (2 configurations)
  - L2 regularization
  - Tuned (GridSearchCV)

## Performance Summary

### Top Performing Models

**Best AUC-ROC**: RF_500trees (0.6412 ± 0.0942)
**Best Accuracy**: LogisticRegression_Tuned (0.7932 ± 0.0623)
**Best Sensitivity**: NeuralNet_Tuned (0.3384 ± 0.1671)

### Key Findings

1. **Class Imbalance Challenge**: Most models struggle with sensitivity due to severe class imbalance (20% Fallers)
2. **High Specificity**: Models achieve good specificity (80-100%) but at the cost of sensitivity
3. **Random Forest Performance**: RF models show strong OOB scores but conservative predictions
4. **Tuned Gradient Boosting**: Best balance with 0.5639 AUC-ROC and reasonable sensitivity (0.3316)

## Metrics Reported

All metrics include **bootstrap standard errors** (1000 iterations):

- **AUC-ROC**: Area Under the ROC Curve (primary metric)
- **Accuracy**: Overall classification accuracy
- **Sensitivity**: True Positive Rate (Recall for Fallers)
- **Specificity**: True Negative Rate
- **Precision**: Positive Predictive Value
- **F1 Score**: Harmonic mean of Precision and Recall
- **OOB Score**: Out-of-Bag score (for Random Forest models)

### Bootstrap Standard Errors

Standard errors are computed via bootstrap resampling:
- 1000 bootstrap iterations
- Sampling with replacement
- 95% confidence intervals calculated

## Key Features

### Data Processing
- Automatic NaN handling (median imputation)
- Stratified train-test split (75/25)
- Feature standardization (StandardScaler)
- Class distribution monitoring

### Model Training
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Cross-validation (5-fold stratified)
- OOB scoring for ensemble methods
- Multiple random seeds for robustness

### Evaluation
- Comprehensive metrics with bootstrap SE
- ROC curve analysis
- Confusion matrix visualization
- Side-by-side model comparison

## Visualization Outputs

The framework generates 4 comprehensive visualizations:

1. **ROC Curves**: All models overlaid with AUC ± SE
2. **Metrics Comparison**: Bar charts with error bars for all metrics
3. **Confusion Matrices**: Heatmaps for each model
4. **Comprehensive Comparison**: 4-panel figure with:
   - ROC curves
   - AUC-ROC bars
   - Sensitivity vs Specificity
   - All metrics grouped

## Dataset Features

### Linear Variables (20 features)
- **Gait timing**: GCTime, RSST, LSST, RSwT, LSwT, DST, StepTime
- **Accelerations**: RMS_AP, RMS_V, RMS_ML, RMSR_AP, RMSR_ML, RMSR_V
- **Velocity**: Velocity, Time2FirstQuartile_Velocity, Time2Median_Velocity, Time2ThirdQuartile_Velocity
- **Anthropometry**: Age, Height, Weight

### Nonlinear Variables (41 features)
- **Variability**: SD and CV for timing parameters (14 features)
- **Harmony/Regularity**: Indices (3 features)
- **Multiscale Entropy**: MSE metrics (8 features)
- **Recurrence Quantification**: RQA metrics (16 features)

## Usage Examples

### Basic Usage

```python
from src.data_loader import load_and_prepare_data
from src.model_evaluation import evaluate_model
from sklearn.ensemble import RandomForestClassifier

# Load data
data = load_and_prepare_data(test_size=0.25, random_state=42)

# Train model
model = RandomForestClassifier(oob_score=True, random_state=42)
model.fit(data['X_train'], data['y_train'])

# Evaluate with bootstrap SE
results = evaluate_model(
    model=model,
    X_train=data['X_train'],
    y_train=data['y_train'],
    X_test=data['X_test'],
    y_test=data['y_test'],
    model_name="My_RF",
    use_oob=True,
    n_bootstrap=1000
)
```

### Custom Experiments

```python
# Modify run_experiments.py or create your own script
# All modules are in src/ for easy importing
from src.data_loader import FallDataLoader
from src.model_evaluation import bootstrap_metrics, format_results_table
from src.visualization import save_all_visualizations
```

## Results Files

After running experiments, the following files are generated:

### CSV Results
- `outputs/results/model_comparison_results.csv`: Summary table with all metrics ± SE
- `outputs/results/detailed_bootstrap_results.csv`: Full bootstrap statistics with CIs
- `outputs/experiment_log.txt`: Complete execution log

### Figures
- `outputs/figures/roc_curves.png`: ROC curves comparison
- `outputs/figures/metrics_comparison.png`: Metrics bar charts with error bars
- `outputs/figures/confusion_matrices.png`: All confusion matrices
- `outputs/figures/comprehensive_comparison.png`: 4-panel comprehensive figure

## Future Improvements

1. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling)
   - Class weights optimization
   - Threshold tuning for sensitivity/specificity trade-off

2. **Feature Engineering**
   - PCA for dimensionality reduction
   - Feature importance analysis
   - Recursive feature elimination

3. **Model Enhancements**
   - Ensemble stacking
   - Calibrated classifiers
   - Cost-sensitive learning

4. **Validation**
   - External validation set
   - Temporal validation
   - Cross-site validation

## Citation

If you use this code or dataset, please cite the original data sources and this repository.

## License

[Specify license here]

## Contact

For questions or issues, please open an issue on the repository.

---

**Last Updated**: 2025-11-10
**Version**: 1.0.0
