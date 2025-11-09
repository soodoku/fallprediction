# Random Forest Experiments for Fall Prediction

This repository contains implementations of three random forest experiments for fall risk prediction using gait analysis data, based on the research methodology described in the paper.

## Overview

The experiments progressively improve a fall prediction model through:
1. **Experiment I**: Base Random Forest models with separate linear and nonlinear gait variables
2. **Experiment II**: Feature engineering using variance selection and Principal Component Analysis (PCA)
3. **Experiment III**: Combined model with optimal feature selection via elbow point analysis

## Dataset

- **Total participants**: 171 (127 training, 44 testing)
- **Features**: 61 gait parameters + anthropometry
  - **Linear variables (20)**: Mean timing values, velocity measures, RMS accelerations, anthropometry
  - **Nonlinear variables (41)**: Variability measures (CV, SD), Multiscale Entropy, Recurrence Quantification Analysis, Harmony/Regularity indices
- **Target**: Binary classification (Faller vs Non-Faller)
- **Class distribution**:
  - Training: 29 Fallers, 98 Non-Fallers
  - Testing: 5 Fallers, 39 Non-Fallers

## Implementation Details

### Experiment I: Base Random Forest Model

**Configuration:**
- 365 trees
- 1 feature at each split
- 10 runs with different random seeds for robust error estimation

**Results:**

| Variable Type | Accuracy | Sensitivity | Specificity |
|--------------|----------|-------------|-------------|
| Linear (20 vars) | 88.6 ± 0.0% | 0.0 ± 0.0% | 100.0 ± 0.0% |
| Nonlinear (41 vars) | 84.5 ± 1.0% | 2.0 ± 6.3% | 95.1 ± 0.8% |

### Experiment II: Feature Engineering with PCA

**Steps:**
1. Unsupervised feature selection via variance threshold
2. PCA transformation (99% variance explained)
3. RF training on principal components

**PCA Results:**
- Linear variables: 20 → 10 PCs (99.42% variance explained)
- Nonlinear variables: 41 → 26 PCs (99.03% variance explained)

**Results:**

| Feature Set | Accuracy | Sensitivity | Specificity |
|-------------|----------|-------------|-------------|
| 10 Linear PCs | 89.3 ± 1.1% | 6.0 ± 9.7% | 100.0 ± 0.0% |
| 26 Nonlinear PCs | 87.0 ± 1.1% | 0.0 ± 0.0% | 98.2 ± 1.2% |

### Experiment III: Combined Model with Elbow Point Analysis

**Methodology:**
- Start with 26 nonlinear PCs (base)
- Gradually add linear PCs (0 to 10)
- Evaluate using Out-of-Bag (OOB) score and AUC
- Identify elbow point for optimal feature combination

**Elbow Point Results:**
- **Optimal configuration**: 26 nonlinear PCs + 4 linear PCs = 30 total PCs
- **Performance**:
  - Accuracy: 87.7 ± 1.2%
  - Sensitivity: 0.0 ± 0.0%
  - Specificity: 99.0 ± 1.3%
  - OOB Score: 76.4%
  - AUC: 0.656

## Files and Outputs

### Main Files
- `random_forest_experiments.ipynb` - Complete Jupyter notebook with all three experiments
- `data/combined_output.csv` - Input dataset (171 participants, 63 columns)

### Generated Outputs
- `experiments_summary.csv` - Summary table of all experiment results
- `experiment_III_elbow_plot.png` - Elbow curves (OOB score and AUC vs. number of linear PCs)
- `experiments_comparison.png` - Bar chart comparing all experiments
- `roc_curve_best_model.png` - ROC curve for the best model (Experiment III)

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Execute the Notebook
```bash
cd scripts
jupyter notebook random_forest_experiments.ipynb
```

Or convert to Python script and run:
```bash
jupyter nbconvert --to script random_forest_experiments.ipynb
python random_forest_experiments.py
```

## Key Implementation Features

1. **Data Preprocessing**
   - Missing value imputation using median strategy (1 missing age value)
   - Standardization of all features before modeling
   - Train-test split: First 127 samples for training, last 44 for testing

2. **Model Configuration**
   - Random Forest with 365 trees
   - Max features per split: 1 (sqrt-based feature sampling)
   - 10-run cross-validation for robust standard error estimation
   - OOB scoring enabled for internal validation

3. **Evaluation Metrics**
   - Accuracy: Overall correct predictions
   - Sensitivity (Recall): True Positive Rate
   - Specificity: True Negative Rate
   - AUC-ROC: Area Under the Receiver Operating Characteristic curve

## Notes on Results

### Class Imbalance Challenge
The test set has **severe class imbalance** (5 fallers vs 39 non-fallers, ~11% positive class), which explains:
- Very low sensitivity scores (model rarely predicts "Faller")
- Very high specificity scores (model frequently predicts "Non-Faller")
- High overall accuracy (driven by correct non-faller predictions)

This is a common challenge in fall prediction research where fall events are rare.

### Comparison to Paper Results
The paper reported:
- Experiment I Linear: 71.8 ± 7.0% accuracy, 53.3 ± 11.5% sensitivity
- Experiment I Nonlinear: 61.4 ± 3.2% accuracy, 86.7 ± 4.7% sensitivity
- Experiment III Best: 81.6 ± 0.7% accuracy, 86.7 ± 0.5% sensitivity

Our lower sensitivity may be due to:
1. Different train-test split strategy
2. Different data preprocessing approaches
3. More severe class imbalance in our test split
4. Different random seeds or model initialization

### Recommendations for Improvement
To achieve better sensitivity-specificity balance:
1. **Use class weights** in RandomForestClassifier (e.g., `class_weight='balanced'`)
2. **Apply SMOTE** or other oversampling techniques to training data
3. **Adjust decision threshold** from default 0.5 to favor recall
4. **Use stratified sampling** to ensure balanced class distribution in train-test split
5. **Employ ensemble methods** that combine multiple threshold-adjusted models

## Variable Definitions

### Linear Variables (20)
- **Timing means**: GCTime_mean, RSST_mean, LSST_mean, RSwT_mean, LSwT_mean, DST_mean, StepTime_mean
- **RMS accelerations**: RMS_AP, RMS_V, RMS_ML, RMSR_AP, RMSR_ML, RMSR_V
- **Velocity measures**: Velocity, Time2FirstQuartile_Velocity, Time2Median_Velocity, Time2ThirdQuartile_Velocity
- **Anthropometry**: Age, Height, Weight

### Nonlinear Variables (41)
- **Variability**: sdTotal and CV for timing parameters (14 variables)
- **Harmony/Regularity**: HR_AP, HR_ML, HR_V (3 variables)
- **Multiscale Entropy**: MSE area and slope for AP, ML, V, Residual (8 variables)
- **Recurrence Quantification**: RQA metrics (Rec, Det, Ent, MaxLine) for AP, ML, V, Residual (16 variables)

## Directory Structure
```
fallprediction/
├── data/
│   └── combined_output.csv          # Input dataset
├── scripts/
│   ├── random_forest_experiments.ipynb   # Main notebook
│   ├── experiments_summary.csv           # Results summary
│   ├── experiment_III_elbow_plot.png     # Elbow analysis
│   ├── experiments_comparison.png        # Performance comparison
│   └── roc_curve_best_model.png          # ROC curve
└── EXPERIMENTS_README.md            # This file
```

## Future Work

1. Implement class-balanced modeling approaches
2. Explore hyperparameter tuning (n_estimators, max_depth, min_samples_split)
3. Compare with other algorithms (XGBoost, SVM, Neural Networks)
4. Feature importance analysis to identify most predictive gait parameters
5. Cross-validation with stratified k-fold for more robust evaluation
6. Temporal validation if longitudinal data becomes available
7. External validation on independent cohorts

## Citation

If you use this code or methodology, please cite the original research paper that inspired these experiments.

## License

This implementation is provided for research and educational purposes.
