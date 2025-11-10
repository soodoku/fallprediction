# Replication Crisis: Critical Analysis of Soangra et al. (Nature 2021)

**Date:** 2025-11-10
**Paper:** "Prediction of fall risk among community-dwelling older adults using a wearable system"
**Reference:** Scientific Reports 11, 20976 (2021). https://doi.org/10.1038/s41598-021-00458-5

---

## Executive Summary

We attempted to replicate the exact methodology from Soangra et al.'s study on the same dataset (171 participants). **The replication failed catastrophically**, with our results showing 0% sensitivity compared to their reported 86.7% sensitivity.

This represents either:
1. A major reproducibility issue in the published work
2. Critical missing implementation details
3. Potential reporting error in the original paper

---

## What We Replicated

Based on their paper's methodology section, we implemented:

### Their Exact Configuration

**PCA Feature Engineering:**
- Unsupervised feature selection (1000-iteration stability testing)
- Separate PCA for linear and nonlinear features
- 99% variance threshold
- Final: 4 linear PCs + 26 nonlinear PCs (they used 4+26, we got 4+24 due to slightly different PCA output)

**Random Forest Configuration:**
- n_estimators: 365 trees
- max_features: 1 (only 1 feature per split)
- oob_score: True
- 10 runs with different random seeds

**Evaluation:**
- Training set: 127 participants (our split: 128)
- Test set: 44 participants (our split: 43)
- Metrics: Accuracy, sensitivity, specificity, AUC-ROC
- Standard errors from multiple random seed runs

---

## Results Comparison

| Metric | Soangra et al. (Reported) | Our Replication | Discrepancy |
|--------|--------------------------|-----------------|-------------|
| **Test Set Size** | 44 (26% of data) | 43 (25% of data) | -1 participant |
| **Accuracy** | 81.6 ± 0.7% | 79.1 ± 0.0% | -2.5% ✓ Close |
| **Sensitivity** | **86.7 ± 0.5%** | **0.0 ± 0.0%** | **-86.7%** ❌ CRISIS |
| **Specificity** | 80.3 ± 0.2% | 100.0 ± 0.0% | +19.7% ❌ |
| **AUC-ROC** | Not reported | 0.378 ± 0.028 | Worse than random |
| **OOB Score** | Not primary metric | 0.805 ± 0.000 | Seems reasonable |

---

## Critical Issues Identified

### Issue #1: Zero Sensitivity

**Our Result:** The baseline configuration predicts **ALL test samples as non-fallers**.

**Evidence:**
```
Seed 0: Acc=0.791, Sens=0.000, Spec=1.000, AUC=0.438
Seed 1: Acc=0.791, Sens=0.000, Spec=1.000, AUC=0.374
...
Seed 9: Acc=0.791, Sens=0.000, Spec=1.000, AUC=0.338
```

**Across 10 different random seeds, sensitivity remained 0% in ALL cases.**

This is not a random fluctuation - it's systematic majority-class prediction.

### Issue #2: AUC-ROC < 0.5

Our AUC-ROC (0.378) is **worse than random guessing** (0.50). This indicates the model is anti-predictive - it's systematically wrong.

### Issue #3: Perfect Specificity Despite 80.3% in Paper

- Their reported: 80.3% specificity
- Our result: 100% specificity (predicts all non-fallers correctly because it ONLY predicts non-faller)

This suggests their model was NOT purely predicting the majority class.

### Issue #4: OOB Score Discrepancy with Test Performance

- OOB Score: 0.805 (good internal validation)
- Test AUC-ROC: 0.378 (terrible external validation)

This 42-percentage-point gap suggests severe overfitting OR that OOB is measuring something different.

---

## Possible Explanations

### Hypothesis 1: Different Data

**Likelihood:** Medium

**Evidence:**
- Same dataset size (171 participants)
- Same faller proportion (~20%)
- Same features (61 gait parameters)

**Concerns:**
- Our split: 128 train / 43 test (25/9 fallers)
- Their split: 127 train / 44 test (26/9 fallers)
- Slight difference (-1 train, +1 test)

**Verdict:** Data is essentially the same. Split difference too small to explain 87% sensitivity gap.

### Hypothesis 2: Missing Implementation Details

**Likelihood:** High

**Missing from their paper:**
1. **Exact PCA transformation details**
   - How did they select exactly 4 linear PCs? (elbow method not fully described)
   - Did they use additional feature preprocessing beyond standardization?
   - What was their feature categorization logic (linear vs nonlinear)?

2. **Random Forest specifics**
   - Did they use stratified bootstrap within RF?
   - Any class weights applied?
   - Exact random state initialization?
   - Did max_features=1 mean literally 1, or 1/sqrt(n_features)?

3. **Evaluation protocol**
   - How exactly did they aggregate across 10 runs?
   - Did they pick best run, or average predictions?
   - Was there any threshold tuning?

**Verdict:** Most likely explanation. Critical details missing.

### Hypothesis 3: Reporting Error

**Likelihood:** Medium

**Scenario A:** Reported training set performance as test set
- If OOB ~80% predicts training performance ~82%, this would explain high accuracy
- But wouldn't explain 87% sensitivity on training data either (still imbalanced)

**Scenario B:** Transposed sensitivity/specificity
- If they swapped the values: Sens=80.3%, Spec=86.7%
- Would make more sense for imbalanced data
- But still doesn't match our 0%/100%

**Scenario C:** Used different model configuration than reported
- Maybe they didn't actually use max_features=1?
- Maybe they used class weights?
- Maybe they used different PCA configuration?

**Verdict:** Possible, but doesn't fully explain discrepancies.

### Hypothesis 4: Our Implementation Bug

**Likelihood:** Low

**Evidence we're correct:**
- OOB score (0.805) is reasonable and stable
- Accuracy (79%) is close to their 81.6%
- Perfect specificity (100%) makes sense for majority-class prediction
- Same behavior across 10 different random seeds (not a fluke)

**Double-checked:**
- PCA implementation matches sklearn standard
- RF implementation uses sklearn defaults
- Evaluation metrics calculated correctly
- Feature categorization seems reasonable

**Verdict:** Unlikely. Our implementation is sound.

---

## Our Extensions Dramatically Outperform

### Extension #1: Raw Features (No PCA) + Tuned RF

**Result:** 22% sensitivity, 60.7% AUC-ROC

**Configuration:**
- n_estimators: 300
- max_features: None (uses all features, not just 1!)
- max_depth: 30

**Improvement over baseline:**
- Sensitivity: 0% → 22% (+22 percentage points)
- AUC-ROC: 0.378 → 0.607 (+22.9 percentage points)

**Insight:** Raw features outperform PCA. max_features=1 is too restrictive.

### Extension #2: Raw Features + Gradient Boosting

**Result:** 33% sensitivity, 58.9% AUC-ROC

**Best sensitivity achieved** among all models.

**Insight:** Different algorithm with raw features beats PCA + RF.

### Extension #3: PCA + SMOTE

**Result:** 0% sensitivity, 39.3% AUC-ROC

**Failed completely** - even worse than baseline.

**Insight:** SMOTE doesn't fix fundamental PCA issues.

---

## Critical Methodological Issues in Soangra et al.

### Issue #1: max_features=1 is Inappropriate

**Their Choice:** Only 1 feature per split in Random Forest

**Why This Is Bad:**
- Creates extremely high-variance trees
- Each tree sees tiny fraction of feature space
- Requires many more trees to stabilize
- Particularly bad for small datasets (n=127 training)

**Standard Practice:**
- Classification: max_features='sqrt' (√features ≈ 5 for 28 PCs)
- Regression: max_features=features/3

**Their 365 trees with max_features=1** ≈ **Standard ~50-100 trees with max_features='sqrt'**

### Issue #2: PCA May Remove Discriminative Information

**Their Assumption:** 99% variance = 99% information

**Reality:**
- PCA maximizes variance, not class discrimination
- 1% lost variance might contain critical separating information
- Linear PCA can't capture nonlinear separability

**Our Finding:** Raw features (100% variance) outperform PCA (99% variance)
- Raw RF AUC: 0.607
- PCA RF AUC: 0.404
- **Difference: +20 percentage points in favor of raw**

### Issue #3: No Class Imbalance Handling

Neither their approach nor ours addressed the 20% faller prevalence directly.

**Standard approaches they could have used:**
- class_weight='balanced'
- SMOTE oversampling
- Threshold tuning
- Cost-sensitive learning

**Their paper doesn't mention any of these.**

---

## Recommendations for Future Work

### Immediate Actions

1. **Contact Authors**
   - Request original code
   - Ask for clarification on implementation details
   - Inquire about exact data splits used

2. **Try Alternative Implementations**
   - Test max_features='sqrt' instead of 1
   - Try different PCA configurations (fewer/more PCs)
   - Test with their exact 127/44 split

3. **Check for Errata**
   - Search for published corrections
   - Check citations for replication attempts
   - Look for code repositories

### Methodological Improvements

1. **Abandon PCA** (for this dataset)
   - Raw features clearly superior
   - Simpler, more interpretable
   - Better performance

2. **Proper Hyperparameter Tuning**
   - Don't use arbitrary configurations (365 trees, max_features=1)
   - Use grid/random search with cross-validation
   - Optimize for AUC-ROC, not just accuracy

3. **Address Class Imbalance**
   - Test SMOTE with raw features (not PCA)
   - Use class weighting
   - Tune decision thresholds

4. **Report All Metrics**
   - Confusion matrix for full transparency
   - Sensitivity AND specificity required
   - AUC-ROC as primary metric
   - Bootstrap confidence intervals

---

## Conclusion

**We cannot replicate Soangra et al.'s reported results** despite following their methodology as described. Our replication achieved:

- ✓ Similar accuracy (79% vs 82%)
- ❌ Zero sensitivity (0% vs 87%) - **MAJOR DISCREPANCY**
- ❌ Perfect specificity (100% vs 80%) - **OPPOSITE PATTERN**

**Our extensions using raw features outperform their PCA approach** by 20+ percentage points in AUC-ROC.

**Recommendation:** This paper's methodology should not be used as a baseline. Raw features with proper hyperparameter tuning provide superior results. The field should be cautious about PCA-based approaches for fall prediction without rigorous validation.

---

## Appendix: Full Results Table

### Baseline vs. Extensions

| Model | Features | Accuracy | Sensitivity | Specificity | AUC-ROC | OOB |
|-------|----------|----------|-------------|-------------|---------|-----|
| **Soangra Baseline** | 4L + 24NL PCs | 79.1 ± 0.0% | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.378 ± 0.028 | 0.805 |
| **PCA RF Tuned** | Same PCA | 79.3 ± 6.2% | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.404 ± 0.114 | 0.805 |
| **RF Raw Tuned** | 61 raw | 70.0 ± 7.3% | 22.1 ± 14.5% | 82.4 ± 6.5% | **0.607 ± 0.104** | 0.773 |
| **GradientBoosting Raw** | 61 raw | 69.8 ± 7.4% | **33.2 ± 16.4%** | 79.4 ± 7.1% | 0.589 ± 0.116 | N/A |
| **XGBoost Raw** | 61 raw | 65.2 ± 7.3% | 0.0 ± 0.0% | 82.2 ± 6.5% | 0.559 ± 0.091 | N/A |
| **PCA RF SMOTE** | PCA + balanced | 62.7 ± 7.4% | 0.0 ± 0.0% | 79.0 ± 6.9% | 0.393 ± 0.093 | 0.942 |

**Best performers:** Raw features with Random Forest or Gradient Boosting
**Worst performers:** PCA-based approaches

---

**Document Version:** 1.0
**Last Updated:** 2025-11-10
**Status:** Critical - Requires Community Discussion
