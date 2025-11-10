"""
PCA feature engineering module for fall prediction.

Implements the PCA approach from Soangra et al. (Nature Scientific Reports, 2021)
for dimensionality reduction and feature decorrelation.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class GaitFeaturePCA:
    """
    PCA feature engineering following Soangra et al.'s methodology.

    Steps:
    1. Unsupervised feature selection (remove features with high source discrepancy)
    2. Separate PCA for linear and nonlinear features
    3. Select PCs explaining 99% variance
    """

    def __init__(self, variance_threshold: float = 0.99, n_iterations: int = 1000):
        """
        Initialize PCA transformer.

        Parameters:
        -----------
        variance_threshold : float
            Cumulative variance to retain (default: 0.99 for 99%)
        n_iterations : int
            Number of iterations for feature stability testing
        """
        self.variance_threshold = variance_threshold
        self.n_iterations = n_iterations

        self.linear_features = None
        self.nonlinear_features = None
        self.selected_linear_features = None
        self.selected_nonlinear_features = None

        self.pca_linear = None
        self.pca_nonlinear = None
        self.scaler_linear = None
        self.scaler_nonlinear = None

    def identify_feature_types(self, feature_names: list) -> Tuple[list, list]:
        """
        Separate features into linear and nonlinear based on naming convention.

        Linear features: temporal parameters, RMS, velocity, anthropometry
        Nonlinear features: SD, CV, entropy, MSE, RQA, harmonic ratio
        """
        linear = []
        nonlinear = []

        # Nonlinear indicators
        nonlinear_keywords = ['_sd', '_cv', 'ApEn', 'SaEn', 'MSE', 'RQA',
                             'Harmony', 'Regularity', 'Ent']

        for feat in feature_names:
            is_nonlinear = any(keyword.lower() in feat.lower()
                             for keyword in nonlinear_keywords)
            if is_nonlinear:
                nonlinear.append(feat)
            else:
                linear.append(feat)

        print(f"\nFeature categorization:")
        print(f"  Linear features: {len(linear)}")
        print(f"  Nonlinear features: {len(nonlinear)}")

        return linear, nonlinear

    def unsupervised_feature_selection(
        self,
        X_train: pd.DataFrame,
        feature_list: list,
        alpha: float = 0.05
    ) -> list:
        """
        Remove features with high source discrepancy using repeated random splits.

        Following Soangra et al.: randomly split training data, test if features
        show significant differences between random splits (indicates instability).

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        feature_list : list
            Features to test
        alpha : float
            Significance threshold (features with p < alpha are unstable)

        Returns:
        --------
        list of stable features
        """
        n_samples = len(X_train)
        p_values = {feat: [] for feat in feature_list}

        for i in range(self.n_iterations):
            # Random split into two groups
            indices = np.random.permutation(n_samples)
            mid = n_samples // 2

            group1_idx = indices[:mid]
            group2_idx = indices[mid:]

            for feat in feature_list:
                group1_vals = X_train[feat].iloc[group1_idx]
                group2_vals = X_train[feat].iloc[group2_idx]

                # Two-sample t-test
                _, p_val = stats.ttest_ind(group1_vals, group2_vals, equal_var=False)
                p_values[feat].append(p_val)

        # Average p-values across iterations
        avg_p_values = {feat: np.mean(p_vals) for feat, p_vals in p_values.items()}

        # Select stable features (high average p-value = no systematic difference)
        stable_features = [feat for feat, p_val in avg_p_values.items()
                          if p_val >= alpha]

        removed = set(feature_list) - set(stable_features)

        print(f"\n  Feature selection results:")
        print(f"    Original features: {len(feature_list)}")
        print(f"    Stable features: {len(stable_features)}")
        print(f"    Removed (unstable): {len(removed)}")
        if removed:
            print(f"    Removed features: {sorted(removed)}")

        return stable_features

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        Fit PCA transformers on training data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series, optional
            Training labels (not used in unsupervised PCA)
        """
        # Identify feature types
        self.linear_features, self.nonlinear_features = \
            self.identify_feature_types(X_train.columns.tolist())

        # Unsupervised feature selection for each type
        print("\nLinear features - stability testing:")
        self.selected_linear_features = self.unsupervised_feature_selection(
            X_train, self.linear_features
        )

        print("\nNonlinear features - stability testing:")
        self.selected_nonlinear_features = self.unsupervised_feature_selection(
            X_train, self.nonlinear_features
        )

        # Fit PCA for linear features
        print(f"\nFitting PCA for linear features...")
        X_linear = X_train[self.selected_linear_features]

        self.scaler_linear = StandardScaler()
        X_linear_scaled = self.scaler_linear.fit_transform(X_linear)

        self.pca_linear = PCA(n_components=self.variance_threshold, svd_solver='full')
        self.pca_linear.fit(X_linear_scaled)

        n_linear_pcs = self.pca_linear.n_components_
        var_explained_linear = self.pca_linear.explained_variance_ratio_.sum()

        print(f"  Linear: {len(self.selected_linear_features)} features → "
              f"{n_linear_pcs} PCs (variance: {var_explained_linear:.3f})")

        # Fit PCA for nonlinear features
        print(f"\nFitting PCA for nonlinear features...")
        X_nonlinear = X_train[self.selected_nonlinear_features]

        self.scaler_nonlinear = StandardScaler()
        X_nonlinear_scaled = self.scaler_nonlinear.fit_transform(X_nonlinear)

        self.pca_nonlinear = PCA(n_components=self.variance_threshold, svd_solver='full')
        self.pca_nonlinear.fit(X_nonlinear_scaled)

        n_nonlinear_pcs = self.pca_nonlinear.n_components_
        var_explained_nonlinear = self.pca_nonlinear.explained_variance_ratio_.sum()

        print(f"  Nonlinear: {len(self.selected_nonlinear_features)} features → "
              f"{n_nonlinear_pcs} PCs (variance: {var_explained_nonlinear:.3f})")

        return self

    def transform(self, X: pd.DataFrame, n_linear_pcs: int = None,
                  n_nonlinear_pcs: int = None) -> pd.DataFrame:
        """
        Transform features to PCA space.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
        n_linear_pcs : int, optional
            Number of linear PCs to use (default: all fitted)
        n_nonlinear_pcs : int, optional
            Number of nonlinear PCs to use (default: all fitted)

        Returns:
        --------
        pd.DataFrame with PC features
        """
        # Transform linear features
        X_linear = X[self.selected_linear_features]
        X_linear_scaled = self.scaler_linear.transform(X_linear)
        linear_pcs = self.pca_linear.transform(X_linear_scaled)

        if n_linear_pcs is not None:
            linear_pcs = linear_pcs[:, :n_linear_pcs]

        # Transform nonlinear features
        X_nonlinear = X[self.selected_nonlinear_features]
        X_nonlinear_scaled = self.scaler_nonlinear.transform(X_nonlinear)
        nonlinear_pcs = self.pca_nonlinear.transform(X_nonlinear_scaled)

        if n_nonlinear_pcs is not None:
            nonlinear_pcs = nonlinear_pcs[:, :n_nonlinear_pcs]

        # Combine
        n_lin = linear_pcs.shape[1]
        n_nonlin = nonlinear_pcs.shape[1]

        linear_cols = [f'Linear_PC{i+1}' for i in range(n_lin)]
        nonlinear_cols = [f'Nonlinear_PC{i+1}' for i in range(n_nonlin)]

        result = pd.DataFrame(
            np.hstack([linear_pcs, nonlinear_pcs]),
            columns=linear_cols + nonlinear_cols,
            index=X.index
        )

        return result

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None,
                      n_linear_pcs: int = None, n_nonlinear_pcs: int = None) -> pd.DataFrame:
        """Fit PCA and transform training data."""
        self.fit(X_train, y_train)
        return self.transform(X_train, n_linear_pcs, n_nonlinear_pcs)

    def get_optimal_n_pcs(self) -> Dict[str, int]:
        """
        Get the number of PCs selected during fitting.

        Returns:
        --------
        dict with 'linear' and 'nonlinear' PC counts
        """
        return {
            'linear': self.pca_linear.n_components_ if self.pca_linear else None,
            'nonlinear': self.pca_nonlinear.n_components_ if self.pca_nonlinear else None
        }

    def get_explained_variance(self) -> Dict[str, np.ndarray]:
        """Get explained variance ratio for each PC."""
        return {
            'linear': self.pca_linear.explained_variance_ratio_ if self.pca_linear else None,
            'nonlinear': self.pca_nonlinear.explained_variance_ratio_ if self.pca_nonlinear else None
        }
