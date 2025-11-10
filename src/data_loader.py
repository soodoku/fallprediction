"""
Data loading and preprocessing module for fall prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FallDataLoader:
    """Handles loading and preprocessing of fall prediction data."""

    def __init__(self, data_path: str = 'data/combined_output.csv'):
        """
        Initialize the data loader.

        Parameters:
        -----------
        data_path : str
            Path to the combined dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.feature_names = None
        self.scaler = None

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV."""
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded data: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
        return self.data

    def get_class_distribution(self) -> dict:
        """Get the distribution of Faller/Non-Faller classes."""
        if self.data is None:
            self.load_data()

        counts = self.data['Faller'].value_counts()
        total = len(self.data)

        return {
            'Faller': counts.get('F', 0),
            'Non-Faller': counts.get('NF', 0),
            'Faller_pct': counts.get('F', 0) / total * 100,
            'Non-Faller_pct': counts.get('NF', 0) / total * 100,
            'Total': total
        }

    def prepare_features_target(
        self,
        exclude_cols: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and target (y) from the dataset.

        Parameters:
        -----------
        exclude_cols : list, optional
            Additional columns to exclude from features

        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector (binary: 1=Faller, 0=Non-Faller)
        """
        if self.data is None:
            self.load_data()

        # Default columns to exclude
        default_exclude = ['ID', 'Faller']
        if exclude_cols:
            default_exclude.extend(exclude_cols)

        # Prepare features
        X = self.data.drop(columns=default_exclude)
        self.feature_names = X.columns.tolist()

        # Check for NaN values
        nan_counts = X.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\nWarning: Found {nan_counts.sum()} NaN values across {(nan_counts > 0).sum()} columns")
            # Fill NaN with median (robust to outliers)
            X = X.fillna(X.median())
            print("NaN values filled with column medians")

        # Prepare target (convert F=1, NF=0)
        y = (self.data['Faller'] == 'F').astype(int)

        print(f"Features: {X.shape[1]} columns")
        print(f"Target distribution: Fallers={y.sum()}, Non-Fallers={len(y)-y.sum()}")

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        test_size : float
            Proportion of data for testing (default: 0.25)
        random_state : int
            Random seed for reproducibility
        stratify : bool
            Whether to use stratified splitting (maintains class proportions)

        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        print(f"\nTrain set: {X_train.shape[0]} samples (Fallers={y_train.sum()})")
        print(f"Test set: {X_test.shape[0]} samples (Fallers={y_test.sum()})")

        return X_train, X_test, y_train, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize features using StandardScaler.

        Fits scaler on training data and transforms both train and test.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features

        Returns:
        --------
        X_train_scaled, X_test_scaled : np.ndarray
            Scaled feature arrays
        """
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Features scaled using StandardScaler")

        return X_train_scaled, X_test_scaled

    def get_full_pipeline(
        self,
        test_size: float = 0.25,
        random_state: int = 42,
        scale: bool = True,
        stratify: bool = True
    ) -> dict:
        """
        Execute the full data preparation pipeline.

        Returns a dictionary with all prepared data components.

        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed
        scale : bool
            Whether to scale features
        stratify : bool
            Whether to use stratified split

        Returns:
        --------
        dict with keys:
            X_train, X_test, y_train, y_test (scaled if scale=True)
            X_train_raw, X_test_raw (unscaled)
            feature_names, class_distribution
        """
        # Load and prepare data
        X, y = self.prepare_features_target()

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Get class distribution
        class_dist = self.get_class_distribution()

        result = {
            'X_train_raw': X_train,
            'X_test_raw': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'class_distribution': class_dist
        }

        # Scale if requested
        if scale:
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            result['X_train'] = X_train_scaled
            result['X_test'] = X_test_scaled
        else:
            result['X_train'] = X_train.values
            result['X_test'] = X_test.values

        return result


def load_and_prepare_data(
    data_path: str = 'data/combined_output.csv',
    test_size: float = 0.25,
    random_state: int = 42,
    scale: bool = True,
    stratify: bool = True
) -> dict:
    """
    Convenience function to load and prepare data in one call.

    Parameters:
    -----------
    data_path : str
        Path to data file
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    scale : bool
        Whether to scale features
    stratify : bool
        Whether to use stratified split

    Returns:
    --------
    dict with prepared data components
    """
    loader = FallDataLoader(data_path)
    return loader.get_full_pipeline(
        test_size=test_size,
        random_state=random_state,
        scale=scale,
        stratify=stratify
    )
