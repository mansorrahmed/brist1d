import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class Preprocessor:
    def __init__(self, target_column='bg+1:00'):
        """
        Initializes the Preprocessor with the target column name.

        Parameters:
            target_column (str): The name of the target variable to predict.
        """
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.fitted = False
        self.pipeline = None

    def _get_feature_groups(self, X):
        """
        Groups features into numerical and categorical based on their prefixes.

        Parameters:
            X (pd.DataFrame): Feature DataFrame.

        Returns:
            dict: Dictionary with 'numerical' and 'categorical' keys mapping to lists of column names.
        """
        feature_groups = {
            'numerical': [col for col in X.columns if col.startswith(('bg-', 'insulin-', 'hr-', 'cals-', 'steps-'))]
            # 'categorical': [col for col in X.columns if col.startswith(( 'activity-'))]
        }
        return feature_groups

    def preprocess_strategy1(self, X_train, X_test):
        """
        Strategy 1:
            - 'mean' imputation for numerical features
            - 'most_frequent' imputation for categorical features
            - Scale all features

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.

        Returns:
            tuple: (X_train_processed, X_test_processed)
        """
        print("Applying Preprocessing Strategy 1: Mean & Most Frequent Imputation")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('num_imputer', SimpleImputer(strategy='mean'), feature_groups['numerical'])
            # ('cat_imputer', SimpleImputer(strategy='most_frequent'), feature_groups['categorical'])
        ]

        self.pipeline = Pipeline(steps=[
            ('impute', ColumnTransformer(transformers=transformers)),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print("Strategy 1 applied: Mean imputation for numerical and Most Frequent for categorical features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed

    def preprocess_strategy2(self, X_train, X_test):
        """
        Strategy 2:
            - 'median' imputation for numerical features
            - 'constant' imputation with 'Unknown' for categorical features
            - Scale all features

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.

        Returns:
            tuple: (X_train_processed, X_test_processed)
        """
        print("Applying Preprocessing Strategy 2: Median & Constant Imputation")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('num_imputer', SimpleImputer(strategy='median'), feature_groups['numerical']),
            ('cat_imputer', SimpleImputer(strategy='constant', fill_value='Unknown'), feature_groups['categorical'])
        ]

        self.pipeline = Pipeline(steps=[
            ('impute', ColumnTransformer(transformers=transformers)),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print("Strategy 2 applied: Median imputation for numerical and Constant ('Unknown') for categorical features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed

    def preprocess_strategy3(self, X_train, X_test):
        """
        Strategy 3:
            - 'mean' imputation for all features
            - Scale all features

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.

        Returns:
            tuple: (X_train_processed, X_test_processed)
        """
        print("Applying Preprocessing Strategy 3: Mean Imputation for All Features")
        imputer = SimpleImputer(strategy='mean')

        self.pipeline = Pipeline(steps=[
            ('impute', imputer),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print("Strategy 3 applied: Mean imputation for all features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed

    def preprocess_strategy4(self, X_train, X_test, n_neighbors=5):
        """
        Strategy 4:
            - K-Nearest Neighbors (KNN) imputation for numerical features
            - 'most_frequent' imputation for categorical features
            - Scale all features

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            n_neighbors (int): Number of neighbors to use for KNN imputation.

        Returns:
            tuple: (X_train_processed, X_test_processed)
        """
        print(f"Applying Preprocessing Strategy 4: KNN Imputation with {n_neighbors} Neighbors")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('knn_imputer', KNNImputer(n_neighbors=n_neighbors), feature_groups['numerical']),
            ('cat_imputer', SimpleImputer(strategy='most_frequent'), feature_groups['categorical'])
        ]

        self.pipeline = Pipeline(steps=[
            ('impute', ColumnTransformer(transformers=transformers)),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print(f"Strategy 4 applied: KNN imputation for numerical and Most Frequent for categorical features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed

    def preprocess_strategy5(self, X_train, X_test, max_iter=10, random_state=42):
        """
        Strategy 5:
            - Iterative Imputer (MICE) for numerical features
            - 'most_frequent' imputation for categorical features
            - Scale all features

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            max_iter (int): Maximum number of imputation iterations.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: (X_train_processed, X_test_processed)
        """
        print(f"Applying Preprocessing Strategy 5: Iterative Imputer (MICE) with max_iter={max_iter}")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('mice_imputer', IterativeImputer(max_iter=max_iter, random_state=random_state), feature_groups['numerical']),
            ('cat_imputer', SimpleImputer(strategy='most_frequent'), feature_groups['categorical'])
        ]

        self.pipeline = Pipeline(steps=[
            ('impute', ColumnTransformer(transformers=transformers)),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print(f"Strategy 5 applied: Iterative Imputer for numerical and Most Frequent for categorical features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed

    def get_feature_columns(self):
        """
        Get the list of feature columns after preprocessing.

        Returns:
            list: List of feature column names.
        """
        if not self.fitted:
            raise Exception("Preprocessor has not been fitted yet.")
        return self.feature_columns
