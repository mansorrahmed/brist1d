import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from pykalman import KalmanFilter  # For Kalman Filter imputation
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
# from keras.models import Sequential
# from layers import SimpleRNN, LSTM, Dense
# from tensorflow.keras.optimizers import Adam



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
        """
        feature_groups = {
            'numerical': [col for col in X.columns if col.startswith(('bg-', 'insulin-', 'hr-', 'cals-', 'steps-'))]
            # 'categorical': [col for col in X.columns if col.startswith(( 'activity-'))]
        }
        return feature_groups

    def mean_imp(self, X_train, X_test):
        """
            - 'mean' imputation for numerical features
            - Scale all features
        """
        print("Applying Preprocessing Strategy 1: Mean & Most Frequent Imputation")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('num_imputer', SimpleImputer(strategy='mean'), feature_groups['numerical'])
        ]

        self.pipeline = Pipeline(steps=[
            ('impute', ColumnTransformer(transformers=transformers)),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print("Strategy 1 applied: Mean imputation for numerical features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed

    def median_imp(self, X_train, X_test):
        """
            - 'median' imputation for numerical features
            - Scale all features
        """
        print("Applying Preprocessing Strategy 2: Median & Constant Imputation")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('num_imputer', SimpleImputer(strategy='median'), feature_groups['numerical'])
        ]

        self.pipeline = Pipeline(steps=[
            ('impute', ColumnTransformer(transformers=transformers)),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print("Strategy 2 applied: Median imputation for numerical features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed

    def ffill_bfill_imp(self, X_train, X_test):
        """
        Forward/Backward Fill Imputation
        """
        X_train_filled = X_train.ffill().bfill()
        X_test_filled = X_test.ffill().bfill()
        return X_train_filled, X_test_filled

    def linear_interp_imp(self, X_train, X_test):
        """
        Linear Interpolation Imputation with additional steps to handle edge cases.
        """
        # Fill NaN values at the start or end with forward fill or backward fill
        X_train_filled = X_train.fillna(method='ffill').fillna(method='bfill')
        X_test_filled = X_test.fillna(method='ffill').fillna(method='bfill')

        # Apply linear interpolation
        X_train_filled = X_train_filled.interpolate(method='linear', axis=0)
        X_test_filled = X_test_filled.interpolate(method='linear', axis=0)

        # Check if there are any NaNs left after interpolation
        if X_train_filled.isnull().values.any():
            print("Warning: NaN values found in X_train after interpolation.")
        if X_test_filled.isnull().values.any():
            print("Warning: NaN values found in X_test after interpolation.")

        return X_train_filled, X_test_filled
    
    def kalman_imp(self, X_train, X_test):
        """
        Kalman Filter Imputation for Time-Series Data
        """
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=X_train.shape[1])
        
        # Kalman filter handles time-series noise and missing data based on the system dynamics.
        X_train_filled = kf.em(X_train).smooth(X_train)[0]
        X_test_filled = kf.em(X_test).smooth(X_test)[0]
        
        return pd.DataFrame(X_train_filled, columns=X_train.columns), pd.DataFrame(X_test_filled, columns=X_test.columns)

    def knn_imp(self, X_train, X_test, n_neighbors=5):
        """
            - K-Nearest Neighbors (KNN) imputation for numerical features
            - Scale all features
        """
        print(f"Applying Preprocessing Strategy 4: KNN Imputation with {n_neighbors} Neighbors")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('knn_imputer', KNNImputer(n_neighbors=n_neighbors), feature_groups['numerical'])
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

    def mice_imp(self, X_train, X_test, max_iter=10, random_state=42):
        """
        - Multiple Imputation by Chained Equations Iterative Imputer (MICE) for numerical features
        """
        print(f"Applying Preprocessing Strategy 5: Iterative Imputer (MICE) with max_iter={max_iter}")
        feature_groups = self._get_feature_groups(X_train)

        transformers = [
            ('mice_imputer', IterativeImputer(max_iter=max_iter, random_state=random_state), feature_groups['numerical'])
            # ('cat_imputer', SimpleImputer(strategy='most_frequent'), feature_groups['categorical'])
        ]

        self.pipeline = Pipeline(steps=[
            ('impute', ColumnTransformer(transformers=transformers)),
            ('scale', self.scaler)
        ])

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        print(f"Strategy 5 applied: Iterative Imputer for numerical features.\n")
        self.fitted = True

        return X_train_processed, X_test_processed
    
    def normalize_data(self, X_train, X_test, mask_value):
        scaler = MinMaxScaler()
        # Fit scaler on non-masked data
        X_train_no_mask = X_train.replace(mask_value, np.nan)
        X_test_no_mask = X_test.replace(mask_value, np.nan)
        scaler.fit(pd.concat([X_train_no_mask, X_test_no_mask], axis=0))
        
        # Transform data while preserving the mask value
        X_train_scaled = X_train_no_mask.copy()
        X_test_scaled = X_test_no_mask.copy()
        X_train_scaled = pd.DataFrame(scaler.transform(X_train_no_mask), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_no_mask), columns=X_test.columns)
        
        # Restore the mask value
        X_train_scaled = X_train_scaled.fillna(mask_value)
        X_test_scaled = X_test_scaled.fillna(mask_value)
        return X_train_scaled, X_test_scaled


    def build_rnn(self, input_shape):
        """
        Build a simple RNN model for time-series imputation.
        The model is designed to reconstruct the input sequences.
        """
        model = models.Sequential()
        model.add(layers.SimpleRNN(64, activation='relu', return_sequences=True, kernel_initializer='he_normal'))
        model.add(layers.SimpleRNN(64, activation='relu', return_sequences=True, kernel_initializer='he_normal'))
        model.add(layers.TimeDistributed(layers.Dense(input_shape[1])))  # Output shape: [samples, seq_length, features_per_step]
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001, clipvalue=1.0), loss='mean_squared_error')
        return model

    def build_lstm(self, input_shape, mask_value):
        """
        Build a simple LSTM model for time-series imputation.
        The model is designed to reconstruct the input sequences.
        """
        model = models.Sequential()
        model.add(layers.Masking(mask_value=mask_value, input_shape=input_shape))
        model.add(layers.LSTM(64, activation='tanh', return_sequences=True, kernel_initializer='he_normal'))
        model.add(layers.LSTM(64, activation='tanh', return_sequences=True, kernel_initializer='he_normal'))
        model.add(layers.TimeDistributed(layers.Dense(input_shape[1])))  # Output shape: [samples, seq_length, features_per_step]
        model.compile(optimizer=optimizers.Adam(learning_rate=0.005, clipvalue=1.0), loss='mean_squared_error')
        return model

    def lstm_imp(self, X_train, X_test, epochs=2, seq_length=72, batch_size=64):
        """
        Impute missing values using LSTM-based model with masking.
        """
        print("Starting LSTM-based Imputation with Masking...")
        
        # Define a mask value that doesn't exist in the data
        mask_value = -999.0

        # Replace missing values with the mask value
        X_train = X_train.fillna(mask_value)
        X_test = X_test.fillna(mask_value)
        
        # Normalize data without affecting the mask value
        X_train_scaled, X_test_scaled = self.normalize_data(X_train, X_test, mask_value)
        
        # Ensure total_features is divisible by seq_length
        total_features = X_train_scaled.shape[1]
        if total_features % seq_length != 0:
            raise ValueError(f"Total features ({total_features}) not divisible by seq_length ({seq_length}). Adjust seq_length or feature engineering.")
        
        features_per_step = total_features // seq_length
        
        # Reshape data to [samples, seq_length, features_per_step]
        X_train_array = X_train_scaled.values.reshape((X_train_scaled.shape[0], seq_length, features_per_step))
        X_test_array = X_test_scaled.values.reshape((X_test_scaled.shape[0], seq_length, features_per_step))
        print((seq_length, features_per_step))
        # Build and train the LSTM model with masking
        model = self.build_lstm(input_shape=(seq_length, features_per_step), mask_value=mask_value)
        
        model.fit(
            X_train_array, 
            X_train_array, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2, 
            verbose=1
        )
        
        # Use the model to predict (impute) the sequences
        X_train_imputed = model.predict(X_train_array)
        X_test_imputed = model.predict(X_test_array)
        
        # Reshape back to original feature shape
        X_train_imputed = X_train_imputed.reshape((X_train_imputed.shape[0], total_features))
        X_test_imputed = X_test_imputed.reshape((X_test_imputed.shape[0], total_features))
        
        # Inverse transform to original scale
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
        
        # Replace mask value with NaN to reflect imputed values
        X_train_imputed = X_train_imputed.replace(mask_value, np.nan)
        X_test_imputed = X_test_imputed.replace(mask_value, np.nan)
        
        print("LSTM-based Imputation with Masking completed.\n")
        return X_train_imputed, X_test_imputed

    def rnn_imp(self, X_train, X_test,  epochs=2, seq_length=12, batch_size=64):
        """
        Impute missing values using RNN-based model.
        """
        print("Starting RNN-based Imputation...")

        # Fill missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Normalize data
        X_train_scaled, X_test_scaled = self.normalize_data(X_train, X_test)

        # Ensure total_features is divisible by seq_length
        total_features = X_train_scaled.shape[1]
        if total_features % seq_length != 0:
            raise ValueError(f"Total features ({total_features}) not divisible by seq_length ({seq_length}). Adjust seq_length or feature engineering.")
        
        features_per_step = total_features // seq_length

        # Reshape data to [samples, seq_length, features_per_step]
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], seq_length, features_per_step))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], seq_length, features_per_step))

        # Check for any remaining NaNs
        if np.isnan(X_train_reshaped).any() or np.isnan(X_test_reshaped).any():
            raise ValueError("NaN values found in the data after imputation. Please check the preprocessing.")

        # Build and train the RNN model
        model = self.build_rnn(input_shape=(seq_length, features_per_step))
        model.fit(
            X_train_reshaped, 
            X_train_reshaped, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2, 
            verbose=2
        )

        # Use the model to predict (reconstruct) the sequences
        X_train_imputed = model.predict(X_train_reshaped)
        X_test_imputed = model.predict(X_test_reshaped)

        # Reshape back to original feature shape
        X_train_imputed = X_train_imputed.reshape((X_train_imputed.shape[0], seq_length * features_per_step))
        X_test_imputed = X_test_imputed.reshape((X_test_imputed.shape[0], seq_length * features_per_step))

        # Inverse transform to original scale
        scaler = MinMaxScaler()
        scaler.fit(X_train)  # Fit on original (non-scaled) X_train
        X_train_imputed = scaler.inverse_transform(X_train_imputed)
        X_test_imputed = scaler.inverse_transform(X_test_imputed)

        # Convert back to DataFrame
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

        print("RNN-based Imputation completed.\n")
        return X_train_imputed, X_test_imputed
