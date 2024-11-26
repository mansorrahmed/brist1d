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
from keras import layers, models



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
        Linear Interpolation Imputation
        """
        X_train_filled = X_train.interpolate(method='linear', axis=0)
        X_test_filled = X_test.interpolate(method='linear', axis=0)
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


    def lstm_imp(self, X_train, X_test, seq_length=10, epochs=5, batch_size=64):
        """
        LSTM-based imputation for time-series data.

        Parameters:
            X_train (pd.DataFrame): Training features with missing values.
            X_test (pd.DataFrame): Test features with missing values.
            seq_length (int): Length of the input sequences.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size for training.

        Returns:
            pd.DataFrame: Imputed X_train and X_test.
        """
        print("Applying LSTM-based Imputation")

        # Combine train and test for consistent scaling and imputation
        X_combined = pd.concat([X_train, X_test], ignore_index=True)

        # Scale features between 0 and 1
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_combined)

        # Prepare data for LSTM
        X_sequences, y_sequences = self._create_sequences(X_scaled, seq_length)

        # Build LSTM model
        model = self._build_lstm_model(input_shape=(seq_length, X_train.shape[1]))

        # Train model
        model.fit(X_sequences, y_sequences, epochs=epochs, batch_size=batch_size, verbose=1)

        # Impute missing values
        X_imputed = self._impute_with_model(model, X_scaled, seq_length)

        # Rescale features back to original scale
        X_imputed_rescaled = scaler.inverse_transform(X_imputed)

        # Split imputed data back into train and test
        X_train_imputed = X_imputed_rescaled[:len(X_train)]
        X_test_imputed = X_imputed_rescaled[len(X_train):]

        # Convert back to DataFrame
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

        print("LSTM-based Imputation completed.\n")
        return X_train_imputed, X_test_imputed

    def rnn_imp(self, X_train, X_test, seq_length=10, epochs=5, batch_size=64):
        """
        RNN-based imputation for time-series data.

        Parameters:
            X_train (pd.DataFrame): Training features with missing values.
            X_test (pd.DataFrame): Test features with missing values.
            seq_length (int): Length of the input sequences.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size for training.

        Returns:
            pd.DataFrame: Imputed X_train and X_test.
        """
        print("Applying RNN-based Imputation")

        # Combine train and test for consistent scaling and imputation
        X_combined = pd.concat([X_train, X_test], ignore_index=True)

        # Scale features between 0 and 1
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_combined)

        # Prepare data for RNN
        X_sequences, y_sequences = self._create_sequences(X_scaled, seq_length)

        # Build RNN model
        model = self._build_rnn_model(input_shape=(seq_length, X_train.shape[1]))

        # Train model
        model.fit(X_sequences, y_sequences, epochs=epochs, batch_size=batch_size, verbose=1)

        # Impute missing values
        X_imputed = self._impute_with_model(model, X_scaled, seq_length)

        # Rescale features back to original scale
        X_imputed_rescaled = scaler.inverse_transform(X_imputed)

        # Split imputed data back into train and test
        X_train_imputed = X_imputed_rescaled[:len(X_train)]
        X_test_imputed = X_imputed_rescaled[len(X_train):]

        # Convert back to DataFrame
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

        print("RNN-based Imputation completed.\n")
        return X_train_imputed, X_test_imputed

    def _create_sequences(self, X, seq_length):
        """
        Helper function to create sequences for LSTM/RNN.

        Parameters:
            X (np.ndarray): Scaled feature array.
            seq_length (int): Length of the sequences.

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []

        for i in range(seq_length, len(X)):
            X_sequences.append(X[i - seq_length:i, :])
            y_sequences.append(X[i, :])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        return X_sequences, y_sequences

    def _build_lstm_model(self, input_shape):
        """
        Builds an LSTM model for imputation.

        Parameters:
            input_shape (tuple): Shape of the input data.

        Returns:
            Compiled LSTM model.
        """
        model = models.Sequential()
        model.add(layers.Masking(mask_value=0., input_shape=input_shape))
        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.Dense(input_shape[1]))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _build_rnn_model(self, input_shape):
        """
        Builds an RNN model for imputation.

        Parameters:
            input_shape (tuple): Shape of the input data.

        Returns:
            Compiled RNN model.
        """
        model = models.Sequential()
        model.add(layers.Masking(mask_value=0., input_shape=input_shape))
        model.add(layers.SimpleRNN(64, return_sequences=False))
        model.add(layers.Dense(input_shape[1]))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _impute_with_model(self, model, X_scaled, seq_length):
        """
        Uses the trained model to impute missing values.

        Parameters:
            model: Trained LSTM/RNN model.
            X_scaled (np.ndarray): Scaled feature array with missing values.
            seq_length (int): Length of the sequences.

        Returns:
            np.ndarray: Imputed feature array.
        """
        X_imputed = X_scaled.copy()
        for i in range(seq_length, len(X_scaled)):
            if np.isnan(X_scaled[i]).any():
                X_seq = X_scaled[i - seq_length:i, :].reshape(1, seq_length, X_scaled.shape[1])
                y_pred = model.predict(X_seq)
                X_imputed[i, np.isnan(X_scaled[i])] = y_pred[0, np.isnan(X_scaled[i])]
        return X_imputed


