import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow import keras
from keras import models, layers, callbacks, optimizers
from tcn import TCN

# from keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, TimeDistributed
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output


class DLModelTrainer:
    def __init__(self, epochs=5, models=None, sequence_length=12, batch_size=64, random_state=42):
        """
        Initializes the DeepLearningModelTrainer with a dictionary of models.

        Parameters:
            models (dict): A dictionary of model names and functions that build model objects.
                           If None, default models are initialized.
            sequence_length (int): Length of the input sequences for time series data.
            batch_size (int): Batch size for training the models.
            epochs (int): Number of epochs to train the models.
            random_state (int): Random state for reproducibility.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        if models is None:
            self.models = {
                'LSTM': self.build_lstm_model,
                'GRU': self.build_gru_model
                # 'TCN': self.build_tcn_model
            }
        else:
            self.models = models
        self.results = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.X_full = None  # To store the full training data for prediction

    def build_lstm_model(self, input_shape):
        """
        Builds and compiles an LSTM model.

        Parameters:
            input_shape (tuple): Shape of the input data (sequence_length, num_features).

        Returns:
            model: Compiled LSTM model.
        """
        model = models.Sequential()
        model.add(layers.LSTM(64, activation='tanh', input_shape=input_shape)) # input and hidden units specified
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1)) # output layer
        model.compile(optimizer=optimizers.Adam(), loss='mse')
        return model

    def build_gru_model(self, input_shape):
        """
        Builds and compiles a GRU model.

        Parameters:
            input_shape (tuple): Shape of the input data (sequence_length, num_features).

        Returns:
            model: Compiled GRU model.
        """
        model = models.Sequential()
        model.add(layers.GRU(64, activation='tanh', input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1))
        model.compile(optimizer=optimizers.Adam(), loss='mse')
        return model
    
    def build_tcn_model(self, input_shape):
        model = models.Sequential()
        model.add(TCN(64, input_shape=(3,64,64)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1))
        model.compile(optimizer=optimizers.Adam(), loss='mse')
        return model

    def create_sequences(self, X, y, sequence_length):
        """
        Creates input sequences and corresponding targets for training.

        Parameters:
            X (pd.DataFrame or np.array): Feature data.
            y (pd.Series or np.array): Target data.
            sequence_length (int): Length of the sequences to create.

        Returns:
            tuple: (X_sequences, y_sequences)
        """
        X_values = X.values
        y_values = y.values
        X_sequences = []
        y_sequences = []
        for i in range(len(X_values) - sequence_length):
            X_sequences.append(X_values[i:i + sequence_length])
            y_sequences.append(y_values[i + sequence_length])
        return np.array(X_sequences), np.array(y_sequences)

    def split_data(self, X, y, test_size=0.2):
        """
        Split the data into training and validation sets, preserving temporal order.

        Parameters:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or pd.DataFrame): Target variable.
            test_size (float): Proportion of the dataset to include in the validation split.

        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Data split into training and validation sets with test size = {test_size}")
        return X_train, X_val, y_train, y_val

    def train_and_evaluate(self, X_train, X_val, y_train, y_val):
        """
            
        Train multiple deep learning models and evaluate their performance.

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series or pd.DataFrame): Training target.
            y_val (pd.Series or pd.DataFrame): Validation target.

        Returns:
            pd.DataFrame: DataFrame containing performance metrics for each model.
        """
        results = []

        # Reshape the data for time-series models (e.g., LSTM, TCN)
        X_train_reshaped, X_val_reshaped = self.reshape_for_timeseries(X_train, X_val)

        for name, build_model_fn in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()

            model = build_model_fn(input_shape=X_train_reshaped.shape[1:])
            
            # Train the model
            model.fit(X_train_reshaped, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val_reshaped, y_val), verbose=2)
            training_time = time.time() - start_time

            # Predict on validation set
            y_pred = model.predict(X_val_reshaped)

            # Evaluate performance
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2,
            'Training Time (s)': round(training_time, 4),
            "Strategy type": self.strategy_type,
            "Preprocessing time": self.preprocessing_time
        })
        print(f"{name} trained in {round(training_time, 4)} seconds.")


        self.results = pd.DataFrame(results)
        return self.results

    def reshape_for_timeseries(self, X_train, X_val):
        """
        Reshapes the data for time-series models, ensuring 3D format (samples, timesteps, features).
        """
        X_train_reshaped = X_train.reshape((X_train.shape[0], self.sequence_length, X_train.shape[1] // self.sequence_length))
        X_val_reshaped = X_val.reshape((X_val.shape[0], self.sequence_length, X_val.shape[1] // self.sequence_length))
        return X_train_reshaped, X_val_reshaped
    

    def display_results(self):
        """
        Display the performance metrics in a well-formatted table and identify the best model.
        """
        # Sort models by RMSE
        results_sorted = self.results.sort_values(by='MSE').reset_index(drop=True)

        print("\nModel Performance Metrics:")
        print(results_sorted[['Model', 'MSE', 'RMSE', 'MAE', 'R2 Score', 'Training Time (s)']])

        # Identify the best model (lowest RMSE)
        self.best_model_name = results_sorted.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        best_model_metrics = results_sorted.iloc[0]

        print(f"\nBest Model: {self.best_model_name}")
        print(f"MSE: {best_model_metrics['MSE']:.4f}")
        print(f"RMSE: {best_model_metrics['RMSE']:.4f}")
        print(f"MAE: {best_model_metrics['MAE']:.4f}")
        print(f"R2 Score: {best_model_metrics['R2 Score']:.4f}")
        print(f"Training Time: {best_model_metrics['Training Time (s)']} seconds")

    def save_results(self, filename='model_performance.csv'):
        """
        Save the performance metrics to a CSV file.

        Parameters:
            filename (str): Name of the CSV file to save the results.
        """
        self.results.to_csv(filename, index=False)
        print(f"\nModel performance results saved to {filename}")

    def plot_results(self, filename='model_performance.png'):
        """
        Plot the performance metrics of the models based on RMSE.

        Parameters:
            filename (str): Name of the image file to save the plot.
        """
        results_sorted = self.results.sort_values(by='RMSE').reset_index(drop=True)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='MSE', y='Model', data=results_sorted, palette='viridis')
        plt.title(f'Model Comparison: {self.strategy_name}', fontsize=16)
        plt.xlabel('MSE', fontsize=14)
        plt.ylabel('Deep Learning Models', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Model performance plot saved to {filename}")

    def retrain_best_model(self, X_full, y_full):
        """
        Retrain the best model on the entire training dataset.

        Parameters:
            X_full (pd.DataFrame): Entire training features.
            y_full (pd.Series or pd.DataFrame): Entire training target.
        """
        if self.best_model is None:
            raise Exception("No best model identified. Please train and evaluate models first.")

        print(f"\nRetraining the best model ({self.best_model_name}) on the entire training data...")

        # Scale data
        scaler = MinMaxScaler()
        X_full_scaled = scaler.fit_transform(X_full)

        # Create sequences
        X_full_sequences, y_full_sequences = self.create_sequences(
            pd.DataFrame(X_full_scaled), y_full.reset_index(drop=True), self.sequence_length)

        input_shape = (X_full_sequences.shape[1], X_full_sequences.shape[2])

        # Build the model again
        build_model_fn = getattr(self, f'build_{self.best_model_name.lower()}_model')
        model = build_model_fn(input_shape)

        # Train the model
        start_time = time.time()
        model.fit(
            X_full_sequences, y_full_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )
        training_time = time.time() - start_time
        print(f"{self.best_model_name} retrained on the entire training data in {round(training_time, 4)} seconds.")

        # Save the trained model and scaler
        self.best_model = model
        self.scaler = scaler
        self.X_full = X_full  # Save for test data preparation

    def predict_test(self, X_test):
        """
        Make predictions on the test set using the best model.

        Parameters:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predicted values for the test set.
        """
        if self.best_model is None:
            raise Exception("No best model identified. Please train and evaluate models first.")

        print(f"Making predictions on the test set using the best model ({self.best_model_name})...")

        # Scale test data using the same scaler
        X_test_scaled = self.scaler.transform(X_test)

        # Combine last sequence_length data points from training data with test data
        X_full_scaled = self.scaler.transform(self.X_full)

        X_combined = np.concatenate((X_full_scaled[-self.sequence_length:], X_test_scaled), axis=0)

        # Create sequences from combined data
        X_test_sequences = []
        for i in range(len(X_test_scaled)):
            X_seq = X_combined[i:i + self.sequence_length]
            if len(X_seq) == self.sequence_length:
                X_test_sequences.append(X_seq)
        X_test_sequences = np.array(X_test_sequences)

        # Predict
        predictions = self.best_model.predict(X_test_sequences)

        return predictions

    def save_predictions(self, test_ids, predictions, filename='test_predictions.csv'):
        """
        Save the test set predictions to a CSV file.

        Parameters:
            test_ids (pd.Series or pd.DataFrame): IDs corresponding to the test set.
            predictions (np.ndarray): Predicted target values.
            filename (str): Name of the CSV file to save the predictions.
        """
        submission = pd.DataFrame({
            'id': test_ids,
            'bg+1:00': predictions.flatten()
        })
        submission.to_csv(filename, index=False)
        print(f"Test set predictions saved to {filename}")

    def train_test_save_models(self, X_train_processed, X_test_processed, y_train, test_ids, strategy_type, strategy_name, preprocessing_time, 
                               results_dir, models_category):
        """
        Orchestrates the entire training, evaluation, and prediction process.

        Parameters:
            X_train_processed (pd.DataFrame): Processed training features.
            X_test_processed (pd.DataFrame): Processed test features.
            y_train (pd.Series): Training target variable.
            test_ids (pd.Series): IDs for the test set predictions.
            strategy_type (str): Description of the preprocessing strategy.
            strategy_name (str): Name of the preprocessing strategy.
            preprocessing_time (float): Time taken for preprocessing.
            results_dir (str): Directory to save results and outputs.
        """
        # Set attributes
        self.preprocessing_time = preprocessing_time
        self.strategy_type = strategy_type
        self.strategy_name = strategy_name
        self.models_category = models_category

        # Split data into training and validation sets
        X_train_split, X_val_split, y_train_split, y_val_split = self.split_data(
            X_train_processed, y_train, test_size=0.2
        )

        # Train and evaluate models
        self.train_and_evaluate(X_train_split, X_val_split, y_train_split, y_val_split)

        # Display results
        self.display_results()

        # Save results
        self.save_results(results_dir + f'performance/model_performance_{strategy_type}_{models_category}.csv')

        # Plot results
        self.plot_results(results_dir + f'performance/model_performance_{strategy_type}_{models_category}.png')

        # Retrain the best model on the entire training set
        self.retrain_best_model(X_train_processed, y_train)

        # print(X_test_processed.shape)
        # Make predictions on the test set
        predictions = self.predict_test(X_test_processed)
        print(predictions.shape)

        # Save predictions
        self.save_predictions(test_ids, predictions, results_dir + f'predictions/test_predictions_{strategy_type}_{models_category}.csv')
