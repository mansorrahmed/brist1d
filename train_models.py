import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output


class ModelTrainer:
    def __init__(self, models=None, random_state=42):
        """
        Initializes the ModelTrainer with a dictionary of models.

        Parameters:
            models (dict): A dictionary of model name and instantiated model objects.
                           If None, default models are initialized.
            random_state (int): Random state for reproducibility.
        """
        if models is None:
            self.models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        # 'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        # 'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
        # 'Support Vector Regressor': SVR()
    }
    
        else:
            self.models = models
        self.results = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split the data into training and validation sets.

        Parameters:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or pd.DataFrame): Target variable.
            test_size (float): Proportion of the dataset to include in the validation split.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Data split into training and validation sets with test size = {test_size}")
        return X_train, X_val, y_train, y_val

    def train_and_evaluate(self, X_train, X_val, y_train, y_val):
        """
        Train multiple regression models and evaluate their performance.

        Parameters:
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series or pd.DataFrame): Training target.
            y_val (pd.Series or pd.DataFrame): Validation target.

        Returns:
            pd.DataFrame: DataFrame containing performance metrics for each model.
        """
        results = []

        for name, model in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Predict on validation set
            y_pred = model.predict(X_val)

            # Evaluate performance
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)

            results.append({
                'Model': name,
                'MSE': mse,
                'RMSE': rmse,
                'R2 Score': r2,
                'Training Time (s)': round(training_time, 4)
            })
            print(f"{name} trained in {round(training_time, 4)} seconds.")

        self.results = pd.DataFrame(results)
        return self.results

    def display_results(self):
        """
        Display the performance metrics in a well-formatted table and identify the best model.
        """
        # Sort models by RMSE
        results_sorted = self.results.sort_values(by='RMSE').reset_index(drop=True)

        print("\nModel Performance Metrics:")
        print(results_sorted[['Model', 'MSE', 'RMSE', 'R2 Score', 'Training Time (s)']])

        # Identify the best model (lowest RMSE)
        self.best_model = self.models[results_sorted.iloc[0]['Model']]
        self.best_model_name = results_sorted.iloc[0]['Model']
        best_model_metrics = results_sorted.iloc[0]

        print(f"\nBest Model: {self.best_model_name}")
        print(f"MSE: {best_model_metrics['MSE']:.4f}")
        print(f"RMSE: {best_model_metrics['RMSE']:.4f}")
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
        Plot the performance metrics of the models based on MSE.

        Parameters:
            filename (str): Name of the file to save the plot.
        """
        results_sorted = self.results.sort_values(by='MSE').reset_index(drop=True)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='MSE', y='Model', data=results_sorted, palette='viridis')
        plt.title('Model Comparison based on MSE', fontsize=16)
        plt.xlabel('MSE', fontsize=14)
        plt.ylabel('Regression Models', fontsize=14)
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
        start_time = time.time()
        self.best_model.fit(X_full, y_full)
        training_time = time.time() - start_time
        print(f"{self.best_model_name} retrained on the entire training data in {round(training_time, 4)} seconds.")

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
        predictions = self.best_model.predict(X_test)
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
            'bg+1:00': predictions
        })
        submission.to_csv(filename, index=False)
        print(f"Test set predictions saved to {filename}")

    def train_test_save_models(self, X_train_processed, X_test_processed, y_train, test_ids, strategy_type, results_dir):
            # Split data into training and validation sets
        X_train_split, X_val_split, y_train_split, y_val_split = self.split_data(
            X_train_processed, y_train, test_size=0.2, random_state=42
        )

        # Train and evaluate models
        self.train_and_evaluate(X_train_split, X_val_split, y_train_split, y_val_split)

        # Display results
        self.display_results()

        # Save results
        self.save_results(results_dir+f'model_performance_{strategy_type}.csv')

        # Plot results
        self.plot_results(results_dir+f'model_performance_{strategy_type}.png')

        # Retrain the best model on the entire training set
        self.retrain_best_model(X_train_processed, y_train)

        # Make predictions on the test set
        predictions = self.predict_test(X_test_processed)

        # Save predictions
        self.save_predictions(test_ids, predictions, results_dir+f'test_predictions_{strategy_type}.csv')

