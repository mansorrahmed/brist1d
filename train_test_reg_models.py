#!/usr/bin/env python3

import pandas as pd
import numpy as np
import time
import warnings
import sys
import re

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')  # To suppress warnings for cleaner output

def load_data(train_path='train.csv', test_path='test.csv'):
    """
    Load training and test data from CSV files.
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print("Data loaded successfully.")
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def preprocess_data(train, test, target_column='bg+1:00'):
    """
    Preprocess the training and test data:
    - Handle missing values using imputation.
    - Scale the features.
    """
    # Separate features and target
    train = train.loc[:, ~train.columns.str.startswith('activity-')]
    test = test.loc[:, ~test.columns.str.startswith('activity-')]

    X = train.drop(columns=[target_column, 'id', 'p_num', 'time'])  # Exclude non-feature columns

    y = train[target_column]
    
    X_test = test.drop(columns=['id', 'p_num', 'time'])  # Exclude non-feature columns
    
    # Impute missing values with mean strategy
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_test_imputed = imputer.transform(X_test)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Convert back to DataFrame for easier handling
    X = pd.DataFrame(X_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("Data preprocessing completed.")
    return X, y, X_test

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Data split into training and validation sets with test size = {test_size}")
    return X_train, X_val, y_train, y_val

def train_and_evaluate_models(X_train, X_val, y_train, y_val):
    """
    Train multiple regression models and evaluate their performance.
    Returns a DataFrame with model performance metrics.
    """
    models = {
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
    
    results = []

    for name, model in models.items():
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
    
    results_df = pd.DataFrame(results)
    return results_df

def display_results(results_df):
    """
    Display the results in a well-formatted table and identify the best model.
    """
    # Sort models by RMSE
    results_df_sorted = results_df.sort_values(by='RMSE').reset_index(drop=True)
    
    print("\nModel Performance Metrics:")
    print(results_df_sorted[['Model', 'MSE', 'RMSE', 'R2 Score', 'Training Time (s)']])
    
    # Identify the best model (lowest RMSE)
    best_model = results_df_sorted.iloc[0]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"MSE: {best_model['MSE']:.4f}")
    print(f"RMSE: {best_model['RMSE']:.4f}")
    print(f"R2 Score: {best_model['R2 Score']:.4f}")
    print(f"Training Time: {best_model['Training Time (s)']} seconds")

def save_results(results_df, filename='model_performance.csv'):
    """
    Save the results to a CSV file.
    """
    results_df.to_csv(filename, index=False)
    print(f"\nModel performance results saved to {filename}")

def plot_results(results_df_sorted):
    """
    Plot the performance metrics of the models.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='RMSE', y='Model', data=results_df_sorted, palette='viridis')
    plt.title('Model Comparison based on RMSE', fontsize=16)
    plt.xlabel('RMSE', fontsize=14)
    plt.ylabel('Regression Models', fontsize=14)
    plt.tight_layout()
    plt.show()

def make_test_predictions(best_model_name, models, X_test):
    """
    Use the best model to make predictions on the test set.
    Save the predictions to a CSV file.
    """
    best_model = models[best_model_name]
    predictions = best_model.predict(X_test)
    
    # Assuming 'id' is present in the test data
    test_ids = test_data['id']
    submission = pd.DataFrame({
        'id': test_ids,
        'bg+1:00': predictions
    })
    submission.to_csv('test_predictions.csv', index=False)
    print("\nTest set predictions saved to test_predictions.csv")

if __name__ == "__main__":
    # Load data
    # proj_dir = '/Users/mansoor/Documents/GSU/Coursework/AML/Project/'
    proj_dir = "/home/mahmed76/Documents/Mansoor/Courses/AML/"

    data_dir = proj_dir+ 'data/'
    results_dir = proj_dir + "/brist1d/results/"


    train_data, test_data = load_data(data_dir+'train.csv', data_dir+'test.csv')

    # Preprocess data
    X, y, X_test = preprocess_data(train_data, test_data, target_column='bg+1:00')
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    results_df = train_and_evaluate_models(X_train, X_val, y_train, y_val)
    
    # Sort results by RMSE for better visualization
    results_df_sorted = results_df.sort_values(by='RMSE').reset_index(drop=True)
    
    # Display results
    display_results(results_df_sorted)
    
    # Save results to CSV
    save_results(results_df_sorted, filename=results_dir+'model_performance.csv')
    
    # Plot results
    plot_results(results_df_sorted)
    
    # Identify the best model name
    best_model_name = results_df_sorted.iloc[0]['Model']
    
    # Retrain all models on the entire training data if needed
    # (Optional) Here, we can retrain the best model on the entire training set before making test predictions
    print(f"\nRetraining the best model ({best_model_name}) on the entire training data for test predictions...")
    
    # Combine training and validation sets
    X_full_train = pd.concat([X_train, X_val], axis=0)
    y_full_train = pd.concat([y_train, y_val], axis=0)
    
    # Re-initialize models
    models = {
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
    
    # Train the best model
    best_model = models[best_model_name]
    best_model.fit(X_full_train, y_full_train)
    print(f"{best_model_name} retrained on the entire training data.")
    
    # Make predictions on the test set
    predictions = best_model.predict(X_test)
    
    # Prepare submission
    submission = pd.DataFrame({
        'id': test_data['id'],
        'bg+1:00': predictions
    })
    submission.to_csv(results_dir+'test_predictions.csv', index=False)
    print("Test set predictions saved to test_predictions.csv")
