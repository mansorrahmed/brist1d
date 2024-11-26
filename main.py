import pandas as pd
import argparse
from preprocess import Preprocessor
from train_models import ModelTrainer  
import time
import os

def main():
    # =========================
    # Step 1: Parse Command-Line Arguments
    # =========================
    parser = argparse.ArgumentParser(description='Data Preprocessing and Model Training Script')
    parser.add_argument(
        '--proj_dir', 
        type=str,
        required=True
        )
    parser.add_argument(
        '--strategy', 
        type=str,
        required=False,
        choices=["mean_imp", "median_imp", "ffill_bfill_imp", "linear_interp_imp", "kalman_imp", "mice_imp", "knn_imp"],
        help='Imputation strategy to use. Choices are: mean_imp, median_imp, ffill_bfill_imp, linear_interp_imp, kalman_imp, mice_imp, knn_imp'
    )
    args = parser.parse_args()
    strategy = args.strategy
    proj_dir = args.proj_dir

    # =========================
    # Step 2: Load Data 
    # =========================
    print("Loading data...")
    
    # proj_dir = '/Users/mansoor/Documents/GSU/Coursework/AML/Project/'
    # proj_dir = "/home/mahmed76/Documents/Mansoor/Courses/AML/"

    data_dir = proj_dir + 'data/'
    results_dir = proj_dir + "/brist1d/results/"
    # check if the directories exist, if not then create them
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    if not os.path.isdir(results_dir+"predictions/"):
        os.mkdir(results_dir+"predictions/")
    if not os.path.isdir(results_dir+"performance/"):
        os.mkdir(results_dir+"performance/")
    
    # Load datasets
    train_df = pd.read_csv(data_dir + 'train.csv')
    test_df = pd.read_csv(data_dir + 'test.csv')

    print("Data loaded successfully.\n")

    # =========================
    # Step 3: Initialize Preprocessor and Apply Selected Preprocessing Strategy
    # =========================

    preprocessor = Preprocessor(target_column='bg+1:00') 
    model_trainer = ModelTrainer()

    # Define feature columns by excluding target and non-feature columns
    feature_cols = [col for col in train_df.columns if col not in ['bg+1:00', 'id', 'p_num', 'time']]
    
    # Extract features and target
    X_train = train_df[feature_cols]
    y_train = train_df['bg+1:00']
    X_test = test_df[feature_cols]
    test_ids = test_df['id']
    # exclude the carbs and activity features because of very high percentage of missing values
    X_train = X_train.loc[:, ~X_train.columns.str.startswith('activity-')]
    X_test = X_test.loc[:, ~X_test.columns.str.startswith('activity-')]
    X_train = X_train.loc[:, ~X_train.columns.str.startswith('carbs-')]
    X_test = X_test.loc[:, ~X_test.columns.str.startswith('carbs-')]
    X_train = X_train.loc[:, ~X_train.columns.str.startswith('insulin-')]
    X_test = X_test.loc[:, ~X_test.columns.str.startswith('insulin-')]


    # Map strategy names to Preprocessor methods
    strategy_methods = {
        "mean_imp": {"method": preprocessor.mean_imp, "name": "Mean Imputation"},
        "median_imp": {"method": preprocessor.median_imp, "name": "Median Imputation"}, 
        "ffill_bfill_imp": {"method": preprocessor.ffill_bfill_imp, "name": "Forward-Backward Imputation"}, 
        "linear_interp_imp": {"method": preprocessor.linear_interp_imp, "name": "Linear Interpolation"},
        "kalman_imp": {"method": preprocessor.kalman_imp, "name": "Kalman Filter Imputation"}, 
        "mice_imp": {"method": preprocessor.mice_imp, "name": "Multilpe Imputation by Chained Equations"},  
        "knn_imp": {"method": preprocessor.knn_imp, "name": "KNN Imputation"},  
    }

    # Check if the selected strategy exists
    if strategy != None:
        if strategy not in strategy_methods:
            print(f"Error: Strategy '{strategy}' is not recognized.")
            return
        else:
            # Apply the selected preprocessing strategy
            preprocess_func = strategy_methods[strategy]
            start_time = time.time()
            if strategy == "knn_imp":
                X_train_processed, X_test_processed = preprocess_func(X_train, X_test, n_neighbors=5)
            elif strategy == "mice_imp":
                X_train_processed, X_test_processed = preprocess_func(X_train, X_test, max_iter=10, random_state=42)
            else:
                X_train_processed, X_test_processed = preprocess_func(X_train, X_test)
            preprocessing_time = time.time() - start_time
            print(f"Preprocessing Strategy {strategy} completed in {preprocessing_time} seconds..\n")

            model_trainer.train_test_save_models(X_train_processed, X_test_processed, y_train, test_ids, strategy,
                                                 strategy_methods[strategy]["name"], preprocessing_time, results_dir)
            print("\nPreprocessing, training, evaluation, and prediction completed successfully.")

    else:
        # run all data imputation techniques and train each model on corresponding data
        for strategy in strategy_methods:
            preprocess_func = strategy_methods[strategy]
            print(f"Started Selected Preprocessing Strategy: {strategy}\n")
            start_time = time.time()
            X_train_processed, X_test_processed = preprocess_func(X_train, X_test)
            preprocessing_time = time.time() - start_time
            print(f"Preprocessing Strategy {strategy} completed in {preprocessing_time} seconds..\n")
            model_trainer.train_test_save_models(X_train_processed, X_test_processed, y_train, test_ids, strategy, 
                                                 strategy_methods[strategy]["name"], preprocessing_time, results_dir)
            print("\nPreprocessing, training, evaluation, and prediction completed successfully.")


    
if __name__ == "__main__":
    main()
