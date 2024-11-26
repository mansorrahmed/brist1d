import pandas as pd
import argparse
from preprocess import Preprocessor
from train_models import ModelTrainer  
import time
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
        required=True,
        choices=[
            "mean_med_const_imp_1",
            "mean_med_const_imp_2",
            "mean_imp_all_features",
            "knn_imp",
            "mice_imp"
        ],
        help='Imputation strategy to use. Choices are: mean_med_const_imp_1, mean_med_const_imp_2, mean_imp_all_features, knn_imp, mice_imp'
    )
    args = parser.parse_args()
    strategy_type = args.strategy
    proj_dir = args.proj_dir

    # =========================
    # Step 2: Load Data 
    # =========================
    print("Loading data...")
    
    # Define project directories
    # proj_dir = '/Users/mansoor/Documents/GSU/Coursework/AML/Project/'
    # proj_dir = "/home/mahmed76/Documents/Mansoor/Courses/AML/"

    data_dir = proj_dir + 'data/'
    results_dir = proj_dir + "/brist1d/results/"

    # Load datasets
    train_df = pd.read_csv(data_dir + 'train.csv')
    test_df = pd.read_csv(data_dir + 'test.csv')

    print("Data loaded successfully.\n")

    # =========================
    # Step 3: Initialize Preprocessor and Apply Selected Preprocessing Strategy
    # =========================

    preprocessor = Preprocessor(target_column='bg+1:00') 

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
        "mean_med_const_imp_1": preprocessor.preprocess_strategy1,
        "mean_med_const_imp_2": preprocessor.preprocess_strategy2,
        "mean_imp_all_features": preprocessor.preprocess_strategy3,
        "knn_imp": preprocessor.preprocess_strategy4,
        "mice_imp": preprocessor.preprocess_strategy5
    }

    # Check if the selected strategy exists
    if strategy_type not in strategy_methods:
        print(f"Error: Strategy '{strategy_type}' is not recognized.")
        print("Available strategies are:")
        for strat in strategy_methods.keys():
            print(f" - {strat}")
        return

    # Apply the selected preprocessing strategy
    preprocess_func = strategy_methods[strategy_type]
    print(f"Started Selected Preprocessing Strategy: {strategy_type}\n")
    start_time = time.time()

    if strategy_type == "knn_imp":
        # Example: You can set n_neighbors as needed or make it another argument
        X_train_processed, X_test_processed = preprocess_func(X_train, X_test, n_neighbors=5)
    elif strategy_type == "mice_imp":
        # Example: You can set max_iter and random_state as needed or make them arguments
        X_train_processed, X_test_processed = preprocess_func(X_train, X_test, max_iter=10, random_state=42)
    else:
        X_train_processed, X_test_processed = preprocess_func(X_train, X_test)

    preprocessing_time = time.time() - start_time
    print(f"Preprocessing Strategy {strategy_type} completed in {preprocessing_time} seconds..\n")

    # =========================
    # Step 4: Initialize and Use ModelTrainer
    # =========================
    model_trainer = ModelTrainer()
    model_trainer.train_test_save_models(
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        y_train=y_train,
        test_ids=test_ids,
        strategy_type=strategy_type,
        preprocessing_time=preprocessing_time,
        results_dir=results_dir
    )

    print("\nPreprocessing, training, evaluation, and prediction completed successfully.")

if __name__ == "__main__":
    main()
