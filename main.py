import pandas as pd
import re
from preprocess import Preprocessor
from train_models import ModelTrainer  # Ensure you have the train_models.py as previously defined

def main():
    # =========================
    # Step 1: Load Data Externally
    # =========================
    print("Loading data...")
    
    # Load data
    data_dir = '/Users/mansoor/Documents/GSU/Coursework/AML/Project/orig-kaggle/'
    results_dir = "/Users/mansoor/Documents/GSU/Coursework/AML/Project/brist1d/results/"
    train_df = pd.read_csv(data_dir+'train.csv')
    test_df = pd.read_csv(data_dir+'test.csv')

    print("Data loaded and categorical features converted.\n")


    # =========================
    # Step 2: Initialize Preprocessor
    # =========================
    preprocessor = Preprocessor(target_column='bg+1:00')  # Ensure target column is sanitized

    # =========================
    # Step 3: Choose and Apply Preprocessing Strategy
    # =========================
    # Define feature columns by excluding target and non-feature columns
    feature_cols = [col for col in train_df.columns if col not in ['bg+1:00', 'id', 'p_num', 'time']]
    
    # Extract features and target
    X_train = train_df[feature_cols]
    y_train = train_df['bg+1:00']
    X_test = test_df[feature_cols]
    test_ids = test_df['id']

    # Strategy Selection: Uncomment the desired strategy
    # Strategy 1: Mean, Median, and Constant Imputation
    X_train_processed, X_test_processed = preprocessor.preprocess_strategy1(X_train, X_test)
    print("Selected Preprocessing Strategy 1.\n")

    # Strategy 2: Median, Mean, and Constant Imputation
    # X_train_processed, X_test_processed = preprocessor.preprocess_strategy2(X_train, X_test)
    # print("Selected Preprocessing Strategy 2.\n")

    # Strategy 3: Mean Imputation for All Features
    # X_train_processed, X_test_processed = preprocessor.preprocess_strategy3(X_train, X_test)
    # print("Selected Preprocessing Strategy 3.\n")

    # Strategy 4: K-Nearest Neighbors (KNN) Imputation
    # X_train_processed, X_test_processed = preprocessor.preprocess_strategy4(X_train, X_test, n_neighbors=5)
    # print("Selected Preprocessing Strategy 4.\n")

    # Strategy 5: Iterative Imputer (MICE) Imputation
    # X_train_processed, X_test_processed = preprocessor.preprocess_strategy5(X_train, X_test, max_iter=10, random_state=42)
    # print("Selected Preprocessing Strategy 5.\n")

    # =========================
    # Step 4: Initialize and Use ModelTrainer
    # =========================
    model_trainer = ModelTrainer()

    # Split data into training and validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = model_trainer.split_data(
        X_train_processed, y_train, test_size=0.2, random_state=42
    )

    # Train and evaluate models
    model_trainer.train_and_evaluate(X_train_split, X_val_split, y_train_split, y_val_split)

    # Display results
    model_trainer.display_results()

    # Save results
    model_trainer.save_results(results_dir+'model_performance_strategy1.csv')

    # Plot results
    model_trainer.plot_results(results_dir+'model_performance_strategy1.png')

    # Retrain the best model on the entire training set
    model_trainer.retrain_best_model(X_train_processed, y_train)

    # Make predictions on the test set
    predictions = model_trainer.predict_test(X_test_processed)

    # Save predictions
    model_trainer.save_predictions(test_ids, predictions, results_dir+'test_predictions_strategy1.csv')

    print("\nPreprocessing, training, evaluation, and prediction completed successfully.")

if __name__ == "__main__":
    main()
