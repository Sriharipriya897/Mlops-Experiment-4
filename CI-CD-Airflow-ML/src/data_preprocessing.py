import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_dir="data/processed", models_dir="models"):
    """
    Loads the Iris dataset, performs train-test split,
    scales feature values using StandardScaler, and saves processed datasets & scaler.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    
    feature_cols = iris.feature_names
    target_col = "target"

    X = df[feature_cols].values
    y = df[target_col].values

    # 2. Train-test split (80% train = 120 samples, 20% test = 30 samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=6
    )

    # 3. Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Save Scaler artifact to models directory
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # 5. Assemble DataFrames with scaled features + target
    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df[target_col] = y_train

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df[target_col] = y_test

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Data preprocessed successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df.shape, test_df.shape

if __name__ == "__main__":
    preprocess_data()
