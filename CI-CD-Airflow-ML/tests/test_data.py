import os
import pandas as pd
import pytest
from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    train_shape, test_shape = preprocess_data()
    
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    scaler_path = "models/scaler.joblib"

    assert os.path.exists(train_path), "Train CSV was not created"
    assert os.path.exists(test_path), "Test CSV was not created"
    assert os.path.exists(scaler_path), "Scaler artifact was not created"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Verify exact required shapes: Train (120, 5), Test (30, 5)
    assert train_df.shape == (120, 5), f"Unexpected train shape: {train_df.shape}"
    assert test_df.shape == (30, 5), f"Unexpected test shape: {test_df.shape}"

    # Verify no missing / NaN values
    assert train_df.isnull().sum().sum() == 0, "Train dataset contains null values"
    assert test_df.isnull().sum().sum() == 0, "Test dataset contains null values"
