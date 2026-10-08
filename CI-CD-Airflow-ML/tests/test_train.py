import os
import joblib
import json
import pytest
from src.data_preprocessing import preprocess_data
from src.train import train_model

def test_train_model():
    # Ensure data is ready
    preprocess_data()

    model = train_model()
    model_path = "models/model.joblib"
    metadata_path = "models/training_metadata.json"

    assert os.path.exists(model_path), "Model artifact model.joblib not found"
    assert os.path.exists(metadata_path), "Training metadata JSON not found"

    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict"), "Loaded model has no predict method"
    assert len(loaded_model.classes_) == 3, "Model classes count mismatch (Iris has 3 classes)"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    assert metadata.get("model_type") == "RandomForestClassifier"
