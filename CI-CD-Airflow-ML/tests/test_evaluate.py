import os
import json
import pytest
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def test_evaluate_model():
    preprocess_data()
    train_model()
    metrics = evaluate_model()

    metrics_path = "models/metrics.json"
    assert os.path.exists(metrics_path), "metrics.json not found"

    with open(metrics_path, "r") as f:
        data = json.load(f)

    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "passed_quality_gate" in data

    # Quality Gate Check
    assert data["accuracy"] >= 0.85, f"Accuracy {data['accuracy']} failed threshold 0.85"
    assert data["passed_quality_gate"] is True, "Quality gate flag is False"
