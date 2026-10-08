import os
import json
import pytest
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model

def test_deploy_model():
    preprocess_data()
    train_model()
    evaluate_model()
    manifest = deploy_model()

    prod_model = "production_model/model.joblib"
    prod_scaler = "production_model/scaler.joblib"
    prod_metrics = "production_model/metrics.json"
    prod_manifest = "production_model/model_manifest.json"

    assert os.path.exists(prod_model), "Production model artifact not found"
    assert os.path.exists(prod_scaler), "Production scaler artifact not found"
    assert os.path.exists(prod_metrics), "Production metrics artifact not found"
    assert os.path.exists(prod_manifest), "Production manifest not found"

    with open(prod_manifest, "r") as f:
        data = json.load(f)

    assert data.get("status") == "DEPLOYED"
    assert data.get("version") == "v1.0.0"
    assert data.get("metrics", {}).get("passed_quality_gate") is True
