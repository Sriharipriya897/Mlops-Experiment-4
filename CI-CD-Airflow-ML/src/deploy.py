import os
import json
import shutil
from datetime import datetime, timezone

def deploy_model(models_dir="models", prod_dir="production_model", version="v1.0.0"):
    """
    Validates the quality gate status and promotes model artifacts
    to the production_model directory with a deployment manifest.
    """
    metrics_path = os.path.join(models_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Evaluation metrics not found at {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if not metrics.get("passed_quality_gate", False):
        raise RuntimeError("Deployment aborted: Model did not pass the quality gate threshold.")

    os.makedirs(prod_dir, exist_ok=True)

    artifacts_to_copy = [
        "model.joblib",
        "scaler.joblib",
        "metrics.json",
        "training_metadata.json"
    ]

    deployed_files = []
    for artifact in artifacts_to_copy:
        src_file = os.path.join(models_dir, artifact)
        if os.path.exists(src_file):
            dst_file = os.path.join(prod_dir, artifact)
            shutil.copy2(src_file, dst_file)
            deployed_files.append(artifact)

    manifest = {
        "status": "DEPLOYED",
        "deployment_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": version,
        "deployed_files": deployed_files,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "passed_quality_gate": metrics.get("passed_quality_gate")
        }
    }

    manifest_path = os.path.join(prod_dir, "model_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Model successfully deployed to '{prod_dir}'")
    return manifest

if __name__ == "__main__":
    deploy_model()
