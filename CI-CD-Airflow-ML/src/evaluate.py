import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(test_data_path="data/processed/test.csv", 
                   model_path="models/model.joblib", 
                   models_dir="models",
                   threshold=0.85):
    """
    Evaluates the trained model against test data and verifies the Quality Gate.
    """
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    test_df = pd.read_csv(test_data_path)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted"))
    rec = float(recall_score(y_test, y_pred, average="weighted"))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    passed_quality_gate = bool(acc >= threshold)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy_threshold": threshold,
        "passed_quality_gate": passed_quality_gate
    }

    os.makedirs(models_dir, exist_ok=True)
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    status_str = "PASSED" if passed_quality_gate else "FAILED"
    print("Model Evaluation Metrics:")
    print(f"- Accuracy: {metrics['accuracy']:.4f} (Threshold: {threshold})")
    print(f"- Quality Gate Status: {status_str}")

    if not passed_quality_gate:
        raise ValueError(f"Quality gate failed: Accuracy {acc:.4f} is below threshold {threshold}")

    return metrics

if __name__ == "__main__":
    evaluate_model()
