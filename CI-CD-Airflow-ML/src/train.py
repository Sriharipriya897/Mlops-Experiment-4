import os
import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

def train_model(train_data_path="data/processed/train.csv", models_dir="models"):
    """
    Trains a RandomForestClassifier on the processed training data
    and saves the model artifact and training metadata.
    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}. Run preprocessing first.")

    train_df = pd.read_csv(train_data_path)
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    # Initialize and train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model artifact
    model_path = os.path.join(models_dir, "model.joblib")
    joblib.dump(model, model_path)

    # Save training metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
        "n_features": X_train.shape[1],
        "n_samples": X_train.shape[0],
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    metadata_path = os.path.join(models_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model trained successfully and saved to {model_path}")
    return model_path

if __name__ == "__main__":
    train_model()
