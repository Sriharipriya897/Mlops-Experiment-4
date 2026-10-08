import os
import joblib
import numpy as np

def predict(sample_input=None, prod_dir="production_model"):
    """
    Loads production model and scaler to perform inference on input samples.
    """
    model_path = os.path.join(prod_dir, "model.joblib")
    scaler_path = os.path.join(prod_dir, "scaler.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found at {model_path}. Please run deployment first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Production scaler not found at {scaler_path}. Please run deployment first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if sample_input is None:
        sample_input = [[5.1, 3.5, 1.4, 0.2]]

    sample_array = np.array(sample_input)
    scaled_input = scaler.transform(sample_array)

    predictions = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "class_id": int(pred),
            "probabilities": [round(float(p), 4) for p in prob]
        })

    print(f"Sample Input: {sample_input}")
    print(f"Prediction Output: {results}")
    return results

if __name__ == "__main__":
    predict()
