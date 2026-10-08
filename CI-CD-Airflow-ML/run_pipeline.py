import sys
import subprocess
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model
from src.predict import predict

def run_pipeline():
    print(">>> STAGE 1: Running Unit Tests (CI)...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        print("STAGE 1 FAILED: Unit tests did not pass.")
        sys.exit(1)
    print("pytest tests/ -v -> PASSED")

    print(">>> STAGE 2: Preprocessing Data...")
    preprocess_data()

    print(">>> STAGE 3: Training Model...")
    train_model()

    print(">>> STAGE 4: Evaluating Model & Validating Quality Gate...")
    evaluate_model()

    print(">>> STAGE 5: Deploying Validated Model...")
    deploy_model()

    print(">>> STAGE 6: Running Production Inference Smoke Test...")
    predict()

    print("SUCCESS: All pipeline stages completed successfully!")

if __name__ == "__main__":
    run_pipeline()
