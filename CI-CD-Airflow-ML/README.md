# CI-CD-Airflow-ML: Automated CI/CD and Airflow ML Pipeline Deployment

## 1. Project Objective
This project implements an end-to-end automated MLOps pipeline for multi-class Iris classification using:
- **Scikit-Learn & RandomForestClassifier**: Feature scaling (StandardScaler) and classification modeling.
- **Pytest**: Automated CI testing verifying data schemas, model integrity, quality gate logic, and deployment manifests.
- **Model Quality Gate**: Enforces strict performance validation (`Accuracy >= 0.85`) before promoting models.
- **Production Artifact Promotion**: Automatic versioning and deployment manifest generation (`production_model/model_manifest.json`).
- **Apache Airflow**: Workflow orchestration with DAG-based task execution (`preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task`).
- **GitHub Actions**: Automated CI/CD pipeline triggered on code push/pull requests, archiving production model artifacts.
- **Inference Smoke Testing**: Real-time prediction service for incoming requests.

---

## 2. Project Directory Structure
```
CI-CD-Airflow-ML/
├── .github/
│   └── workflows/
│       └── ml_cicd.yml          # GitHub Actions CI/CD Pipeline configuration
├── dags/
│   └── ml_pipeline_dag.py       # Apache Airflow DAG defining task execution order
├── src/
│   ├── data_preprocessing.py    # Dataset loading, cleaning, feature scaling, & split
│   ├── train.py                 # ML model training & hyperparameter metadata output
│   ├── evaluate.py              # Model evaluation & performance threshold quality gate
│   ├── deploy.py                # Model artifact promotion & production manifest generation
│   └── predict.py               # Deployed model inference service
├── tests/
│   ├── test_data.py             # Unit tests for preprocessing & dataset shapes
│   ├── test_train.py            # Unit tests for training logic & artifact output
│   ├── test_evaluate.py         # Unit tests for quality gate evaluation
│   └── test_deploy.py           # Unit tests for deployment promotion & inference
├── data/
│   └── processed/               # Processed train/test data CSV files
├── models/                      # Staging directory for model & scaler artifacts
├── production_model/            # Deployment directory containing active production model
├── requirements.txt             # Project Python dependencies
├── run_pipeline.py              # Local end-to-end pipeline execution runner
└── README.md                    # Technical documentation
```

---

## 3. Quick Start & Execution

### Setup Environment
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Automated Unit Tests (Pytest)
```powershell
pytest tests/ -v
```

### Execute End-to-End Pipeline
```powershell
python run_pipeline.py
```

### Test Production Inference
```powershell
python src/predict.py
```

---

## 4. Quality Gate & Production Promotion
- **Quality Gate Rule**: `Accuracy >= 0.85`
- **Output Metrics (`models/metrics.json`)**:
  ```json
  {
    "accuracy": 0.9333,
    "precision": 0.9333,
    "recall": 0.9333,
    "f1_score": 0.9333,
    "accuracy_threshold": 0.85,
    "passed_quality_gate": true
  }
  ```
- **Manifest (`production_model/model_manifest.json`)**:
  ```json
  {
    "status": "DEPLOYED",
    "deployment_timestamp": "...",
    "version": "v1.0.0",
    "deployed_files": [
      "model.joblib",
      "scaler.joblib",
      "metrics.json",
      "training_metadata.json"
    ],
    "metrics": {
      "accuracy": 0.9333,
      "passed_quality_gate": true
    }
  }
  ```

---

## 5. Apache Airflow Orchestration
The DAG `ml_pipeline_dag` executes tasks sequentially:
```
preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task
```
Test individual tasks via Airflow CLI:
```powershell
airflow tasks test ml_pipeline_dag preprocess_data_task 2026-01-01
airflow tasks test ml_pipeline_dag train_model_task 2026-01-01
airflow tasks test ml_pipeline_dag evaluate_model_task 2026-01-01
airflow tasks test ml_pipeline_dag deploy_model_task 2026-01-01
```

---

## 6. GitHub Actions CI/CD
On push to `main`/`master`:
1. **`unit-tests` Job**: Installs dependencies and runs `pytest tests/ -v`.
2. **`train-evaluate-deploy` Job**: Preprocesses data, trains model, evaluates quality gate, promotes model, and uploads `production_model/` as artifact `trained-production-model`.
