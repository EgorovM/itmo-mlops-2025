# Homework 4: Experiment Tracking and Model Comparison

This homework demonstrates the use of MLflow for experiment tracking and DVC for data version control.

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize DVC:
```bash
dvc init
```

3. Start MLflow tracking server:
```bash
mlflow ui
```
The MLflow UI will be available at http://localhost:5000

## Project Structure

```
homework4/
├── data/           # Data directory (managed by DVC)
├── experiments/    # MLflow experiments code
├── models/        # Saved model artifacts
└── reports/       # Analysis reports and visualizations
```

## Using MLflow for Experiment Tracking

MLflow is used to track:
- Model parameters
- Model metrics
- Model artifacts
- Dataset information

Example of logging an experiment:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("model_name", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## Using DVC for Data Version Control

1. Add data to DVC:
```bash
dvc add data/dataset.csv
```

2. Push data to remote storage (if configured):
```bash
dvc push
```

3. Pull data from remote storage:
```bash
dvc pull
```

## Experiment Results

The experiments compare different models:
1. Random Forest
2. XGBoost
3. LightGBM

Detailed results and comparisons can be found in the `reports/` directory. 