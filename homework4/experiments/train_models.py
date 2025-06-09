import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import os
import json

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Create directories
os.makedirs(MLRUNS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri("file://" + MLRUNS_DIR)
mlflow.set_experiment("model_comparison")

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    # For this example, we'll use the breast cancer dataset
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model."""
    with mlflow.start_run(run_name="RandomForest"):
        # Set parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return metrics

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost model."""
    with mlflow.start_run(run_name="XGBoost"):
        # Set parameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        return metrics

def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train and evaluate LightGBM model."""
    with mlflow.start_run(run_name="LightGBM"):
        # Set parameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.lightgbm.log_model(model, "model")
        
        return metrics

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train and evaluate models
    results = {
        "RandomForest": train_random_forest(X_train, X_test, y_train, y_test),
        "XGBoost": train_xgboost(X_train, X_test, y_train, y_test),
        "LightGBM": train_lightgbm(X_train, X_test, y_train, y_test)
    }
    
    # Save results to a JSON file
    results_file = os.path.join(REPORTS_DIR, "model_comparison.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()