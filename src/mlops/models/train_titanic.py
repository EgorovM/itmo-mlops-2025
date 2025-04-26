"""Train models on Titanic dataset."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name: str) -> dict:
    """Train model and evaluate its performance.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model
        
    Returns:
        dict: Model metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision': float(precision_score(y_val, y_pred)),
        'recall': float(recall_score(y_val, y_pred)),
        'f1': float(f1_score(y_val, y_pred))
    }
    
    return metrics


def main():
    """Train different models on Titanic dataset."""
    # Setup paths
    data_dir = Path("data")
    models_dir = Path("models") / "trained"
    metrics_dir = Path("metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_data = pd.read_csv(data_dir / "processed" / "titanic_train.csv")
    val_data = pd.read_csv(data_dir / "processed" / "titanic_val.csv")
    
    # Split features and target
    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']
    X_val = val_data.drop('Survived', axis=1)
    y_val = val_data['Survived']
    
    # Define models
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    }
    
    # Train and evaluate models
    all_metrics = []
    for name, model in models.items():
        print(f"Training {name}...")
        metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val, name)
        all_metrics.append(metrics)
        
        # Save model
        model_path = models_dir / f"titanic_{name}.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = metrics_dir / "titanic_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main() 