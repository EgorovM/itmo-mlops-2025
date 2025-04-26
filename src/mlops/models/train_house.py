"""Train models on House Price dataset."""
import json
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
) -> Dict[str, Union[str, float]]:
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
    metrics: Dict[str, Union[str, float]] = {
        "model_name": model_name,
        "mse": float(mean_squared_error(y_val, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "r2": float(r2_score(y_val, y_pred)),
    }

    return metrics


def main() -> None:
    """Train different models on House Price dataset."""
    # Setup paths
    data_dir = Path("data")
    models_dir = Path("models") / "trained"
    metrics_dir = Path("metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = pd.read_csv(data_dir / "processed" / "house_train.csv")
    val_data = pd.read_csv(data_dir / "processed" / "house_val.csv")

    # Split features and target
    X_train = train_data.drop("SalePrice", axis=1)
    y_train = train_data["SalePrice"]
    X_val = val_data.drop("SalePrice", axis=1)
    y_val = val_data["SalePrice"]

    # Define models
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    }

    # Train and evaluate models
    all_metrics = []
    for name, model in models.items():
        print(f"Training {name}...")
        metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val, name)
        all_metrics.append(metrics)

        # Save model
        model_path = models_dir / f"house_{name}.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = metrics_dir / "house_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
