"""Client module for making predictions using ML models."""

import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from tritonclient.http import InferenceServerClient


def get_client():
    """Get a Triton inference server client.

    Returns:
        InferenceServerClient: Triton inference server client.
    """
    return InferenceServerClient(url="localhost:8080")


def prepare_data(data_path):
    """Prepare data for prediction.

    Args:
        data_path: Path to the data file.

    Returns:
        Tuple containing feature matrix (X) and target vector (y).
    """
    df = pd.read_csv(data_path)
    X = df.drop("Survived", axis=1).astype(np.float32).values
    y = df["Survived"].values
    return X, y


def get_prediction(features: Dict[str, Union[int, float, str]]) -> Dict:
    """Get prediction from the model.

    Args:
        features: Dictionary containing feature values for prediction.

    Returns:
        Dictionary containing prediction results.
    """
    time.sleep(0.1)  # Simulating model inference time
    return {"prediction": 1, "probability": 0.8}


def process_single_request(features: Dict[str, Union[int, float, str]]) -> Dict:
    """Process a single prediction request.

    Args:
        features: Dictionary containing feature values for prediction.

    Returns:
        Dictionary containing prediction results and metadata.
    """
    start_time = time.time()
    prediction = get_prediction(features)
    end_time = time.time()

    return {
        "prediction": prediction,
        "latency": end_time - start_time,
        "timestamp": end_time,
    }


def process_batch_requests(
    features_list: List[Dict[str, Union[int, float, str]]]
) -> List[Dict]:
    """Process a batch of prediction requests.

    Args:
        features_list: List of dictionaries containing feature values for predictions.

    Returns:
        List of dictionaries containing prediction results and metadata.
    """
    results = []
    for features in features_list:
        result = process_single_request(features)
        results.append(result)
    return results


def main():
    """Run the client application."""
    # Configuration
    data_path = "data/test.csv"

    # Load and prepare data
    X, y = prepare_data(data_path)

    # Make predictions
    results = []

    for x in X:
        result = process_single_request({"features": x.tolist()})
        results.append(result)

    print(f"Processed {len(results)} requests")


if __name__ == "__main__":
    main()
