"""Logistic Regression model for Titanic survival prediction."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import triton_python_backend_utils as pb_utils
from sklearn.linear_model import LogisticRegression


class TritonPythonModel:
    """Titanic survival prediction model using Logistic Regression."""

    def initialize(self, args):
        """Initialize the model."""
        model_path = Path(__file__).parent / "model.pkl"
        self.model = joblib.load(model_path)

    def execute(self, requests):
        """Execute the model on the input requests."""
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
            input_data = input_tensor.as_numpy()

            predictions = self.model.predict(input_data)

            output_tensor = pb_utils.Tensor(
                "output__0", predictions.astype(np.int64).reshape([-1, 1])
            )

            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        """Finalize the model."""
        self.model = None


class TitanicModel:
    """Titanic survival prediction model using Logistic Regression."""

    def __init__(self):
        """Initialize the model."""
        self.model = LogisticRegression()
        self.load_model()

    def load_model(self):
        """Load the pre-trained model parameters."""
        model_path = os.path.join(os.path.dirname(__file__), "model.json")
        with open(model_path, "r") as f:
            model_params = json.load(f)

        self.model.classes_ = np.array(model_params["classes"])
        self.model.coef_ = np.array(model_params["coefficients"])
        self.model.intercept_ = np.array(model_params["intercept"])

    def predict(self, X):
        """Make predictions on the input data.

        Args:
            X: Input features as a numpy array.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X)
