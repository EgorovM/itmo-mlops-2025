"""Random Forest model for Titanic survival prediction."""

from pathlib import Path

import joblib
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton model wrapper for Random Forest Titanic survival prediction."""

    def initialize(self, args):
        """Initialize the model.

        Args:
            args: Initialization arguments from Triton.
        """
        model_path = Path(__file__).parent / "model.pkl"
        self.model = joblib.load(model_path)

    def execute(self, requests):
        """Execute the model on the input requests.

        Args:
            requests: List of inference requests.

        Returns:
            List of inference responses.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
            x = input_tensor.as_numpy()

            y = self.model.predict(x)
            output_tensor = pb_utils.Tensor("output__0", y.astype(np.float32))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """Clean up resources when the model is unloaded."""
        self.model = None
