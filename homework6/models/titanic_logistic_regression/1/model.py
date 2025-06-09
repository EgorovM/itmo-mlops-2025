from pathlib import Path

import joblib
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_path = Path(__file__).parent / "model.pkl"
        self.model = joblib.load(model_path)

    def execute(self, requests):
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
        self.model = None
