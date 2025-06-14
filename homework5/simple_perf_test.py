"""Module for performing simple performance tests on ML models."""

import concurrent.futures
import statistics
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


def get_client() -> InferenceServerClient:
    """Get a Triton inference server client.

    Returns:
        InferenceServerClient: A configured client instance.
    """
    return InferenceServerClient(url="localhost:8080")


def prepare_request(model_name: str, data: np.ndarray) -> Dict:
    """Prepare a request for the inference server.

    Args:
        model_name: Name of the model to use for inference.
        data: Input data for the model.

    Returns:
        Dictionary containing the prepared request parameters.
    """
    inputs = []
    inputs.append(InferInput("input__0", data.shape, "FP32"))
    inputs[0].set_data_from_numpy(data)

    outputs = []
    outputs.append(InferRequestedOutput("output__0"))

    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
    }


def worker(model_name: str, data: np.ndarray) -> Dict:
    """Worker function for processing inference requests.

    Args:
        model_name: Name of the model to use for inference.
        data: Input data for the model.

    Returns:
        Dictionary containing inference results and timing information.
    """
    client = get_client()
    request = prepare_request(model_name, data)

    start_time = time.time()
    client.infer(**request)
    end_time = time.time()

    return {
        "latency": end_time - start_time,
        "timestamp": end_time,
        "success": True,
    }


def performance_test(
    model_name: str,
    concurrency_levels: Optional[List[int]] = None,
    test_duration: int = 60,
) -> Dict[int, Dict[str, float]]:
    """Run performance tests for a model with different concurrency levels.

    Args:
        model_name: Name of the model to test.
        concurrency_levels: List of concurrency levels to test.
        test_duration: Duration of each test in seconds.

    Returns:
        Dictionary containing performance metrics for each concurrency level.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]

    # Load test data
    df = pd.read_csv("data/test.csv")
    X = df.drop("Survived", axis=1).astype(np.float32).values

    results = {}

    for concurrency in concurrency_levels:
        print(f"\nTesting with concurrency: {concurrency}")
        start_time = time.time()
        latencies = []
        request_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            while time.time() - start_time < test_duration:
                if len(futures) < concurrency:
                    future = executor.submit(worker, model_name, X[0:1])
                    futures.append(future)

                done, futures = concurrent.futures.wait(
                    futures, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    try:
                        result = future.result()
                        if result["success"]:
                            latencies.append(result["latency"])
                            request_count += 1
                    except Exception as e:
                        print(f"Request failed: {e}")

        test_duration_actual = time.time() - start_time

        if latencies:
            avg_latency = statistics.mean(latencies) * 1000  # Convert to ms
            throughput = request_count / test_duration_actual

            results[concurrency] = {
                "avg_latency_ms": avg_latency,
                "throughput_rps": throughput,
                "total_requests": request_count,
            }

            print(f"Average latency: {avg_latency:.2f} ms")
            print(f"Throughput: {throughput:.2f} requests/second")
            print(f"Total requests: {request_count}")
        else:
            print("No successful requests completed")

    return results


def main():
    """Run the performance testing pipeline."""
    models = ["titanic_logistic_regression", "titanic_random_forest"]

    all_results = {}

    for model in models:
        try:
            results = performance_test(model, test_duration=5)  # Reduced duration
            all_results[model] = results
        except Exception as e:
            print(f"Error testing {model}: {e}")

    # Save detailed results to CSV
    performance_data = []

    for model, concurrency_results in all_results.items():
        for concurrency, metrics in concurrency_results.items():
            row = {"model_name": model, "concurrency": concurrency, **metrics}
            performance_data.append(row)

    if performance_data:
        df = pd.DataFrame(performance_data)
        df.to_csv("performance_report.csv", index=False)
        print("Detailed performance report saved to performance_report.csv")

        # Print summary
        print("\n=== PERFORMANCE SUMMARY ===")
        for model in models:
            model_data = df[df["model_name"] == model]
            if not model_data.empty:
                best_throughput = model_data["throughput_rps"].max()
                best_concurrency = model_data.loc[
                    model_data["throughput_rps"].idxmax(), "concurrency"
                ]
                avg_latency_at_best = model_data.loc[
                    model_data["throughput_rps"].idxmax(), "avg_latency_ms"
                ]

                print(f"{model}:")
                print(
                    f"  Best throughput: {best_throughput:.2f} RPS at concurrency {best_concurrency}"
                )
                print(f"  Latency at best throughput: {avg_latency_at_best:.2f} ms")


if __name__ == "__main__":
    main()
