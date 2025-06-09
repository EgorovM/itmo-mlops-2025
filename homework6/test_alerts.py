#!/usr/bin/env python3
"""ML Monitoring Test Script.

Sends requests to models to generate metrics and potentially trigger alerts.
"""

import argparse
import time
from typing import Dict, List, TypedDict

import numpy as np
import requests

# Configuration
TRITON_URL = "http://localhost:8090"
TELEGRAM_BOT_URL = "http://localhost:5001"
MODELS = ["titanic_logistic_regression", "titanic_random_forest"]


# Type definitions
class TritonOutput(TypedDict):
    """Type for Triton model output."""

    output__0: List[float]


class TritonResponse(TypedDict, total=False):
    """Type for Triton API response.

    Contains either outputs with model predictions or an error message.
    """

    outputs: Dict[str, List[float]]
    error: str


class MLTestRunner:
    """Test runner for ML monitoring system."""

    def __init__(self) -> None:
        """Initialize MLTestRunner."""
        pass

    def send_triton_request(self, model_name: str, data: np.ndarray) -> TritonResponse:
        """Send request to Triton using HTTP API.

        Args:
            model_name: Name of the model to query.
            data: Input data for the model.

        Returns:
            Dict containing model response or error message.
            On success: {"outputs": {"output__0": [float, ...]}}
            On error: {"error": str}
        """
        url = f"{TRITON_URL}/v2/models/{model_name}/infer"

        # Ensure we have 10 features
        if data.shape[1] < 10:
            padding = np.zeros((data.shape[0], 10 - data.shape[1]), dtype=np.float32)
            data = np.concatenate([data, padding], axis=1)
        elif data.shape[1] > 10:
            data = data[:, :10]

        # Clean data
        data = np.nan_to_num(data, nan=0.0, posinf=100.0, neginf=-100.0)
        data = np.clip(data, -1000, 1000)

        payload = {
            "inputs": [
                {
                    "name": "input__0",
                    "shape": [data.shape[0], 10],
                    "datatype": "FP32",
                    "data": data.flatten().tolist(),
                }
            ]
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()  # type: ignore
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def send_requests(
        self, model_name: str, num_requests: int = 100, delay: float = 0.1
    ) -> tuple[int, int]:
        """Send multiple requests to a model.

        Args:
            model_name: Name of the model to test.
            num_requests: Number of requests to send.
            delay: Delay between requests in seconds.

        Returns:
            Tuple of (successful_requests, failed_requests).
        """
        print(f"Sending {num_requests} requests to {model_name}...")

        successful = 0
        failed = 0

        for i in range(num_requests):
            try:
                # Generate random data
                batch_size = np.random.randint(1, 5)
                data = np.random.randn(batch_size, 10).astype(np.float32)

                response = self.send_triton_request(model_name, data)

                if "error" in response:
                    failed += 1
                    print(f"  Request {i + 1} failed: {response['error']}")
                else:
                    successful += 1

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{num_requests}")

                time.sleep(delay)

            except Exception as e:
                failed += 1
                print(f"  Request {i + 1} failed: {e}")

        print(f"Completed: {successful} successful, {failed} failed")
        return successful, failed

    def generate_load(self, duration_minutes: int = 5) -> None:
        """Generate continuous load on all models.

        Args:
            duration_minutes: Duration of the load test in minutes.
        """
        print(f"Generating load for {duration_minutes} minutes...")
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            for model_name in MODELS:
                try:
                    self.send_requests(model_name, num_requests=10, delay=0.05)
                except Exception as e:
                    print(f"Error with {model_name}: {e}")

            remaining = (end_time - time.time()) / 60
            print(f"Remaining time: {remaining:.1f} minutes")
            time.sleep(5)  # Small break between cycles

    def test_telegram_bot(self) -> bool:
        """Test Telegram bot functionality.

        Returns:
            True if all tests pass, False otherwise.
        """
        print("Testing Telegram bot...")

        try:
            # Test health endpoint
            response = requests.get(f"{TELEGRAM_BOT_URL}/health")
            if response.status_code == 200:
                print("✅ Telegram bot health check passed")
                print(f"   Status: {response.json()}")
            else:
                print(f"❌ Telegram bot health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Telegram bot: {e}")
            return False

        try:
            # Test message endpoint
            response = requests.post(f"{TELEGRAM_BOT_URL}/test")
            if response.status_code == 200:
                print("✅ Test message sent successfully")
                return True
            else:
                print(f"❌ Failed to send test message: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error sending test message: {e}")
            return False

    def simulate_model_degradation(self, model_name: str) -> None:
        """Simulate model performance degradation.

        Args:
            model_name: Name of the model to degrade.
        """
        print(f"Simulating degradation for {model_name}...")

        for i in range(50):
            try:
                data = np.random.randn(10, 10).astype(np.float32)

                # Add artificial delay
                time.sleep(0.2)  # This should increase observed latency

                response = self.send_triton_request(model_name, data)

                if "error" in response:
                    print(f"  Error in degradation test {i + 1}: {response['error']}")

                if (i + 1) % 10 == 0:
                    print(f"  Degradation requests: {i + 1}/50")

            except Exception as e:
                print(f"  Error in degradation test: {e}")


def main() -> None:
    """Run the ML monitoring test script."""
    parser = argparse.ArgumentParser(description="ML Monitoring Test Script")
    parser.add_argument(
        "--action",
        choices=["load", "test-bot", "degrade", "quick-test"],
        default="quick-test",
        help="Action to perform",
    )
    parser.add_argument(
        "--duration", type=int, default=5, help="Duration for load test in minutes"
    )
    parser.add_argument(
        "--model", choices=MODELS, help="Specific model for degradation test"
    )

    args = parser.parse_args()

    runner = MLTestRunner()

    if args.action == "test-bot":
        print("🤖 Testing Telegram Bot")
        print("=" * 50)
        runner.test_telegram_bot()

    elif args.action == "load":
        print(f"🚀 Generating Load Test ({args.duration} minutes)")
        print("=" * 50)
        runner.generate_load(args.duration)

    elif args.action == "degrade":
        if not args.model:
            args.model = MODELS[0]
        print(f"📉 Simulating Model Degradation: {args.model}")
        print("=" * 50)
        runner.simulate_model_degradation(args.model)

    elif args.action == "quick-test":
        print("🧪 Quick Test of All Components")
        print("=" * 50)

        # Test Telegram bot
        print("\n1. Testing Telegram Bot...")
        runner.test_telegram_bot()

        # Test models
        print("\n2. Testing Models...")
        for model in MODELS:
            print(f"\n   Testing {model}...")
            try:
                successful, failed = runner.send_requests(
                    model, num_requests=20, delay=0.05
                )
                print(f"   ✅ {model}: {successful} successful, {failed} failed")
            except Exception as e:
                print(f"   ❌ {model}: Error - {e}")

        # Brief load test
        print("\n3. Brief Load Test (1 minute)...")
        runner.generate_load(1)

        print("\n✅ Quick test completed!")
        print("Check Grafana dashboard at http://localhost:3000")
        print("Check Prometheus at http://localhost:9090")
        print("Check Alertmanager at http://localhost:9093")


if __name__ == "__main__":
    main()
