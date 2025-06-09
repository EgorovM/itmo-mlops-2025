import time
import numpy as np
import pandas as pd
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


def get_client():
    return InferenceServerClient(url="localhost:8080")


def prepare_sample_data():
    """Prepare sample data for performance testing"""
    np.random.seed(42)
    data = np.random.randn(50, 10).astype(np.float32)
    return data


def test_model_performance(model_name, client, data, num_requests=100):
    """Test model performance with sequential requests"""
    print(f"\nTesting {model_name} with {num_requests} requests...")
    
    latencies = []
    successful_requests = 0
    failed_requests = 0
    
    for i in range(num_requests):
        # Random batch size between 1 and 4
        batch_size = np.random.randint(1, 5)
        indices = np.random.choice(len(data), size=batch_size, replace=True)
        batch = data[indices]
        
        input_data = InferInput("input__0", batch.shape, "FP32")
        input_data.set_data_from_numpy(batch)
        outputs = [InferRequestedOutput("output__0")]
        
        start_time = time.time()
        try:
            response = client.infer(model_name, inputs=[input_data], outputs=outputs)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # milliseconds
            latencies.append(latency)
            successful_requests += 1
            
        except Exception as e:
            end_time = time.time()
            failed_requests += 1
            print(f"  Request {i+1} failed: {e}")
    
    if latencies:
        return {
            'model_name': model_name,
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': successful_requests / num_requests * 100,
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'total_time_ms': sum(latencies),
            'throughput_rps': successful_requests / (sum(latencies) / 1000) if sum(latencies) > 0 else 0
        }
    else:
        return {
            'model_name': model_name,
            'total_requests': num_requests,
            'successful_requests': 0,
            'failed_requests': failed_requests,
            'success_rate': 0,
            'error': 'All requests failed'
        }


def main():
    print("Starting simple performance test...")
    
    client = get_client()
    data = prepare_sample_data()
    
    models = ["titanic_logistic_regression", "titanic_random_forest"]
    results = []
    
    for model in models:
        try:
            result = test_model_performance(model, client, data, num_requests=50)
            results.append(result)
            
            print(f"\nResults for {model}:")
            print(f"  Success rate: {result['success_rate']:.1f}%")
            print(f"  Average latency: {result['avg_latency_ms']:.2f} ms")
            print(f"  P95 latency: {result['p95_latency_ms']:.2f} ms")
            print(f"  Throughput: {result['throughput_rps']:.2f} requests/second")
            
        except Exception as e:
            print(f"Error testing {model}: {e}")
            results.append({
                'model_name': model,
                'error': str(e)
            })
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv('performance_report.csv', index=False)
        print(f"\nPerformance report saved to performance_report.csv")
        
        print("\n=== PERFORMANCE SUMMARY ===")
        for result in results:
            if 'error' not in result:
                print(f"{result['model_name']}:")
                print(f"  Throughput: {result['throughput_rps']:.2f} RPS")
                print(f"  Average latency: {result['avg_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main() 