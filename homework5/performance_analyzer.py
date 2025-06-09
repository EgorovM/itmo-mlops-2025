import time
import numpy as np
import pandas as pd
import concurrent.futures
import statistics
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


def get_client():
    return InferenceServerClient(url="localhost:8080")


def prepare_sample_data():
    """Prepare sample data for performance testing"""
    # Create sample data with 10 features (matching our model input)
    np.random.seed(42)
    data = np.random.randn(100, 10).astype(np.float32)
    return data


def send_inference_request(client, model_name, data_batch):
    """Send a single inference request"""
    start_time = time.time()
    
    input_data = InferInput("input__0", data_batch.shape, "FP32")
    input_data.set_data_from_numpy(data_batch)
    
    outputs = [InferRequestedOutput("output__0")]
    
    try:
        response = client.infer(model_name, inputs=[input_data], outputs=outputs)
        end_time = time.time()
        
        result = response.as_numpy("output__0")
        latency = (end_time - start_time) * 1000  # в миллисекундах
        
        return {
            'success': True,
            'latency': latency,
            'result_shape': result.shape
        }
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'latency': (end_time - start_time) * 1000,
            'error': str(e)
        }


def run_concurrent_requests(client, model_name, data, concurrency, duration_seconds):
    """Run concurrent requests for a specified duration"""
    
    results = []
    start_time = time.time()
    
    def worker():
        local_results = []
        while time.time() - start_time < duration_seconds:
            # Select random batch from data
            batch_size = np.random.randint(1, 5)  # Random batch size 1-4
            indices = np.random.choice(len(data), size=batch_size, replace=True)
            batch = data[indices]
            
            result = send_inference_request(client, model_name, batch)
            local_results.append(result)
            
        return local_results
    
    # Run concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    
    return results


def analyze_results(results):
    """Analyze performance results"""
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if not successful_results:
        return {
            'total_requests': len(results),
            'successful_requests': 0,
            'failed_requests': len(failed_results),
            'success_rate': 0.0
        }
    
    latencies = [r['latency'] for r in successful_results]
    
    return {
        'total_requests': len(results),
        'successful_requests': len(successful_results),
        'failed_requests': len(failed_results),
        'success_rate': len(successful_results) / len(results) * 100,
        'avg_latency_ms': statistics.mean(latencies),
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
        'p50_latency_ms': statistics.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_rps': len(successful_results) / max(duration_seconds, 1)
    }


def performance_test(model_name, test_duration=5):
    """Run comprehensive performance test for a model"""
    print(f"\n=== Performance Analysis for {model_name} ===")
    
    client = get_client()
    data = prepare_sample_data()
    
    # Test different concurrency levels (reduced)
    concurrency_levels = [1, 4]
    results_by_concurrency = {}
    
    for concurrency in concurrency_levels:
        print(f"\nTesting with concurrency level: {concurrency}")
        
        results = run_concurrent_requests(
            client, model_name, data, concurrency, test_duration
        )
        
        analysis = analyze_results(results)
        results_by_concurrency[concurrency] = analysis
        
        print(f"  Total requests: {analysis['total_requests']}")
        print(f"  Success rate: {analysis['success_rate']:.1f}%")
        print(f"  Average latency: {analysis['avg_latency_ms']:.2f} ms")
        print(f"  P95 latency: {analysis['p95_latency_ms']:.2f} ms")
        print(f"  Throughput: {analysis['throughput_rps']:.2f} requests/second")
    
    return results_by_concurrency


def main():
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
            row = {
                'model_name': model,
                'concurrency': concurrency,
                **metrics
            }
            performance_data.append(row)
    
    if performance_data:
        df = pd.DataFrame(performance_data)
        df.to_csv('performance_report.csv', index=False)
        print(f"\nDetailed performance report saved to performance_report.csv")
        
        # Print summary
        print("\n=== PERFORMANCE SUMMARY ===")
        for model in models:
            model_data = df[df['model_name'] == model]
            if not model_data.empty:
                best_throughput = model_data['throughput_rps'].max()
                best_concurrency = model_data.loc[model_data['throughput_rps'].idxmax(), 'concurrency']
                avg_latency_at_best = model_data.loc[model_data['throughput_rps'].idxmax(), 'avg_latency_ms']
                
                print(f"{model}:")
                print(f"  Best throughput: {best_throughput:.2f} RPS at concurrency {best_concurrency}")
                print(f"  Latency at best throughput: {avg_latency_at_best:.2f} ms")


if __name__ == "__main__":
    duration_seconds = 5  # Global variable for worker function
    main() 