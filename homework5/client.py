import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

def get_client():
    return InferenceServerClient(url="localhost:8080")

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("Survived", axis=1).astype(np.float32).values
    y = df["Survived"].values
    return X, y

def main():
    MODELS = [
        "titanic_logistic_regression",
        "titanic_random_forest",
    ]
    DATA_PATH = "../data/processed/titanic_val.csv"

    triton_client = get_client()
    X_test, y_test = prepare_data(DATA_PATH)

    results = []

    for model_name in MODELS:
        print(f"--- Testing model: {model_name} ---")
        
        input_data = InferInput("input__0", X_test.shape, "FP32")
        input_data.set_data_from_numpy(X_test, binary_data=True)
        
        output_data = InferRequestedOutput("output__0", binary_data=True)

        start_time = time.time()
        
        response = triton_client.infer(
            model_name=model_name,
            inputs=[input_data],
            outputs=[output_data]
        )
        
        end_time = time.time()

        predictions = response.as_numpy("output__0").flatten()
        
        total_time = end_time - start_time
        avg_time_per_request = total_time / len(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        results.append({
            "model_name": model_name,
            "total_time": total_time,
            "avg_time_per_request": avg_time_per_request,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        print(f"Total time: {total_time:.4f}s")
        print(f"Average time per request: {avg_time_per_request * 1000:.4f}ms")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 30)

    # Save report
    report_df = pd.DataFrame(results)
    report_df.to_csv("report.csv", index=False)
    print("Report saved to report.csv")


if __name__ == "__main__":
    main() 