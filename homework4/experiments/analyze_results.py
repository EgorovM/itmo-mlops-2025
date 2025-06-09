import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def load_results():
    """Load results from the JSON file."""
    results_file = os.path.join(REPORTS_DIR, "model_comparison.json")
    with open(results_file, "r") as f:
        return json.load(f)

def create_comparison_dataframe(results):
    """Convert results to a DataFrame for easier visualization."""
    data = []
    for model_name, metrics in results.items():
        for metric_name, value in metrics.items():
            data.append({
                "Model": model_name,
                "Metric": metric_name,
                "Value": value
            })
    return pd.DataFrame(data)

def plot_metrics_comparison(df):
    """Create visualizations comparing model performance."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Metric", y="Value", hue="Model")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(plot_file)
    plt.close()

def generate_report(results, df):
    """Generate a markdown report with the analysis."""
    report = """# Model Comparison Report

## Overview
This report compares the performance of three different models:
- Random Forest
- XGBoost
- LightGBM

## Performance Metrics

"""
    
    # Add table of results
    report += df.pivot(index="Model", columns="Metric", values="Value").to_markdown()
    
    report += """

## Analysis

### Best Performing Model
"""
    
    # Find best model for each metric
    for metric in df["Metric"].unique():
        metric_data = df[df["Metric"] == metric]
        best_model = metric_data.loc[metric_data["Value"].idxmax()]
        report += f"\n- {metric}: {best_model['Model']} ({best_model['Value']:.4f})"
    
    report += """

## Visualization
A bar plot comparing all metrics across models can be found in `model_comparison.png`.

## Conclusion
"""
    
    # Add simple conclusion based on accuracy
    accuracy_data = df[df["Metric"] == "accuracy"]
    best_model = accuracy_data.loc[accuracy_data["Value"].idxmax()]
    report += f"\nBased on overall accuracy, the {best_model['Model']} model performed best with an accuracy of {best_model['Value']:.4f}."
    
    # Save report
    report_file = os.path.join(REPORTS_DIR, "model_comparison.md")
    with open(report_file, "w") as f:
        f.write(report)

def main():
    # Load results
    results = load_results()
    
    # Create DataFrame
    df = create_comparison_dataframe(results)
    
    # Create visualizations
    plot_metrics_comparison(df)
    
    # Generate report
    generate_report(results, df)

if __name__ == "__main__":
    main() 