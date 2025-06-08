import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import os

def load_model(model_path):
    return joblib.load(model_path)

def load_data(X_path):
    return pd.read_csv(X_path)

def generate_shap_explanations(model, X, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Global importance plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))

    # Individual prediction example
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.savefig(os.path.join(output_dir, "shap_individual_plot.png"))

    print(f"✅ SHAP plots saved to {output_dir}")

if __name__ == "__main__":
    model_path = "models/best_model.pkl"
    X_path = "data/processed/X_features.csv"
    output_dir = "outputs/shap"  # <--- this was missing

    model = load_model(model_path)
    X = load_data(X_path)
    generate_shap_explanations(model, X, output_dir)

