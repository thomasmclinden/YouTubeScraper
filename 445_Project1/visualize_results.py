import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# CONFIGURATION
DATA_DIR = "data"
MODEL_DIR = "models"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# LOAD MODEL PERFORMANCE SUMMARY
performance_path = os.path.join(DATA_DIR, "model_performance.csv")

if not os.path.exists(performance_path):
    raise FileNotFoundError(f"{performance_path} not found. Run train_models.py first.")

results_df = pd.read_csv(performance_path)
print("Loaded model performance summary:")
print(results_df)

# VISUALIZE MODEL PERFORMANCE
plt.figure(figsize=(8, 5))
bar_width = 0.35
x = np.arange(len(results_df["label"]))

plt.bar(x - bar_width/2, results_df["r2_rf"], bar_width, label="Random Forest (R²)")
plt.bar(x + bar_width/2, results_df["r2_xgb"], bar_width, label="XGBoost (R²)")
plt.xticks(x, results_df["label"])
plt.ylabel("R² Score")
plt.title("Model Performance Comparison (Higher = Better)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_r2_comparison.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, results_df["mae_rf"], bar_width, label="Random Forest (MAE)")
plt.bar(x + bar_width/2, results_df["mae_xgb"], bar_width, label="XGBoost (MAE)")
plt.xticks(x, results_df["label"])
plt.ylabel("Mean Absolute Error (Lower = Better)")
plt.title("Model Error Comparison (Lower = Better)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_mae_comparison.png"))
plt.close()

# VISUALIZE FEATURE IMPORTANCES (OPTIONAL)
def plot_top_features(model_path, dataset_label, top_n=10):
    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        print(f"{dataset_label}: No feature_importances_ attribute, skipping.")
        return

    importances = model.feature_importances_
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        print(f"{dataset_label}: Could not find feature names.")
        return

    sorted_idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 5))
    plt.barh(range(top_n), importances[sorted_idx][::-1], align="center")
    plt.yticks(range(top_n), np.array(feature_names)[sorted_idx][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Features - {dataset_label}")
    plt.tight_layout()
    filename = f"{dataset_label.lower()}_top_features.png"
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()
    print(f"Saved {filename}")


# Plot top features for each model
for name in ["api_rf", "api_xgb", "scraped_rf", "scraped_xgb"]:
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        plot_top_features(path, name.upper())
    else:
        print(f"Model file not found: {path}")

# CORRELATION HEATMAP
for dataset in ["api_processed.csv", "scraped_processed.csv"]:
    csv_path = os.path.join(DATA_DIR, dataset)
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[float, int])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation - {dataset.replace('_processed.csv','').capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{dataset.replace('.csv','')}_correlation.png"))
    plt.close()
    print(f"Saved correlation heatmap for {dataset}")

# ENGAGEMENT DISTRIBUTION VISUALIZATION
for dataset in ["api_processed.csv", "scraped_processed.csv"]:
    csv_path = os.path.join(DATA_DIR, dataset)
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    if "engagement_rate" not in df.columns:
        continue

    plt.figure(figsize=(8, 5))
    sns.histplot(df["engagement_rate"], bins=30, kde=True)
    plt.title(f"Engagement Rate Distribution - {dataset.replace('_processed.csv','').capitalize()}")
    plt.xlabel("Engagement Rate (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{dataset.replace('.csv','')}_engagement_distribution.png"))
    plt.close()
    print(f"Saved engagement distribution for {dataset}")

print("\nAll visualizations complete. Check the 'figures/' folder!")