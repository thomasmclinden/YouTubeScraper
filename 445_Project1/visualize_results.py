# visualize_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, r2_score

# CONFIGURATION
DATA_DIR = "data"
MODEL_DIR = "models"
FIG_DIR = "figures"

API_PROCESSED = os.path.join(DATA_DIR, "api_processed.csv")
SCRAPED_PROCESSED = os.path.join(DATA_DIR, "scraped_processed.csv")
PERF_PATH = os.path.join(DATA_DIR, "model_performance.csv")

os.makedirs(FIG_DIR, exist_ok=True)

# Helper: load model (joblib then pickle)
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# Build performance CSV if missing
def compute_and_save_performance():
    rows = []
    # We'll evaluate available models on the matching processed dataset
    # API models
    dataset_map = {
        "api": API_PROCESSED,
        "scraped": SCRAPED_PROCESSED
    }
    for prefix in ["api", "scraped"]:
        csv_path = dataset_map[prefix]
        if not os.path.exists(csv_path):
            print(f"Dataset not found for {prefix}: {csv_path} (skipping)")
            continue

        df = pd.read_csv(csv_path)
        if "viewCount" not in df.columns:
            print(f"No viewCount column in {csv_path} (skipping)")
            continue

        # Build feature matrix: drop non-feature cols
        non_feature_cols = ["video_id", "title", "channel_title", "publish_date", "viewCount"]
        X = df[[c for c in df.columns if c not in non_feature_cols]].copy()
        y = df["viewCount"].copy()

        # Try load RF and XGB models
        rf_model = load_model(os.path.join(MODEL_DIR, f"{prefix}_rf.pkl"))
        xgb_model = load_model(os.path.join(MODEL_DIR, f"{prefix}_xgb.pkl"))

        # helper to compute metrics given model and X/y
        def eval_model(name, model):
            if model is None:
                return None
            # Align features if model records feature names
            expected = None
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
            elif hasattr(model, "get_booster"):
                try:
                    booster = model.get_booster()
                    if booster and booster.feature_names:
                        expected = list(booster.feature_names)
                except Exception:
                    expected = None
            # if expected set, align X
            X_eval = X.copy()
            if expected is not None:
                # create DataFrame with expected cols, fill missing with 0
                X_al = pd.DataFrame(0, index=X.index, columns=expected)
                common = [c for c in expected if c in X.columns]
                if common:
                    # cast to float to avoid dtype warnings
                    X_al.loc[:, common] = X.loc[:, common].astype(float).values
                X_eval = X_al
            else:
                # fallback: keep X as-is (may error)
                X_eval = X
            try:
                preds = model.predict(X_eval)
                mae = mean_absolute_error(y, preds)
                r2 = r2_score(y, preds)
                return {"mae": mae, "r2": r2}
            except Exception as e:
                print(f"Failed to evaluate {name} on {prefix} dataset: {e}")
                return None

        rf_metrics = eval_model("rf", rf_model)
        xgb_metrics = eval_model("xgb", xgb_model)

        row = {
            "label": prefix.capitalize(),
            "mae_rf": rf_metrics["mae"] if rf_metrics else np.nan,
            "r2_rf": rf_metrics["r2"] if rf_metrics else np.nan,
            "mae_xgb": xgb_metrics["mae"] if xgb_metrics else np.nan,
            "r2_xgb": xgb_metrics["r2"] if xgb_metrics else np.nan
        }
        rows.append(row)

    if rows:
        perf_df = pd.DataFrame(rows)
        perf_df.to_csv(PERF_PATH, index=False)
        print(f"Saved model performance summary to {PERF_PATH}")
        return perf_df
    else:
        print("No performance rows generated (no datasets/models available).")
        return None

# If performance file not present, compute it
if not os.path.exists(PERF_PATH):
    print(f"{PERF_PATH} not found. Attempting to compute performance from available models/datasets...")
    perf_df = compute_and_save_performance()
    if perf_df is None:
        raise FileNotFoundError(f"Could not compute model performance; {PERF_PATH} missing and no models/datasets available.")
else:
    perf_df = pd.read_csv(PERF_PATH)
    print("Loaded model performance summary:")
    print(perf_df)

# Plot model performance summaries (if available)
if perf_df is not None and not perf_df.empty:
    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(perf_df["label"]))

    plt.bar(x - bar_width/2, perf_df["r2_rf"], bar_width, label="Random Forest (R²)")
    plt.bar(x + bar_width/2, perf_df["r2_xgb"], bar_width, label="XGBoost (R²)")
    plt.xticks(x, perf_df["label"])
    plt.ylabel("R² Score")
    plt.title("Model Performance Comparison (Higher = Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_r2_comparison.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x - bar_width/2, perf_df["mae_rf"], bar_width, label="Random Forest (MAE)")
    plt.bar(x + bar_width/2, perf_df["mae_xgb"], bar_width, label="XGBoost (MAE)")
    plt.xticks(x, perf_df["label"])
    plt.ylabel("Mean Absolute Error (Lower = Better)")
    plt.title("Model Error Comparison (Lower = Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_mae_comparison.png"))
    plt.close()
    print("Saved model performance comparison charts.")
else:
    print("No performance data to plot.")

# Plot top features for models (if available)
def plot_top_features(model_path, dataset_label, top_n=10):
    model = load_model(model_path)
    if model is None:
        return
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

for name in ["api_rf", "api_xgb", "scraped_rf", "scraped_xgb"]:
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        plot_top_features(path, name.upper())
    else:
        print(f"Model file not found: {path}")

# Correlation heatmaps & engagement distribution
for dataset in [API_PROCESSED, SCRAPED_PROCESSED]:
    if not os.path.exists(dataset):
        continue
    df = pd.read_csv(dataset)
    numeric_df = df.select_dtypes(include=[float, int])
    if numeric_df.shape[1] < 2:
        continue
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation - {os.path.basename(dataset).replace('_processed.csv','').capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{os.path.basename(dataset).replace('.csv','')}_correlation.png"))
    plt.close()
    print(f"Saved correlation heatmap for {dataset}")

    # Engagement distribution
    if "engagement_rate" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["engagement_rate"], bins=30, kde=True)
        plt.title(f"Engagement Rate Distribution - {os.path.basename(dataset).replace('_processed.csv','').capitalize()}")
        plt.xlabel("Engagement Rate")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{os.path.basename(dataset).replace('.csv','')}_engagement_distribution.png"))
        plt.close()
        print(f"Saved engagement distribution for {dataset}")

# Engagement trends for API data (by category, duration, upload time)
if os.path.exists(API_PROCESSED):
    api_df = pd.read_csv(API_PROCESSED)
    # engagement by category (category column may be categoryId or category; try both)
    category_col = None
    for c in ["category", "categoryId", "category_id"]:
        if c in api_df.columns:
            category_col = c
            break

    if category_col and "engagement_rate" in api_df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_col, y="engagement_rate", data=api_df, ci=None)
        plt.title("Average Engagement Rate by Category (API)")
        plt.xlabel("Category")
        plt.ylabel("Engagement Rate")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "api_engagement_by_category.png"))
        plt.close()
        print("Saved api_engagement_by_category.png")

    # engagement vs duration (try duration_min then duration)
    duration_col = None
    if "duration_min" in api_df.columns:
        duration_col = "duration_min"
    elif "duration" in api_df.columns:
        duration_col = "duration"

    if duration_col and "engagement_rate" in api_df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=duration_col, y="engagement_rate", data=api_df, alpha=0.6)
        plt.title("Engagement Rate vs Duration (API)")
        plt.xlabel("Duration (minutes)" if duration_col == "duration_min" else "Duration")
        plt.ylabel("Engagement Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "api_engagement_vs_duration.png"))
        plt.close()
        print("Saved api_engagement_vs_duration.png")

    # engagement vs days_since_upload
    if "days_since_upload" in api_df.columns and "engagement_rate" in api_df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="days_since_upload", y="engagement_rate", data=api_df, alpha=0.6)
        plt.title("Engagement Rate vs Days Since Upload (API)")
        plt.xlabel("Days Since Upload")
        plt.ylabel("Engagement Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "api_engagement_vs_days_since_upload.png"))
        plt.close()
        print("Saved api_engagement_vs_days_since_upload.png")

print("\nAll visualizations complete. Check the 'figures/' folder.")