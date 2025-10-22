import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

# CONFIGURATION
DATA_DIR = "data"
API_FILE = os.path.join(DATA_DIR, "api_processed.csv")
SCRAPED_FILE = os.path.join(DATA_DIR, "scraped_processed.csv")
MODEL_DIR = "models"

TARGET = "viewCount"  # You can switch this to 'engagement_rate' if you prefer

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)

# HELPER FUNCTIONS
def evaluate_model(model, X_train, X_test, y_train, y_test, label):
    """Evaluate model and print metrics."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, train_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)

    print(f"\n{label} Results:")
    print(f"Train MAE: {mae_train:.2f},  Test MAE: {mae_test:.2f}")
    print(f"Train R² : {r2_train:.3f},  Test R² : {r2_test:.3f}")

    return mae_test, r2_test


def plot_feature_importance(model, feature_names, title, filename):
    """Plot feature importance bar chart."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", filename))
        plt.close()
    else:
        print(f"{title}: No feature_importances_ attribute found.")

# MODEL TRAINING
def train_on_dataset(csv_path, label):
    print(f"\nTraining on {label} dataset...")
    df = pd.read_csv(csv_path)

    # Select numeric features
    features = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors="ignore")
    X = features.fillna(0)
    y = df[TARGET].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    mae_rf, r2_rf = evaluate_model(rf, X_train, X_test, y_train, y_test, f"{label} - RandomForest")
    plot_feature_importance(rf, X.columns, f"{label} RandomForest Feature Importance", f"{label.lower()}_rf_importance.png")
    joblib.dump(rf, os.path.join(MODEL_DIR, f"{label.lower()}_rf.pkl"))

    # --- XGBoost ---
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    mae_xgb, r2_xgb = evaluate_model(xgb, X_train, X_test, y_train, y_test, f"{label} - XGBoost")
    plot_feature_importance(xgb, X.columns, f"{label} XGBoost Feature Importance", f"{label.lower()}_xgb_importance.png")
    joblib.dump(xgb, os.path.join(MODEL_DIR, f"{label.lower()}_xgb.pkl"))

    return {
        "label": label,
        "mae_rf": mae_rf,
        "r2_rf": r2_rf,
        "mae_xgb": mae_xgb,
        "r2_xgb": r2_xgb,
    }

# MAIN
def main():
    results = []
    results.append(train_on_dataset(API_FILE, "API"))
    results.append(train_on_dataset(SCRAPED_FILE, "Scraped"))

    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)

    results_df.to_csv(os.path.join(DATA_DIR, "model_performance.csv"), index=False)
    print(f"\nSaved model results to {os.path.join(DATA_DIR, 'model_performance.csv')}")


if __name__ == "__main__":
    main()