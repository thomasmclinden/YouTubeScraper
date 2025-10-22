import os
import pandas as pd
import numpy as np
import pickle
from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score

# CONFIG
API_PROCESSED_FILE = "data/api_processed.csv"
RF_MODEL_FILE = "models/api_rf.pkl"
XGB_MODEL_FILE = "models/api_xgb.pkl"
OUTPUT_PRED_FILE = "data/api_predictions.csv"

# Helpers to load model and extract expected feature names
def load_model_auto(path):
    try:
        return load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def model_feature_names(model):
    # sklearn objects typically have feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # XGBoost sklearn wrapper (XGBRegressor) also has feature_names_in_ in newer versions
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if booster is not None:
                fn = booster.feature_names
                if fn:
                    return list(fn)
        except Exception:
            pass
    # Some models were saved with a custom attribute; attempt common names
    for attr in ("feature_names", "feature_names_in", "input_features"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (list, tuple, np.ndarray)):
                return list(val)
    return None

def align_features(X, expected):
    expected = list(expected)
    # Create empty frame with expected columns
    X_aligned = pd.DataFrame(0, index=X.index, columns=expected)
    # Find intersection
    common = [c for c in expected if c in X.columns]
    if common:
        X_aligned.loc[:, common] = X.loc[:, common].values
    return X_aligned

# Main
if __name__ == "__main__":
    # load data
    if not os.path.exists(API_PROCESSED_FILE):
        raise FileNotFoundError(f"{API_PROCESSED_FILE} not found.")
    df = pd.read_csv(API_PROCESSED_FILE)
    print(f"Loaded processed data: {df.shape}")

    # define target and candidate features (drop obvious non-feature columns)
    target_col = "viewCount"
    non_feature_cols = ["video_id", "title", "channel_title", "publish_date", target_col]
    feature_candidates = [c for c in df.columns if c not in non_feature_cols]

    X_full = df[feature_candidates].copy()
    y = df[target_col].copy()

    # load models
    print("Loading models...")
    rf_model = load_model_auto(RF_MODEL_FILE)
    xgb_model = load_model_auto(XGB_MODEL_FILE)
    print("Models loaded.")

    # determine expected feature sets
    rf_features = model_feature_names(rf_model)
    xgb_features = model_feature_names(xgb_model)

    # fallback: if model has no recorded feature names, assume it used feature_candidates
    if rf_features is None:
        print("Warning: Random Forest model has no recorded feature names. Using processed dataframe columns as features.")
        rf_features = feature_candidates
    if xgb_features is None:
        print("Warning: XGBoost model has no recorded feature names. Using processed dataframe columns as features.")
        xgb_features = feature_candidates

    # Align features for each model
    print(f"Random Forest expects {len(rf_features)} features; dataset has {len(X_full.columns)} candidate features.")
    print(f"XGBoost expects {len(xgb_features)} features; dataset has {len(X_full.columns)} candidate features.")

    X_rf = align_features(X_full, rf_features)
    X_xgb = align_features(X_full, xgb_features)

    # Predict
    print("Making predictions...")
    rf_preds = rf_model.predict(X_rf)
    xgb_preds = xgb_model.predict(X_xgb)

    # Evaluate
    def metrics(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n{name} - MAE: {mae:,.2f} | R²: {r2:.4f}")

    metrics("Random Forest", y, rf_preds)
    metrics("XGBoost", y, xgb_preds)

    # Save predictions
    out = df.copy()
    out["RF_Pred"] = rf_preds
    out["XGB_Pred"] = xgb_preds
    out.to_csv(OUTPUT_PRED_FILE, index=False)
    print(f"\nSaved predictions to {OUTPUT_PRED_FILE}")