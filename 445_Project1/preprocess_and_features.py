import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import os

# CONFIGURATION
DATA_DIR = "data"
API_INPUT = os.path.join(DATA_DIR, "api_data.csv")
SCRAPED_INPUT = os.path.join(DATA_DIR, "scraped_data.csv")
API_OUTPUT = os.path.join(DATA_DIR, "api_processed.csv")
SCRAPED_OUTPUT = os.path.join(DATA_DIR, "scraped_processed.csv")

# HELPER FUNCTIONS
def safe_int(x):
    try:
        return int(x)
    except:
        return np.nan

def normalize_views(df):
    """Convert all view counts to numeric."""
    df["views"] = df["views"].replace("None", np.nan).astype(float)
    df["views"] = df["views"].fillna(df["views"].median())
    return df

def days_since_upload(date_str):
    """Convert ISO date or human-readable date to days since upload."""
    if pd.isna(date_str):
        return np.nan
    try:
        date_obj = parser.parse(date_str)
        return (datetime.now() - date_obj).days
    except:
        # For strings like "3 weeks ago", "2 months ago"
        date_str = str(date_str).lower()
        num = [int(s) for s in date_str.split() if s.isdigit()]
        if not num:
            return np.nan
        n = num[0]
        if "week" in date_str:
            return n * 7
        elif "month" in date_str:
            return n * 30
        elif "year" in date_str:
            return n * 365
        elif "day" in date_str:
            return n
        else:
            return np.nan

def clean_numeric_columns(df, cols):
    for col in cols:
        df[col] = df[col].apply(safe_int)
        df[col] = df[col].fillna(df[col].median())
    return df

def add_common_features(df, source="api"):
    """Feature engineering for both datasets."""
    df["title_length"] = df["title"].astype(str).apply(len)
    df["channel_length"] = df["channelTitle" if source == "api" else "channel"].astype(str).apply(len)

    if "description" in df.columns:
        df["desc_length"] = df["description"].astype(str).apply(len)
    else:
        df["desc_length"] = 0

    if "publishTime" in df.columns:
        df["days_since_upload"] = df["publishTime"].apply(days_since_upload)
    elif "upload_date" in df.columns:
        df["days_since_upload"] = df["upload_date"].apply(days_since_upload)
    else:
        df["days_since_upload"] = np.nan

    df["duration_min"] = df["duration_min"].fillna(df["duration_min"].median())

    # Engagement rate = (likes + comments) / views
    if source == "api":
        df["engagement_rate"] = (df["likeCount"].astype(float) + df["commentCount"].astype(float)) / df["viewCount"].astype(float)
        df["engagement_rate"] = df["engagement_rate"].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["engagement_rate"] = np.nan  # scraped data doesn't have likes/comments

    return df

# MAIN PREPROCESSING PIPELINE
def preprocess_api():
    print("Loading API data...")
    df = pd.read_csv(API_INPUT)

    # Clean numeric
    numeric_cols = ["viewCount", "likeCount", "commentCount", "duration_min"]
    df = clean_numeric_columns(df, numeric_cols)

    # Feature engineering
    df = add_common_features(df, source="api")

    # Drop unnecessary columns
    drop_cols = ["tags", "categoryId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    print(f"API data processed: {len(df)} samples.")
    df.to_csv(API_OUTPUT, index=False)
    return df


def preprocess_scraped():
    print("Loading scraped data...")
    df = pd.read_csv(SCRAPED_INPUT)

    # Normalize view counts
    df = normalize_views(df)

    # Feature engineering
    df = add_common_features(df, source="scraped")

    print(f"Scraped data processed: {len(df)} samples.")
    df.to_csv(SCRAPED_OUTPUT, index=False)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df_api = preprocess_api()
    df_scraped = preprocess_scraped()

    print("\nSummary:")
    print(f"API processed shape: {df_api.shape}")
    print(f"Scraped processed shape: {df_scraped.shape}")


if __name__ == "__main__":
    main()