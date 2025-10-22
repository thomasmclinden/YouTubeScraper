import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

# File paths
DATA_DIR = "data"
API_INPUT = os.path.join(DATA_DIR, "api_data.csv")
SCRAPED_INPUT = os.path.join(DATA_DIR, "scraped_data.csv")
API_OUTPUT = os.path.join(DATA_DIR, "api_processed.csv")
SCRAPED_OUTPUT = os.path.join(DATA_DIR, "scraped_processed.csv")


# Helper functions
def safe_fill_numeric(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    print(f"Filled {len(num_cols)} numeric columns with 0 where missing.")
    return df


def parse_duration(duration):
# Convert ISO 8601 duration like PT5M33S to minutes.
    try:
        duration = duration.replace("PT", "")
        hours, minutes, seconds = 0, 0, 0
        if "H" in duration:
            hours, duration = duration.split("H")
        if "M" in duration:
            minutes, duration = duration.split("M")
        if "S" in duration:
            seconds = duration.replace("S", "")
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0
        return hours * 60 + minutes + seconds / 60
    except Exception:
        return np.nan


def add_common_features(df, source="api"):
    """Add derived features shared across API and scraped datasets."""
    df["title_length"] = df["title"].astype(str).apply(len)
    df["channel_length"] = df["channel_title" if source == "api" else "channel"].astype(str).apply(len)

    if "description" in df.columns:
        df["desc_length"] = df["description"].astype(str).apply(len)
    else:
        df["desc_length"] = 0

    # Compute days since upload
    def compute_days(date_str):
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - date).days
        except Exception:
            return np.nan

    df["days_since_upload"] = df["publish_date"].apply(compute_days)

    # Convert duration and compute engagement rate
    if "duration" in df.columns:
        df["duration_min"] = df["duration"].astype(str).apply(parse_duration)
    else:
        df["duration_min"] = np.nan

    for col in ["viewCount", "likeCount", "commentCount"]:
        if col not in df.columns:
            df[col] = 0

    df["engagement_rate"] = (
        (df["likeCount"] + df["commentCount"]) / df["viewCount"].replace(0, np.nan)
    ).fillna(0)

    return df


# Preprocessing functions
def preprocess_api():
    df = pd.read_csv(API_INPUT)
    print(f"Loaded API data with {len(df)} rows and {len(df.columns)} columns.")
    df = safe_fill_numeric(df)
    df = add_common_features(df, source="api")
    df.to_csv(API_OUTPUT, index=False)
    print(f"Saved processed API data to {API_OUTPUT}")
    return df


def preprocess_scraped():
    df = pd.read_csv(SCRAPED_INPUT)
    print(f"Loaded scraped data with {len(df)} rows and {len(df.columns)} columns.")
    df = safe_fill_numeric(df)
    df = add_common_features(df, source="scraped")
    df.to_csv(SCRAPED_OUTPUT, index=False)
    print(f"Saved processed scraped data to {SCRAPED_OUTPUT}")
    return df


# Main execution
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Process API Data
    if os.path.exists(API_INPUT):
        print("Loading API data...")
        df_api = preprocess_api()
        print(f"API data processed: {len(df_api)} samples.")
    else:
        print(f"Error: {API_INPUT} not found.")
        return

    # Try to process scraped data only if present
    if os.path.exists(SCRAPED_INPUT):
        print("Loading scraped data...")
        df_scraped = preprocess_scraped()
        print(f"Scraped data processed: {len(df_scraped)} samples.")
    else:
        print("No scraped data found. Skipping scraped preprocessing.")
        df_scraped = None

    # Summary
    print("\nSummary:")
    print(f"API processed shape: {df_api.shape}")
    if df_scraped is not None:
        print(f"Scraped processed shape: {df_scraped.shape}")
    else:
        print("Scraped data was skipped (file not found).")


if __name__ == "__main__":
    main()
