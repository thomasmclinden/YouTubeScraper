import os
import csv
import time
from googleapiclient.discovery import build

# CONFIGURATION
API_KEY = "AIzaSyDeR6QTIIVZ35CvzicgpGTBLnhsX-KIvAA"  # <-- replace with your API key
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "api_data.csv")

TARGET_COUNT = 3000   # total videos, not per category
SAVE_EVERY = 500      # save CSV every N videos
MAX_RESULTS_PER_REQUEST = 50
REGION_CODE = "US"

# Queries to fetch videos from multiple categories
QUERIES = ["Music", "Gaming", "News", "Education", "Comedy", "Sports", "Tech"]

# INITIALIZE YOUTUBE API
youtube = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)

# FUNCTION TO SAVE CSV
def save_csv(videos, output_file):
    if not videos:
        return
    keys = videos[0].keys()
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(videos)
    print(f"Saved {len(videos)} videos to {output_file}")

# FUNCTION TO COLLECT VIDEOS
def collect_videos(target_count=3000):
    collected_videos = []
    seen_video_ids = set()

    for query in QUERIES:
        if len(collected_videos) >= target_count:
            break

        print(f"Collecting videos for query: '{query}'")
        next_page_token = None

        while len(collected_videos) < target_count:
            try:
                request = youtube.search().list(
                    q=query,
                    part="snippet",
                    type="video",
                    maxResults=MAX_RESULTS_PER_REQUEST,
                    pageToken=next_page_token,
                    regionCode=REGION_CODE
                )
                response = request.execute()
            except Exception as e:
                print(f"Error fetching search results: {e}")
                break

            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                if video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)

                snippet = item["snippet"]

                try:
                    stats_request = youtube.videos().list(
                        part="statistics,contentDetails",
                        id=video_id
                    )
                    stats_response = stats_request.execute()
                    stats = stats_response["items"][0]

                    video_data = {
                        "video_id": video_id,
                        "title": snippet.get("title", ""),
                        "channel_title": snippet.get("channelTitle", ""),
                        "publish_date": snippet.get("publishedAt", ""),
                        "category_id": snippet.get("categoryId", ""),
                        "tags": ",".join(snippet.get("tags", [])),
                        "duration": stats["contentDetails"].get("duration", ""),
                        "viewCount": int(stats["statistics"].get("viewCount", 0)),
                        "likeCount": int(stats["statistics"].get("likeCount", 0)),
                        "commentCount": int(stats["statistics"].get("commentCount", 0)),
                    }

                    collected_videos.append(video_data)

                    if len(collected_videos) % SAVE_EVERY == 0:
                        save_csv(collected_videos, OUTPUT_FILE)

                    if len(collected_videos) >= target_count:
                        break

                except Exception as e:
                    print(f"Error fetching stats for video {video_id}: {e}")
                    continue

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                print(f"No more pages for query '{query}', moving to next query.")
                break

            time.sleep(1)  # avoid hitting quota limits

        if len(collected_videos) >= target_count:
            break

    return collected_videos

# MAIN
if __name__ == "__main__":
    print(f"Collecting up to {TARGET_COUNT} videos across multiple queries...")
    all_videos = collect_videos(TARGET_COUNT)
    save_csv(all_videos, OUTPUT_FILE)
    print(f"Collected {len(all_videos)} videos and saved to {OUTPUT_FILE}")
