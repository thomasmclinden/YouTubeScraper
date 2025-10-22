import os
import csv
import time
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from tqdm import tqdm

# CONFIGURATION
OUTPUT_FILE = "data/scraped_data.csv"
SEARCH_QUERY = "trending music"   # Change as needed
MAX_RESULTS = 3000
SCROLL_PAUSE = 2

# SELENIUM SETUP
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# UTILITIES
def parse_views(views_text):
    if not views_text:
        return 0
    views_text = views_text.lower().replace("views", "").strip()
    try:
        if "k" in views_text:
            return int(float(views_text.replace("k", "")) * 1_000)
        elif "m" in views_text:
            return int(float(views_text.replace("m", "")) * 1_000_000)
        elif "b" in views_text:
            return int(float(views_text.replace("b", "")) * 1_000_000_000)
        else:
            return int(views_text.replace(",", "").strip())
    except:
        return 0

def parse_duration(duration_text):
    try:
        parts = duration_text.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
        elif len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
        else:
            return 0
    except:
        return 0

# SCRAPER
def scrape_youtube(search_query, max_results=1000):
    driver = setup_driver()
    search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
    driver.get(search_url)
    time.sleep(3)

    collected = []
    scrolls = 0

    print(f"Collecting up to {max_results} results for '{search_query}'...\n")

    while len(collected) < max_results and scrolls < 30:
        scrolls += 1
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(SCROLL_PAUSE)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        videos = soup.select("ytd-video-renderer")

        for v in videos:
            title_elem = v.select_one("#video-title")
            channel_elem = v.select_one("#text > a")
            meta = v.select_one("#metadata-line")

            title = title_elem.get("title") if title_elem else None
            link = title_elem.get("href") if title_elem else None
            video_id = link.split("v=")[-1] if link and "v=" in link else None
            channel_title = channel_elem.text.strip() if channel_elem else None
            views_text = meta.find_all("span")[0].text if meta and len(meta.find_all("span")) > 0 else "0"
            publish_date = meta.find_all("span")[1].text if meta and len(meta.find_all("span")) > 1 else None

            if not video_id or any(d["video_id"] == video_id for d in collected):
                continue

            collected.append({
                "video_id": video_id,
                "title": title,
                "channel_title": channel_title,
                "publish_date": publish_date,
                "category_id": None,
                "tags": None,
                "duration": None,
                "viewCount": parse_views(views_text),
                "likeCount": None,
                "commentCount": None
            })

        print(f"Collected {len(collected)} videos so far...")
        if len(collected) >= max_results:
            break

    driver.quit()

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(collected).to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Saved {len(collected)} videos to {OUTPUT_FILE}")

# MAIN
if __name__ == "__main__":
    scrape_youtube(SEARCH_QUERY, MAX_RESULTS)