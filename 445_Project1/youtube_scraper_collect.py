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
SEARCH_QUERY = "trending music"   # Change to "gaming", "education", etc.
MAX_RESULTS = 3000                # Adjust as needed
SCROLL_PAUSE = 2                  # Seconds to wait while scrolling

# SELENIUM SETUP
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# UTILITY FUNCTIONS
def parse_views(views_text):
    """Convert text like '2.3M views' → integer."""
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
    """Convert duration string (e.g. '12:34') → minutes."""
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

# SCRAPE FUNCTION
def scrape_youtube(search_query, max_results=1000):
    driver = setup_driver()

    # Perform search
    search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
    driver.get(search_url)
    time.sleep(3)

    # Scroll to load videos
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    video_count = 0
    scroll_attempts = 0

    while video_count < max_results and scroll_attempts < 30:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time