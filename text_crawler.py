import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Setup reusable session
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.6778.39 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
})


def extract_page_information(url: str) -> dict:
    """Extract metadata and visible page content from a given URL."""
    try:
        time.sleep(1)
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title else ""
        meta = soup.find("meta", {"name": "description"})
        description = meta["content"].strip() if meta and meta.get("content") else ""
        h1 = soup.find("h1")
        h1_text = h1.get_text(strip=True) if h1 else ""

        # Full text as fallback if no sentence-filtering applied
        content = soup.get_text(separator=" ", strip=True)

        return {
            "meta-title": title,
            "meta-description": description,
            "h1": h1_text,
            "content": content
        }

    except Exception as e:
        print(f"[Error] {url}: {e}")
        return {
            "meta-title": "",
            "meta-description": "",
            "h1": "",
            "content": ""
        }