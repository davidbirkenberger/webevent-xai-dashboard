import pandas as pd
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import random
import re

# Session setup with retry logic
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.39 Safari/537.36',
    'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7'
})


def load_index_urls(filepath: str) -> list:
    '''Load index URLs from CSV file'''
    df = pd.read_csv(filepath, sep=";", skiprows=5)
    return df['source'].dropna().tolist()


def crawl_links(url: str) -> list:
    '''Crawl internal links and link texts from a single URL'''
    links = []
    start_domain = urlparse(url).netloc

    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for link in soup.find_all("a", href=True):
            href = link.get("href")
            full_url = urljoin(url, href)
            link_domain = urlparse(full_url).netloc

            if link_domain == start_domain:
                link_text = link.get_text(strip=True) or "N/A"
                links.append({
                    "url": full_url,
                    "link_text": link_text
                })

        time.sleep(random.uniform(1, 3))

    except requests.RequestException as e:
        print(f"Fehler beim Abrufen der URL {url}: {e}")

    return links


def crawl_all_links(index_urls: list) -> pd.DataFrame:
    '''Crawl all links from a list of index URLs'''
    all_links = []

    for index_url in index_urls:
        links = crawl_links(index_url)
        for link in links:
            all_links.append({
                "index_website": index_url,
                "webpage_url": link["url"],
                "link_text": link["link_text"]
            })

    return pd.DataFrame(all_links)


def clean_links(df_links: pd.DataFrame) -> pd.DataFrame:
    '''Remove duplicates, anchors, and unwanted keywords'''
    df = df_links.drop_duplicates()

    # Remove anchor links
    df = df[~df["webpage_url"].str.contains("#", na=False)]

    # Remove known irrelevant keywords
    keywords = ["impressum", "datenschutz", "datenschutzerklaerung", "kontakt", "jobs", "karriere", "sitemap", "policy", "legal", "anfrage"]
    pattern = "|".join(map(re.escape, keywords))
    df = df[~df["webpage_url"].str.contains(pattern, na=False, case=False)]

    return df


def save_links(df_links: pd.DataFrame, filepath: str):
    '''Save the cleaned links to a CSV file'''
    df_links.to_csv(filepath, sep=";", encoding="utf-8-sig", index=False)


def main():
    input_csv = "data/rifel_data.csv"
    output_csv = "data/urls_crawled.csv"

    index_urls = load_index_urls(input_csv)
    raw_links_df = crawl_all_links(index_urls)
    cleaned_links_df = clean_links(raw_links_df)
    save_links(cleaned_links_df, output_csv)
    print(f"{len(cleaned_links_df)} Links gespeichert in {output_csv}")


if __name__ == "__main__":
    main()