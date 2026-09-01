"""
Stage 1a: Harvest judgment URLs from Kenya Law's KEELRC listing pages.

We don't fetch full judgments here -- just collect the /akn/ke/judgment/keelrc/...
links from the paginated listing so we know what to fetch next.

Usage:
    python 1_harvest_urls.py --pages 15 --out urls.txt

Each listing page holds ~25-50 judgments, so ~4-6 pages comfortably gets you
150+ URLs (harvest extra -- some pages will be Rulings/Notices you may want
to skip, and a few fetches will inevitably fail).
"""
import argparse
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

BASE_LISTING = "https://new.kenyalaw.org/judgments/KEELRC/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research prototype; contact: youremail@example.com)"
}
JUDGMENT_LINK_RE = re.compile(r"/akn/ke/judgment/keelrc/\d{4}/\d+/[^\"'>]+")


def fetch_listing_page(page_num: int) -> list[str]:
    """Fetch one listing page and return the judgment URLs found on it."""
    params = {"page": page_num} if page_num > 1 else {}
    resp = requests.get(BASE_LISTING, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()

    urls = set()
    for match in JUDGMENT_LINK_RE.findall(resp.text):
        urls.add("https://new.kenyalaw.org" + match if match.startswith("/") else match)

    # Fallback: parse <a href> tags directly in case the regex above misses some
    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if "/akn/ke/judgment/keelrc/" in a["href"]:
            href = a["href"]
            if href.startswith("/"):
                href = "https://new.kenyalaw.org" + href
            urls.add(href)

    return sorted(urls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=6, help="number of listing pages to harvest")
    parser.add_argument("--workers", type=int, default=4, help="concurrent requests (keep this low)")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between request batches")
    parser.add_argument("--out", type=str, default="urls.txt")
    args = parser.parse_args()

    all_urls = set()

    # Bounded concurrency: fetch a small batch of pages at once, pause, repeat.
    # This is dramatically faster than one page at a time, while still capping
    # how many simultaneous requests hit their server.
    page_numbers = list(range(1, args.pages + 1))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_listing_page, p): p for p in page_numbers}
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                urls = future.result()
                all_urls.update(urls)
                print(f"Page {page_num}: found {len(urls)} judgment URLs (running total: {len(all_urls)})")
            except Exception as e:
                print(f"Page {page_num}: FAILED ({e})")
            time.sleep(args.delay / args.workers)  # gentle pacing even within the pool

    with open(args.out, "w") as f:
        f.write("\n".join(sorted(all_urls)))

    print(f"\nDone. {len(all_urls)} unique judgment URLs written to {args.out}")


if __name__ == "__main__":
    main()
