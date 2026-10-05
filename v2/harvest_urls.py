"""
Generalized judgment URL harvester -- works for any Kenya Law court listing.
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research prototype; contact: youremail@example.com)"
}


def fetch_listing_page(court_code, page_num):
    base_listing = f"https://new.kenyalaw.org/judgments/{court_code}/"
    params = {"page": page_num} if page_num > 1 else {}
    resp = requests.get(base_listing, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()

    link_re = re.compile(rf"/akn/ke/judgment/{court_code.lower()}/\d{{4}}/\d+/[^\"'>]+")
    urls = set()
    for match in link_re.findall(resp.text):
        urls.add("https://new.kenyalaw.org" + match if match.startswith("/") else match)

    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if f"/akn/ke/judgment/{court_code.lower()}/" in a["href"]:
            href = a["href"]
            if href.startswith("/"):
                href = "https://new.kenyalaw.org" + href
            urls.add(href)

    return sorted(urls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--court", type=str, required=True)
    parser.add_argument("--pages", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    all_urls = set()
    page_numbers = list(range(1, args.pages + 1))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_listing_page, args.court, p): p for p in page_numbers}
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                urls = future.result()
                all_urls.update(urls)
                print(f"[{args.court}] Page {page_num}: found {len(urls)} URLs (running total: {len(all_urls)})")
            except Exception as e:
                print(f"[{args.court}] Page {page_num}: FAILED ({e})")
            time.sleep(args.delay / args.workers)

    with open(args.out, "w") as f:
        f.write("\n".join(sorted(all_urls)))

    print(f"\nDone. {len(all_urls)} unique judgment URLs written to {args.out}")


if __name__ == "__main__":
    main()
