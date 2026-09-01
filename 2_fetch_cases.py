"""
Stage 1b (v2): Fetch each judgment, parse the labeled metadata sidebar,
download the linked DOCX for the actual judgment text, and save one clean
JSON file per case.

Why v2: the judgment page itself only renders a PDF-viewer shell client-side
-- the real text lives in a downloadable .docx/.pdf file linked from the page.
Grabbing that file directly is more reliable than scraping rendered text anyway.

Usage:
    python 2_fetch_cases.py --urls urls.txt --limit 100 --out cases/
"""
import argparse
import io
import json
import re
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research prototype; contact: youremail@example.com)"
}

# Sidebar labels appear as "Label\nValue" pairs in the page's visible text.
LABELS = {
    "citation": "Citation",
    "neutral_citation": "Media Neutral Citation",
    "court": "Court",
    "court_station": "Court station",
    "case_number": "Case number",
    "judge": "Judges",
    "date": "Judgment date",
}


def url_to_case_id(url: str) -> str:
    m = re.search(r"/keelrc/(\d{4})/(\d+)/", url)
    if m:
        return f"keelrc-{m.group(1)}-{m.group(2)}"
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def extract_sidebar_fields(text: str) -> dict:
    """Pull labeled metadata out of the flattened page text.

    The sidebar renders as consecutive lines: label, then value, e.g.
        Court
        Employment and Labour Relations Court
    We find each label and take the next non-empty line as its value.
    """
    lines = [l.strip() for l in text.split("\n")]
    fields = {}
    for key, label in LABELS.items():
        for i, line in enumerate(lines):
            if line == label:
                # value is the next non-empty line, skipping a stray "Copy" button label
                for j in range(i + 1, min(i + 3, len(lines))):
                    val = lines[j].strip()
                    if val and val.lower() != "copy":
                        fields[key] = val
                        break
                break
    return fields


def find_docx_link(html: str, page_url: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        path = urlparse(href).path
        if path.lower().endswith(".docx"):
            return urljoin(page_url, href)
    return None


def extract_docx_text(docx_bytes: bytes) -> str:
    doc = Document(io.BytesIO(docx_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def fetch_one(url: str, out_dir: Path) -> str:
    case_id = url_to_case_id(url)
    out_path = out_dir / f"case_{case_id}.json"
    if out_path.exists():
        return f"SKIP (exists): {case_id}"

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    page_text = soup.get_text("\n", strip=True)

    fields = extract_sidebar_fields(page_text)

    # Case name is the page's title line -- first substantial line of text
    lines = [l for l in page_text.split("\n") if l.strip()]
    case_name = lines[0] if lines else None

    docx_url = find_docx_link(html, url)
    full_text = None
    if docx_url:
        docx_resp = requests.get(docx_url, headers=HEADERS, timeout=30)
        docx_resp.raise_for_status()
        full_text = extract_docx_text(docx_resp.content)

    record = {
        "case_id": case_id,
        "case_name": case_name,
        "case_number": fields.get("case_number"),
        "citation": fields.get("neutral_citation") or fields.get("citation"),
        "court": fields.get("court"),
        "court_station": fields.get("court_station"),
        "judge": fields.get("judge"),
        "date": fields.get("date"),
        "source_url": url,
        "docx_url": docx_url,
        "full_text": full_text,
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    status = "OK" if full_text else "OK (no docx found -- check manually)"
    return f"{status}: {case_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=str, default="urls.txt")
    parser.add_argument("--out", type=str, default="cases")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.urls) as f:
        urls = [line.strip() for line in f if line.strip()][: args.limit]

    print(f"Fetching {len(urls)} judgments with {args.workers} concurrent workers...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, url, out_dir): url for url in urls}
        for i, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            try:
                msg = future.result()
            except Exception as e:
                msg = f"FAILED: {url} ({e})"
            print(f"[{i}/{len(urls)}] {msg}")
            results.append(msg)
            time.sleep(args.delay / args.workers)

    ok = sum(1 for r in results if r.startswith("OK"))
    skipped = sum(1 for r in results if r.startswith("SKIP"))
    failed = sum(1 for r in results if r.startswith("FAILED"))
    print(f"\nDone. {ok} fetched, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
