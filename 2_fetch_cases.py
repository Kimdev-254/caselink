"""
Stage 1b (v2): Fetch each judgment, parse the labeled metadata sidebar,
download the linked DOCX (or PDF fallback for legacy .doc cases), and
save one clean JSON file per case.
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
import pdfplumber

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research prototype; contact: youremail@example.com)"
}

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
    lines = [l.strip() for l in text.split("\n")]
    fields = {}
    for key, label in LABELS.items():
        for i, line in enumerate(lines):
            if line == label:
                for j in range(i + 1, min(i + 3, len(lines))):
                    val = lines[j].strip()
                    if val and val.lower() != "copy":
                        fields[key] = val
                        break
                break
    return fields


def find_document_link(html: str, page_url: str):
    """Prefer .docx (cleanly parseable); fall back to PDF if only a
    legacy .doc link exists, since python-docx can't read .doc files."""
    soup = BeautifulSoup(html, "html.parser")
    docx_link = doc_link = pdf_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        path = urlparse(href).path.lower()
        if path.endswith(".docx"):
            docx_link = urljoin(page_url, href)
        elif path.endswith(".doc"):
            doc_link = urljoin(page_url, href)
        elif path.endswith(".pdf") or "/source.pdf" in path:
            pdf_link = urljoin(page_url, href)
    if docx_link:
        return docx_link, "docx"
    if pdf_link:
        return pdf_link, "pdf"
    return None, None


def extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


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
    lines = [l for l in page_text.split("\n") if l.strip()]
    case_name = lines[0] if lines else None

    doc_url, filetype = find_document_link(html, url)
    full_text = None
    if doc_url:
        doc_resp = requests.get(doc_url, headers=HEADERS, timeout=30)
        doc_resp.raise_for_status()
        if filetype == "docx":
            full_text = extract_docx_text(doc_resp.content)
        elif filetype == "pdf":
            full_text = extract_pdf_text(doc_resp.content)

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
        "doc_url": doc_url,
        "doc_filetype": filetype,
        "full_text": full_text,
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    status = "OK" if full_text else "OK (no document found -- check manually)"
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
