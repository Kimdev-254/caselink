"""
Stage 2: Turn raw judgment full_text into paragraph-aware chunks ready
for embedding.

Approach:
  - Split on single newlines (each line == one paragraph in this corpus).
  - Separate the caption/header block (court, parties, case number) from
    the substantive body, using the "RULING"/"JUDGMENT" marker line.
  - Group consecutive body paragraphs into chunks targeting ~300-500 words,
    never splitting a paragraph across two chunks.
  - If a single paragraph alone exceeds the max, split it by sentence.
  - Every chunk carries the case's metadata, so a retrieved chunk always
    knows which case, and (implicitly) which part of it, it came from.

Usage:
    python 3_chunk_cases.py --cases cases/ --out chunks.jsonl
"""
import argparse
import glob
import json
import re

TARGET_MIN_WORDS = 300
TARGET_MAX_WORDS = 500
HARD_MAX_WORDS = 700  # safety valve before we force a sentence-level split

BODY_MARKERS = {"ruling", "judgment", "judgement"}


def word_count(text: str) -> int:
    return len(text.split())


def split_header_and_body(full_text: str):
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if line.lower() in BODY_MARKERS:
            header = "\n".join(lines[: i + 1])
            body_paragraphs = lines[i + 1 :]
            return header, body_paragraphs
    return "", lines


def split_long_paragraph(paragraph: str, max_words: int):
    sentences = re.split(r"(?<=[.;])\s+", paragraph)
    pieces, buf, buf_words = [], [], 0
    for sent in sentences:
        w = word_count(sent)
        if buf and buf_words + w > max_words:
            pieces.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(sent)
        buf_words += w
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def make_chunks(paragraphs):
    chunks = []
    buf, buf_words = [], 0

    for para in paragraphs:
        pw = word_count(para)

        if pw > HARD_MAX_WORDS:
            if buf:
                chunks.append("\n".join(buf))
                buf, buf_words = [], 0
            chunks.extend(split_long_paragraph(para, TARGET_MAX_WORDS))
            continue

        if buf and buf_words + pw > TARGET_MAX_WORDS:
            chunks.append("\n".join(buf))
            buf, buf_words = [], 0

        buf.append(para)
        buf_words += pw

        if buf_words >= TARGET_MIN_WORDS:
            chunks.append("\n".join(buf))
            buf, buf_words = [], 0

    if buf:
        chunks.append("\n".join(buf))

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="cases")
    parser.add_argument("--out", type=str, default="chunks.jsonl")
    args = parser.parse_args()

    case_files = sorted(glob.glob(f"{args.cases}/*.json"))
    total_chunks = 0

    with open(args.out, "w") as out_f:
        for path in case_files:
            with open(path) as f:
                case = json.load(f)

            full_text = case.get("full_text")
            if not full_text:
                continue

            header, body_paragraphs = split_header_and_body(full_text)
            chunk_texts = make_chunks(body_paragraphs)

            for idx, chunk_text in enumerate(chunk_texts):
                record = {
                    "chunk_id": f"{case['case_id']}-chunk{idx:03d}",
                    "case_id": case["case_id"],
                    "chunk_index": idx,
                    "text": chunk_text,
                    "word_count": word_count(chunk_text),
                    "case_name": case.get("case_name"),
                    "case_number": case.get("case_number"),
                    "citation": case.get("citation"),
                    "court": case.get("court"),
                    "court_station": case.get("court_station"),
                    "judge": case.get("judge"),
                    "date": case.get("date"),
                    "source_url": case.get("source_url"),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Processed {len(case_files)} cases -> {total_chunks} chunks written to {args.out}")


if __name__ == "__main__":
    main()
