"""
Stage 3: Embed chunks.jsonl with OpenAI and load everything into Postgres.
"""
import argparse
import json
import os
import time

import psycopg2
from psycopg2.extras import execute_values
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "caselink",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", ""),
}


def load_chunks(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def upsert_cases(conn, records):
    seen = {}
    for r in records:
        if r["case_id"] not in seen:
            seen[r["case_id"]] = (
                r["case_id"],
                r.get("case_name"),
                r.get("case_number"),
                r.get("citation"),
                r.get("court"),
                r.get("court_station"),
                r.get("judge"),
                r.get("date"),
                r.get("source_url"),
            )

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO cases
                (case_id, case_name, case_number, citation, court,
                 court_station, judge, decision_date, source_url)
            VALUES %s
            ON CONFLICT (case_id) DO NOTHING
            """,
            list(seen.values()),
        )
    conn.commit()
    print(f"Upserted {len(seen)} unique cases.")


def embed_batch(client, texts):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="chunks.jsonl")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found -- check your .env file.")

    client = OpenAI(api_key=api_key)
    conn = psycopg2.connect(**DB_CONFIG)

    records = load_chunks(args.chunks)
    print(f"Loaded {len(records)} chunks from {args.chunks}")

    upsert_cases(conn, records)

    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM chunks")
        already_done = {row[0] for row in cur.fetchall()}
    remaining = [r for r in records if r["chunk_id"] not in already_done]
    print(f"{len(remaining)} chunks left to embed ({len(already_done)} already done).")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        texts = [r["text"] for r in batch]

        try:
            embeddings = embed_batch(client, texts)
        except Exception as e:
            print(f"Batch {i}-{i+len(batch)} FAILED: {e}. Retrying once after a pause...")
            time.sleep(5)
            embeddings = embed_batch(client, texts)

        rows = [
            (
                r["chunk_id"],
                r["case_id"],
                r["chunk_index"],
                r["text"],
                r["word_count"],
                "[" + ",".join(str(x) for x in emb) + "]",
            )
            for r, emb in zip(batch, embeddings)
        ]

        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO chunks
                    (chunk_id, case_id, chunk_index, text, word_count, embedding)
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                rows,
            )
        conn.commit()
        print(f"Embedded and inserted chunks {i + 1}-{i + len(batch)} of {len(remaining)}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
