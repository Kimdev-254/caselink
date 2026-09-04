"""
Stage 4: Semantic search over the embedded judgments.
"""
import argparse
import os

import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "caselink",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", ""),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="the legal question to search for")
    parser.add_argument("--top", type=int, default=5, help="number of results to return")
    args = parser.parse_args()

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    query_text = "Represent this sentence for searching relevant passages: " + args.question
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                c.chunk_id,
                c.text,
                ca.case_name,
                ca.citation,
                ca.judge,
                ca.decision_date,
                ca.source_url,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN cases ca ON ca.case_id = c.case_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_str, embedding_str, args.top),
        )
        results = cur.fetchall()
    conn.close()

    print(f"\nQuestion: {args.question}\n")
    print(f"Top {len(results)} results:\n" + "=" * 60)

    for i, (chunk_id, text, case_name, citation, judge, date, url, similarity) in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {similarity:.3f}")
        print(f"Case: {case_name}")
        print(f"Citation: {citation} | Judge: {judge} | Date: {date}")
        print(f"Source: {url}")
        print(f"Text: {text[:400]}{'...' if len(text) > 400 else ''}")
        print("-" * 60)


if __name__ == "__main__":
    main()
