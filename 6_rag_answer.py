"""
Stage 5: Full RAG pipeline.

Retrieve -> constrain the LLM to only use retrieved text -> generate an
answer with citations back to the real cases it came from.
"""
import argparse
import os

import psycopg2
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 3

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "caselink",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", ""),
}


def retrieve(model, question, top_k):
    query_text = "Represent this sentence for searching relevant passages: " + question
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.text, ca.case_name, ca.citation, ca.source_url,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN cases ca ON ca.case_id = c.case_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_str, embedding_str, top_k),
        )
        results = cur.fetchall()
    conn.close()
    return results


def build_prompt(question, chunks):
    context_blocks = []
    for i, (text, case_name, citation, url, sim) in enumerate(chunks, 1):
        context_blocks.append(f"[Source {i}] {citation}\n{text}")
    context = "\n\n".join(context_blocks)

    return f"""You are a legal research assistant. Answer the question using ONLY the information in the sources below. Do not use any outside knowledge.

For every claim you make, cite the source number in brackets, like [Source 1].

If the sources don't fully answer the question, say so explicitly rather than filling gaps with assumptions.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""


def call_ollama(prompt):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top", type=int, default=TOP_K)
    args = parser.parse_args()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Retrieving relevant cases...")
    chunks = retrieve(embed_model, args.question, args.top)

    print(f"Found {len(chunks)} relevant chunks. Generating answer with {OLLAMA_MODEL}...\n")
    prompt = build_prompt(args.question, chunks)
    answer = call_ollama(prompt)

    print("=" * 60)
    print("QUESTION:", args.question)
    print("=" * 60)
    print(answer)
    print("\n" + "=" * 60)
    print("SOURCES USED:")
    for i, (text, case_name, citation, url, sim) in enumerate(chunks, 1):
        print(f"[Source {i}] {case_name}")
        print(f"           {url}")


if __name__ == "__main__":
    main()
