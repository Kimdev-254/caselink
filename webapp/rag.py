"""
Shared retrieval + RAG logic, used by both the CLI scripts and the web app.
"""
import os

import psycopg2
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "caselink",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", ""),
}

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def retrieve(question: str, top_k: int = 3):
    model = get_embed_model()
    query_text = "Represent this sentence for searching relevant passages: " + question
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.text, ca.case_name, ca.citation, ca.judge,
                   ca.decision_date, ca.source_url,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN cases ca ON ca.case_id = c.case_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_str, embedding_str, top_k),
        )
        rows = cur.fetchall()
    conn.close()

    return [
        {
            "text": r[0],
            "case_name": r[1],
            "citation": r[2],
            "judge": r[3],
            "date": r[4],
            "source_url": r[5],
            "similarity": r[6],
        }
        for r in rows
    ]


def build_prompt(question: str, chunks: list) -> str:
    context_blocks = [
        f"[Source {i}] {c['citation']}\n{c['text']}" for i, c in enumerate(chunks, 1)
    ]
    context = "\n\n".join(context_blocks)

    return f"""You are a legal research assistant. Answer the question using ONLY the information in the sources below. Do not use any outside knowledge.

For every claim you make, cite the source number in brackets, like [Source 1].

If the sources don't fully answer the question, say so explicitly rather than filling gaps with assumptions.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""


def generate_answer(question: str, chunks: list) -> str:
    prompt = build_prompt(question, chunks)
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def ask(question: str, top_k: int = 3) -> dict:
    chunks = retrieve(question, top_k)
    if not chunks:
        return {"answer": "No relevant cases found in the database.", "sources": []}
    answer = generate_answer(question, chunks)
    return {"answer": answer, "sources": chunks}
