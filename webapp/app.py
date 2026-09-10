"""
CaseLink web app - a simple, clean interface for the RAG pipeline.

Run with:
    python app.py
Then open http://localhost:5001 in your browser.
"""
from fasthtml.common import *
from rag import ask

app, rt = fast_app(
    hdrs=(
        Style("""
            body { max-width: 900px; margin: 40px auto; font-family: -apple-system, sans-serif; color: #1a1a1a; }
            h1 { font-size: 1.4rem; }
            .subtitle { color: #666; margin-bottom: 2rem; }
            form { display: flex; gap: 8px; margin-bottom: 2rem; }
            input[type=text] { flex: 1; padding: 10px; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }
            button { padding: 10px 20px; font-size: 1rem; background: #1a1a1a; color: white; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background: #333; }
            .answer-box { background: #f7f7f5; border-radius: 8px; padding: 20px; margin-bottom: 24px; line-height: 1.6; white-space: pre-wrap; }
            .sources { display: flex; flex-direction: column; gap: 12px; }
            .source-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px 16px; }
            .source-label { font-size: 0.75rem; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.03em; }
            .source-name { font-weight: 600; margin: 4px 0; }
            .source-meta { font-size: 0.85rem; color: #666; margin-bottom: 8px; }
            .source-excerpt { font-size: 0.9rem; color: #333; border-left: 3px solid #ddd; padding-left: 10px; margin-bottom: 8px; }
            .source-link { font-size: 0.85rem; }
            .loading { color: #888; font-style: italic; }
        """),
    )
)


def source_card(i: int, source: dict):
    return Div(
        Div(f"Source {i}", cls="source-label"),
        Div(source["case_name"], cls="source-name"),
        Div(
            f"{source['citation']} | Judge: {source['judge']} | {source['date']} | Match: {source['similarity']:.2f}",
            cls="source-meta",
        ),
        Div(source["text"][:350] + ("..." if len(source["text"]) > 350 else ""), cls="source-excerpt"),
        A("View full judgment on Kenya Law →", href=source["source_url"], target="_blank", cls="source-link"),
        cls="source-card",
    )


@rt("/")
def get():
    return Titled(
        "CaseLink",
        P("Ask a question about Kenyan employment law. Every answer is grounded in real, cited judgments.", cls="subtitle"),
        Form(
            Input(type="text", name="question", placeholder="e.g. Can an employee be dismissed without a hearing?"),
            Button("Ask"),
            hx_post="/search",
            hx_target="#results",
            hx_swap="innerHTML",
            hx_indicator="#spinner",
        ),
        Div("Searching case law and generating an answer... (may take a minute on local hardware)", id="spinner", cls="loading htmx-indicator"),
        Div(id="results"),
    )


@rt("/search")
def post(question: str):
    result = ask(question, top_k=3)

    return Div(
        Div(result["answer"], cls="answer-box"),
        H3("Sources"),
        Div(
            *[source_card(i, s) for i, s in enumerate(result["sources"], 1)],
            cls="sources",
        ),
    )


if __name__ == "__main__":
    serve(port=5001)
