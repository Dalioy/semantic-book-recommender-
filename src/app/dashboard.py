"""
dashboard.py
============
Gradio UI for the semantic book recommender. Uses the shared retrieval
logic in src/recommender/retrieval.py and the shared vector store loader
in src/vector_store/build_index.py, so behavior stays identical to the
notebook version.

Run with:
    python -m src.app.dashboard
"""

import gradio as gr
import pandas as pd

from src.config import BOOKS_WITH_EMOTIONS_CSV, COVER_NOT_FOUND_IMG
from src.recommender.retrieval import (
    add_large_thumbnail,
    build_gallery_items,
    retrieve_semantic_recommendations,
)
from src.vector_store.build_index import load_chroma_index

books = pd.read_csv(BOOKS_WITH_EMOTIONS_CSV)
books = add_large_thumbnail(books, fallback_path=str(COVER_NOT_FOUND_IMG))

db_books = load_chroma_index()


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(db_books, books, query, category, tone)
    return build_gallery_items(recommendations)


categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

CSS = """
* {
    font-family: 'Georgia', serif;
}
body {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh;
}
.gradio-container {
    background: transparent !important;
    max-width: 1200px !important;
    margin: 0 auto;
}
#component-0 {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(10px);
}
h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #e2b96f, #f5e6c8, #e2b96f) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    letter-spacing: 2px !important;
    margin-bottom: 2rem !important;
    text-transform: uppercase;
}
h2 {
    color: #e2b96f !important;
    font-size: 1.4rem !important;
    letter-spacing: 1px !important;
    border-bottom: 1px solid rgba(226,185,111,0.3);
    padding-bottom: 0.5rem;
    margin-top: 1.5rem !important;
}
label {
    color: #c9a96e !important;
    font-size: 0.85rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
input[type="text"], textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(226,185,111,0.3) !important;
    border-radius: 10px !important;
    color: #f5e6c8 !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease;
}
input[type="text"]:focus, textarea:focus {
    border-color: #e2b96f !important;
    box-shadow: 0 0 15px rgba(226,185,111,0.2) !important;
    outline: none !important;
}
select, .gr-dropdown {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(226,185,111,0.3) !important;
    border-radius: 10px !important;
    color: #f5e6c8 !important;
}
button.primary {
    background: linear-gradient(135deg, #c9933a, #e2b96f) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(226,185,111,0.3) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(226,185,111,0.4) !important;
}
.gallery-item {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(226,185,111,0.2) !important;
    transition: all 0.3s ease !important;
}
.gallery-item:hover {
    transform: translateY(-5px) !important;
    border-color: rgba(226,185,111,0.6) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
}
.gr-gallery { background: transparent !important; }
.gr-box {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}
.gr-form { background: transparent !important; gap: 1rem !important; }
.gr-row { gap: 1rem !important; align-items: flex-end !important; }
footer {
    display: none !important;
}
"""


def build_dashboard() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Base(), css=CSS) as dashboard:
        gr.Markdown("# \U0001F4DA Semantic Book Recommender")

        with gr.Row():
            user_query = gr.Textbox(
                label="Describe a book you're looking for",
                placeholder="e.g., A story about forgiveness and redemption...",
                scale=3,
            )
            category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All", scale=1)
            tone_dropdown = gr.Dropdown(choices=tones, label="Emotional Tone", value="All", scale=1)
            submit_button = gr.Button("Find Books", variant="primary", scale=1)

        gr.Markdown("## Recommendations")
        output = gr.Gallery(label="Recommended Books", columns=8, rows=2, object_fit="cover", height=500)

        submit_button.click(
            fn=recommend_books,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=output,
        )

    return dashboard


if __name__ == "__main__":
    build_dashboard().launch()
