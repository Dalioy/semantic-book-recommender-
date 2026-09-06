"""
retrieval.py
============
Semantic recommendation logic shared between the exploration notebook and
the Gradio app (src/app/dashboard.py) — implemented once here so both stay
in sync.
"""

import numpy as np
import pandas as pd

TONE_TO_EMOTION = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


def retrieve_semantic_recommendations(
    db,
    books: pd.DataFrame,
    query: str,
    category: str | None = None,
    tone: str | None = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Run a semantic similarity search against the Chroma index, then filter
    by category and re-sort by emotional tone.
    """
    recs = db.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    emotion_col = TONE_TO_EMOTION.get(tone)
    if emotion_col:
        book_recs = book_recs.sort_values(by=emotion_col, ascending=False)

    return book_recs


def format_authors(authors_field: str) -> str:
    """Turn a semicolon-separated author list into readable prose ('A and B' / 'A, B, and C')."""
    authors_split = authors_field.split(";")
    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    if len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    return authors_field


def add_large_thumbnail(books: pd.DataFrame, fallback_path: str) -> pd.DataFrame:
    """Add a `large_thumbnail` column, falling back to a local placeholder image when missing."""
    books = books.copy()
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(), fallback_path, books["large_thumbnail"]
    )
    return books


def build_gallery_items(recommendations: pd.DataFrame, description_words: int = 30) -> list[tuple[str, str]]:
    """Turn a recommendations DataFrame into (image, caption) tuples for a gallery widget."""
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:description_words]) + "..."
        authors_str = format_authors(row["authors"])
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results
