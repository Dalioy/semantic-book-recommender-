"""
pipeline.py
===========
End-to-end pipeline: download raw data -> clean -> classify categories ->
score emotions -> build the vector index. Run this to go from nothing to
a fully working `src/app/dashboard.py`.

Usage:
    python -m src.pipeline
"""

from src.data.clean import clean_books, load_raw_books, save_cleaned_books, save_tagged_description_txt
from src.data.download import download_books_dataset
from src.features.classification import build_categorized_books
from src.features.emotions import build_books_with_emotions
from src.vector_store.build_index import build_chroma_index, load_tagged_documents


def run_pipeline(download: bool = False, device: int = -1, evaluate_classifier: bool = True) -> None:
    """
    Run the full pipeline. Set `download=True` on first run if data/raw/
    is empty (requires a configured Kaggle API token). Set `device=0` to
    use a GPU for the classification/emotion models.
    """
    if download:
        print("\n### 1. Downloading raw dataset ###")
        download_books_dataset()

    print("\n### 2. Cleaning data ###")
    raw = load_raw_books()
    cleaned = clean_books(raw)
    save_cleaned_books(cleaned)
    save_tagged_description_txt(cleaned)

    print("\n### 3. Classifying categories (zero-shot) ###")
    with_categories = build_categorized_books(cleaned, device=device, evaluate=evaluate_classifier)

    print("\n### 4. Scoring emotions ###")
    build_books_with_emotions(with_categories, device=device)

    print("\n### 5. Building the vector index (requires `ollama serve` running) ###")
    documents = load_tagged_documents()
    build_chroma_index(documents)

    print("\nDone. Run the app with: python -m src.app.dashboard")


if __name__ == "__main__":
    run_pipeline()
