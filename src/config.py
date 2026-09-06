"""
config.py
=========
Single source of truth for paths and model names used across the project.
Import from here instead of hardcoding paths in individual modules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_BOOKS_CSV = DATA_RAW_DIR / "books.csv"
CLEANED_BOOKS_CSV = DATA_PROCESSED_DIR / "books_cleaned.csv"
BOOKS_WITH_CATEGORIES_CSV = DATA_PROCESSED_DIR / "books_with_categories.csv"
BOOKS_WITH_EMOTIONS_CSV = DATA_PROCESSED_DIR / "books_with_emotions.csv"
TAGGED_DESCRIPTION_TXT = DATA_PROCESSED_DIR / "tagged_description.txt"

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"  # served locally via Ollama

# ---------------------------------------------------------------------------
# Kaggle source dataset
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "dylanjcastillo/7k-books-with-metadata"

# ---------------------------------------------------------------------------
# Classification / emotion models
# ---------------------------------------------------------------------------
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

CATEGORY_MAPPING = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
}
FICTION_CATEGORIES = ["Fiction", "Nonfiction"]

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PREDICTIONS_CSV = RESULTS_TABLES_DIR / "predictions.csv"
COVER_NOT_FOUND_IMG = RESULTS_FIGURES_DIR / "cover-not-found.jpg"
