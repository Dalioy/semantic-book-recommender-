"""
clean.py
========
Cleans the raw Kaggle books dataset and prepares it for the rest of the
pipeline: flags missing values, drops incomplete rows, filters
too-short descriptions, builds the title+subtitle field and the
isbn13-tagged description used later for embedding.

Refactored from notebooks/01_EDA.ipynb.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.config import CLEANED_BOOKS_CSV, RAW_BOOKS_CSV, TAGGED_DESCRIPTION_TXT

MIN_DESCRIPTION_WORDS = 25


def load_raw_books(path=RAW_BOOKS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def flag_missing_and_age(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Add `missing_description` flag and `age_of_book` column."""
    df = df.copy()
    current_year = current_year or datetime.now().year
    df["missing_description"] = np.where(df["description"].isna(), 1, 0)
    df["age_of_book"] = current_year - df["published_year"]
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing description, num_pages, average_rating or published_year.

    These are a small minority and hard to impute reliably (would require
    web scraping), so dropping is the simplest safe option.
    """
    mask_complete = (
        ~df["description"].isna()
        & ~df["num_pages"].isna()
        & ~df["average_rating"].isna()
        & ~df["published_year"].isna()
    )
    return df[mask_complete].copy()


def filter_by_description_length(df: pd.DataFrame, min_words: int = MIN_DESCRIPTION_WORDS) -> pd.DataFrame:
    """Drop books whose description is too short to be useful for embedding/classification."""
    df = df.copy()
    df["words_in_description"] = df["description"].str.split().str.len()
    return df[df["words_in_description"] >= min_words].copy()


def add_title_and_subtitle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_and_subtitle"] = np.where(
        df["subtitle"].isna(),
        df["title"],
        df[["title", "subtitle"]].astype(str).agg(": ".join, axis=1),
    )
    return df


def add_tagged_description(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix each description with its isbn13 so it can be matched back after retrieval."""
    df = df.copy()
    df["tagged_description"] = df[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
    return df


def clean_books(df: pd.DataFrame | None = None, current_year: int | None = None) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the cleaned DataFrame."""
    if df is None:
        df = load_raw_books()

    df = flag_missing_and_age(df, current_year=current_year)
    df = drop_incomplete_rows(df)
    df = filter_by_description_length(df)
    df = add_title_and_subtitle(df)
    df = add_tagged_description(df)

    df = df.drop(columns=["subtitle", "missing_description", "age_of_book", "words_in_description"])
    return df


def save_cleaned_books(df: pd.DataFrame, path=CLEANED_BOOKS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", df.shape, "->", path)


def save_tagged_description_txt(df: pd.DataFrame, path=TAGGED_DESCRIPTION_TXT) -> None:
    """Save `tagged_description` as a newline-delimited .txt for LangChain's TextLoader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df["tagged_description"].to_csv(path, sep="\n", index=False, header=False, encoding="utf-8")
    print("Saved:", path)


if __name__ == "__main__":
    raw = load_raw_books()
    cleaned = clean_books(raw)
    save_cleaned_books(cleaned)
    save_tagged_description_txt(cleaned)
