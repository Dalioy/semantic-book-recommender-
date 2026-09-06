"""
classification.py
==================
Simplifies the raw `categories` column into a small set of buckets
(Fiction / Nonfiction / Children's Fiction / Children's Nonfiction), then
uses zero-shot classification to fill in the books whose original category
didn't map to any of those buckets.

Refactored from notebooks/04_text-classification.ipynb.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from src.config import (
    BOOKS_WITH_CATEGORIES_CSV,
    CATEGORY_MAPPING,
    CLEANED_BOOKS_CSV,
    FICTION_CATEGORIES,
    PREDICTIONS_CSV,
    ZERO_SHOT_MODEL,
)


def map_simple_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw `categories` column to a simplified `simple_categories`."""
    df = df.copy()
    df["simple_categories"] = df["categories"].map(CATEGORY_MAPPING)
    return df


def load_zero_shot_pipeline(model_name: str = ZERO_SHOT_MODEL, device: int = -1):
    """Load the zero-shot classification pipeline. device=0 for GPU, -1 for CPU."""
    return pipeline("zero-shot-classification", model=model_name, device=device)


def predict_label(pipe, sequence: str, candidate_labels=FICTION_CATEGORIES) -> str:
    """Classify a single description and return the highest-scoring label."""
    predictions = pipe(sequence, candidate_labels)
    max_index = np.argmax(predictions["scores"])
    return predictions["labels"][max_index]


def evaluate_classifier(
    df: pd.DataFrame,
    pipe,
    n_samples_per_class: int = 300,
    candidate_labels=FICTION_CATEGORIES,
    save: bool = True,
) -> float:
    """
    Sanity-check the zero-shot classifier against books that already have a
    known Fiction/Nonfiction label, and return accuracy.
    """
    actual_cats, predicted_cats = [], []

    for label in candidate_labels:
        descriptions = (
            df.loc[df["simple_categories"] == label, "description"]
            .reset_index(drop=True)
            .head(n_samples_per_class)
        )
        for sequence in tqdm(descriptions, desc=f"Evaluating on '{label}'"):
            predicted_cats.append(predict_label(pipe, sequence, candidate_labels))
            actual_cats.append(label)

    results_df = pd.DataFrame({"actual": actual_cats, "predicted": predicted_cats})
    accuracy = (results_df["actual"] == results_df["predicted"]).mean()

    if save:
        PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(PREDICTIONS_CSV, index=False)

    print(f"Zero-shot classifier accuracy: {accuracy:.2%}")
    return accuracy


def classify_missing_categories(
    df: pd.DataFrame,
    pipe,
    candidate_labels=FICTION_CATEGORIES,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Fill in `simple_categories` for every book whose category didn't map to a known bucket."""
    missing = df.loc[df["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)

    descriptions = missing["description"].tolist()
    isbns = missing["isbn13"].tolist()
    predicted_cats = []

    for i in tqdm(range(0, len(descriptions), batch_size), desc="Classifying missing categories"):
        batch = descriptions[i:i + batch_size]
        results = pipe(batch, candidate_labels)
        for result in results:
            max_index = np.argmax(result["scores"])
            predicted_cats.append(result["labels"][max_index])

    predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})

    df = pd.merge(df, predicted_df, on="isbn13", how="left")
    df["simple_categories"] = np.where(
        df["simple_categories"].isna(), df["predicted_categories"], df["simple_categories"]
    )
    df = df.drop(columns=["predicted_categories"])
    return df


def build_categorized_books(
    df: pd.DataFrame | None = None,
    device: int = -1,
    evaluate: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """Run the full category pipeline: simple mapping → (optional eval) → fill missing via zero-shot."""
    if df is None:
        df = pd.read_csv(CLEANED_BOOKS_CSV)

    df = map_simple_categories(df)
    pipe = load_zero_shot_pipeline(device=device)

    if evaluate:
        evaluate_classifier(df, pipe)

    df = classify_missing_categories(df, pipe)

    if save:
        BOOKS_WITH_CATEGORIES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(BOOKS_WITH_CATEGORIES_CSV, index=False)
        print("Saved:", df.shape, "->", BOOKS_WITH_CATEGORIES_CSV)

    return df


if __name__ == "__main__":
    build_categorized_books()
