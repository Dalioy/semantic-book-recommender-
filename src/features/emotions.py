"""
emotions.py
===========
Scores each book description on 7 emotions (anger, disgust, fear, joy,
sadness, surprise, neutral) using a sentence-level classifier, taking the
max score per emotion across all sentences in the description.

Refactored from notebooks/05_sentiment_analysis.ipynb.
"""

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from src.config import (
    BOOKS_WITH_CATEGORIES_CSV,
    BOOKS_WITH_EMOTIONS_CSV,
    EMOTION_LABELS,
    EMOTION_MODEL,
)


def load_emotion_classifier(model_name: str = EMOTION_MODEL, device: int = -1):
    """Load the emotion classification pipeline. device=0 for GPU, -1 for CPU."""
    return pipeline("text-classification", model=model_name, device=device)


def calculate_max_emotion_scores(predictions, emotion_labels=EMOTION_LABELS) -> dict:
    """Given per-sentence predictions, return the max score per emotion across sentences."""
    if predictions and isinstance(predictions[0], dict):
        predictions = [predictions]

    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        scores_by_label = {item["label"]: item["score"] for item in prediction}
        for label in emotion_labels:
            per_emotion_scores[label].append(scores_by_label.get(label, 0.0))

    return {label: max(scores) for label, scores in per_emotion_scores.items()}


def compute_emotion_scores(df: pd.DataFrame, classifier, emotion_labels=EMOTION_LABELS) -> pd.DataFrame:
    """Compute emotion scores for every book's description. Returns a DataFrame keyed by isbn13."""
    isbns = []
    emotion_scores = {label: [] for label in emotion_labels}

    for i in tqdm(range(len(df)), desc="Scoring emotions"):
        isbns.append(df["isbn13"].iloc[i])
        description = df["description"].iloc[i]

        if not isinstance(description, str) or not description.strip():
            for label in emotion_labels:
                emotion_scores[label].append(0.0)
            continue

        sentences = [s.strip() for s in description.split(".") if s.strip()]
        predictions = classifier(sentences)
        max_scores = calculate_max_emotion_scores(predictions, emotion_labels)
        for label in emotion_labels:
            emotion_scores[label].append(max_scores[label])

    emotions_df = pd.DataFrame(emotion_scores)
    emotions_df["isbn13"] = isbns
    return emotions_df


def build_books_with_emotions(
    df: pd.DataFrame | None = None,
    device: int = -1,
    save: bool = True,
) -> pd.DataFrame:
    """Run the full emotion pipeline and merge scores back onto the books DataFrame."""
    if df is None:
        df = pd.read_csv(BOOKS_WITH_CATEGORIES_CSV)

    classifier = load_emotion_classifier(device=device)
    emotions_df = compute_emotion_scores(df, classifier)

    df = pd.merge(df, emotions_df, on="isbn13")

    if save:
        BOOKS_WITH_EMOTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(BOOKS_WITH_EMOTIONS_CSV, index=False)
        print("Saved:", df.shape, "->", BOOKS_WITH_EMOTIONS_CSV)

    return df


if __name__ == "__main__":
    build_books_with_emotions()
