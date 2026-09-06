"""
download.py
===========
Downloads the raw "7k Books with Metadata" dataset from Kaggle into
data/raw/. Requires a configured Kaggle API token (~/.kaggle/kaggle.json)
since it uses `kagglehub` under the hood.
"""

import os
import shutil

import kagglehub

from src.config import DATA_RAW_DIR, KAGGLE_DATASET


def download_books_dataset(destination: str = None) -> str:
    """
    Download the Kaggle books dataset and move its files into data/raw/.

    Returns the destination folder path.
    """
    destination = destination or str(DATA_RAW_DIR)
    os.makedirs(destination, exist_ok=True)

    cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Downloaded to Kaggle cache:", cache_path)

    for file_name in os.listdir(cache_path):
        source_path = os.path.join(cache_path, file_name)
        target_path = os.path.join(destination, file_name)

        if os.path.exists(target_path):
            base, ext = os.path.splitext(file_name)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(destination, f"{base}_{counter}{ext}")
                counter += 1

        shutil.move(source_path, target_path)

    print("Files moved to:", destination)
    return destination


if __name__ == "__main__":
    download_books_dataset()
