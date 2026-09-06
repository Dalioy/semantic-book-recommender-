# Semantic Book Recommender

An LLM-powered semantic book recommender that understands **meaning**, not just keywords.

Describe the kind of book you feel like reading, optionally filter by category and emotional tone, and get semantically matched recommendations with book cover art.

> **Reconstruction Project:** This repository is a reconstructed and modularized implementation based on the original [LLM Semantic Book Recommender](https://github.com/t-redactyl/llm-semantic-book-recommender) project and its accompanying freeCodeCamp tutorial. The goal of this reconstruction was to understand the complete workflow and reorganize it into a cleaner, reusable project structure.

## Overview

The project combines semantic search, zero-shot text classification, emotion analysis, local embeddings, and a Gradio interface to build a natural-language book recommendation system.

```text
Kaggle dataset
      │
      ▼
   Clean
      │
      ▼
Classify (Fiction / Nonfiction / Children's)
      │
      ▼
Score emotions
      │
      ▼
Embed descriptions with Ollama
      │
      ▼
Chroma vector store
      │
      ▼
Semantic retrieval + filtering
      │
      ▼
Gradio recommendation app
```

The main idea is to represent each book description as a vector embedding and retrieve books whose descriptions are semantically similar to the user's natural-language query.

For example:

```text
"a story about someone seeking revenge"
```

can retrieve books related to revenge even when the exact word "revenge" is not present in the book description.

## Project Structure

```text
data/
├── raw/                           # Raw Kaggle CSV — not tracked
└── processed/                     # Cleaned/enriched data — generated

src/
├── config.py                      # Central configuration for paths and models
├── pipeline.py                    # End-to-end pipeline orchestration
├── visualization.py               # EDA plotting functions
│
├── data/
│   ├── download.py                # Kaggle dataset download
│   └── clean.py                   # Cleaning and tagged description creation
│
├── features/
│   ├── classification.py          # Zero-shot book categorization
│   └── emotions.py                # Emotion scoring
│
├── vector_store/
│   └── build_index.py             # Chroma vector-store construction
│
├── recommender/
│   └── retrieval.py               # Semantic retrieval and filtering
│
└── app/
    └── dashboard.py               # Gradio application

notebooks/
├── 00_setup_and_check.ipynb       # Environment/model sanity checks
├── 01_EDA.ipynb                   # Dataset exploration and cleaning
├── 02_vector_search.ipynb         # Vector search experiments
├── 03_text_classification.ipynb   # Zero-shot classification
└── 04_sentiment_analysis.ipynb    # Emotion analysis

chroma_db/                          # Local Chroma vector store — generated

results/
├── figures/                        # Generated EDA figures
└── tables/                         # Generated evaluation tables

reports/                            # Project reports / summaries

utils/
└── helper.py                       # Generic CSV helper functions

requirements.txt
.gitignore
LICENSE
```

The notebooks are used for exploration and experimentation, while the reusable implementation lives inside `src/`.

The Gradio application uses the same retrieval logic from:

```text
src/recommender/retrieval.py
```

This avoids having separate recommendation logic in the notebook and the application.

## How It Works

### 1. Download the dataset

The project uses the **7k Books with Metadata** dataset from Kaggle.

The dataset is downloaded into:

```text
data/raw/
```

The download functionality is implemented in:

```text
src/data/download.py
```

### 2. Clean and prepare the data

`src/data/clean.py` performs the main preprocessing steps.

The pipeline:

* Handles missing values.
* Removes books without required metadata such as descriptions, pages, ratings, or publication year.
* Filters out descriptions shorter than 25 words.
* Creates a combined `title_and_subtitle` field.
* Creates a `tagged_description` field containing the book's `isbn13`.

The ISBN tag is useful because the vector-search result can later be matched back to the original book metadata.

### 3. Book classification

The raw Kaggle categories are mapped into simplified categories such as:

```text
Fiction
Nonfiction
Children's Fiction
Children's Nonfiction
```

Books whose original category cannot be mapped directly are classified using zero-shot classification with:

```text
facebook/bart-large-mnli
```

This allows the system to infer a category directly from the book description/category information without training a custom classifier.

### 4. Emotion analysis

The project also analyzes the emotional tone of each book description.

The model:

```text
j-hartmann/emotion-english-distilroberta-base
```

is used to score the following emotions:

```text
anger
disgust
fear
joy
sadness
surprise
neutral
```

Emotion scores are calculated sentence-by-sentence and the maximum score for each emotion is retained for the book.

These scores are later used by the recommendation interface to allow users to filter or prioritize books based on emotional tone.

### 5. Generate embeddings

Each tagged book description is converted into a vector embedding using Ollama:

```text
nomic-embed-text
```

The embeddings are generated locally rather than through a hosted embedding API.

This makes the embedding step reproducible without requiring an external embedding API key.

### 6. Build the vector store

The generated embeddings are stored in a local Chroma vector database:

```text
chroma_db/
```

The vector store allows similarity search over the book descriptions.

Conceptually:

```text
User query
    │
    ▼
Embedding
    │
    ▼
Vector similarity search
    │
    ▼
Most semantically similar books
```

### 7. Recommendation

`src/recommender/retrieval.py` handles the recommendation process.

The retrieval pipeline:

1. Converts the user's query into an embedding.
2. Performs similarity search against the Chroma collection.
3. Retrieves semantically similar books.
4. Applies the selected category filter.
5. Uses the selected emotional tone to re-rank/filter results.
6. Returns the corresponding book metadata and cover information.

The same retrieval implementation is shared between the notebooks and the Gradio application.

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download Ollama from:

https://ollama.com

Start the Ollama server:

```bash
ollama serve
```

Then pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Keep the Ollama server running while building the vector store or using the application.

### 3. Set up the Kaggle dataset

The project can download the dataset automatically through Kaggle.

If using the download functionality, configure your Kaggle API credentials:

```text
~/.kaggle/kaggle.json
```

Alternatively, download the dataset manually and place:

```text
books.csv
```

inside:

```text
data/raw/
```

Dataset:

https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata

## Usage

### Option A — Run the complete pipeline

The complete workflow can be executed through:

```bash
python -m src.pipeline
```

On the first run, enable dataset downloading inside `run_pipeline()` if the raw dataset is not already available.

The pipeline can generate:

```text
data/processed/
results/tables/
chroma_db/
```

and all other artifacts required by the application.

### Option B — Run the notebooks

The notebooks can be executed individually to understand each stage of the system.

Recommended order:

```text
00_setup_and_check.ipynb
        ↓
01_EDA.ipynb
        ↓
02_vector_search.ipynb
        ↓
03_text_classification.ipynb
        ↓
04_sentiment_analysis.ipynb
```

The notebooks are mainly intended for experimentation and understanding the individual components of the system.

## Launch the Application

After the pipeline and vector store have been generated:

```bash
python -m src.app.dashboard
```

This launches the Gradio interface.

The application allows the user to:

* Describe the type of book they want.
* Perform semantic search.
* Filter by book category.
* Filter or prioritize books based on emotional tone.
* Browse recommended books and their cover images.

## Configuration

All major paths and model names are centralized in:

```text
src/config.py
```

This makes it possible to change the configuration without modifying every module individually.

| Setting            | Default                                         | Purpose                                      |
| ------------------ | ----------------------------------------------- | -------------------------------------------- |
| `EMBEDDING_MODEL`  | `nomic-embed-text`                              | Local Ollama embedding model                 |
| `ZERO_SHOT_MODEL`  | `facebook/bart-large-mnli`                      | Zero-shot book classifier                    |
| `EMOTION_MODEL`    | `j-hartmann/emotion-english-distilroberta-base` | Emotion classifier                           |
| `CATEGORY_MAPPING` | Defined in `config.py`                          | Maps raw categories to simplified categories |

## Technologies Used

| Technology   | Purpose                                     |
| ------------ | ------------------------------------------- |
| Python       | Main programming language                   |
| Pandas       | Data processing                             |
| NumPy        | Numerical operations                        |
| Matplotlib   | Visualization                               |
| Seaborn      | Statistical visualization                   |
| Squarify     | Treemap visualization                       |
| Scikit-learn | Evaluation utilities                        |
| PyTorch      | Deep learning backend                       |
| Transformers | Zero-shot classification and emotion models |
| Accelerate   | Transformer model acceleration              |
| TQDM         | Progress tracking                           |
| KaggleHub    | Dataset downloading                         |
| LangChain    | LLM/vector-store integration                |
| ChromaDB     | Local vector database                       |
| Ollama       | Local embedding generation                  |
| Gradio       | Web application                             |

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
pandas
numpy
matplotlib
seaborn
squarify
scikit-learn

torch
transformers
tf-keras
accelerate
tqdm

jupyter
kagglehub

langchain
langchain-community
langchain-text-splitters
langchain-chroma
langchain-ollama

chromadb
gradio
```

## Data and Generated Files

The following directories are intentionally excluded from version control:

```text
data/
chroma_db/
```

These artifacts are generated locally and can be recreated through the pipeline.

In particular:

```text
data/raw/
```

contains the downloaded Kaggle dataset, while:

```text
data/processed/
```

contains generated cleaned/enriched datasets.

The Chroma database:

```text
chroma_db/
```

is also generated locally from the processed book descriptions.

## Cover Images

The project includes:

```text
results/figures/cover-not-found.jpg
```

This is a static fallback image used when a book cover cannot be retrieved.

Unlike the other generated files under `results/`, this fallback asset is tracked in Git.

## GPU Support

The zero-shot classification and emotion models can run on CPU by default.

For systems with a CUDA-compatible GPU, the pipeline can be configured to use:

```python
device=0
```

instead of:

```python
device=-1
```

Using a GPU can significantly reduce the processing time for the classification and emotion-scoring stages.

## Reconstruction Notes

This project was created as a **learning-focused reconstruction** of the original semantic book recommender.

The reconstruction focuses on understanding and reorganizing the original workflow into a more modular architecture:

```text
Original project
      │
      ├── Data exploration
      ├── Vector search
      ├── Text classification
      ├── Sentiment analysis
      └── Gradio application
              │
              ▼
      Reconstructed project
              │
              ├── src/data/
              ├── src/features/
              ├── src/vector_store/
              ├── src/recommender/
              ├── src/app/
              └── src/pipeline.py
```

The original repository and tutorial can be found here:

https://github.com/t-redactyl/llm-semantic-book-recommender

The reconstruction is intended for educational purposes and for practicing:

* Semantic search
* Vector databases
* Text embeddings
* Zero-shot classification
* Emotion analysis
* LangChain
* Ollama
* ChromaDB
* Gradio
* Modular Python project organization

## Original Project

This reconstruction is based on:

**t-redactyl — LLM Semantic Book Recommender**

https://github.com/t-redactyl/llm-semantic-book-recommender

The original repository accompanies the freeCodeCamp tutorial **"Build a Semantic Book Recommender with LLMs – Full Course"** and demonstrates text cleaning, semantic/vector search, zero-shot classification, sentiment/emotion analysis, and a Gradio application.

## License

See `LICENSE` for the license applicable to this reconstruction.
