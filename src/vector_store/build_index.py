"""
build_index.py
==============
Builds (or loads) the Chroma vector store over book descriptions, embedded
locally via Ollama's `nomic-embed-text` model.

Refactored from notebooks/02_vector-search.ipynb.
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TAGGED_DESCRIPTION_TXT


def load_tagged_documents(path=TAGGED_DESCRIPTION_TXT):
    """Load tagged_description.txt and split it one line (= one book) per document."""
    raw_documents = TextLoader(str(path), encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
    return text_splitter.split_documents(raw_documents)


def build_chroma_index(
    documents,
    persist_dir=CHROMA_PERSIST_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 50,
) -> Chroma:
    """
    Embed every document in batches and persist them to a local Chroma
    collection. Requires `ollama serve` running with the embedding model
    pulled (`ollama pull nomic-embed-text`).
    """
    embedding = OllamaEmbeddings(model=embedding_model)
    persist_dir = str(persist_dir)

    all_texts = [doc.page_content for doc in documents]
    all_metadatas = [doc.metadata for doc in documents]

    db = None
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding books"):
        batch_texts = all_texts[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]

        if db is None:
            db = Chroma.from_texts(
                batch_texts,
                embedding=embedding,
                metadatas=batch_metadatas,
                persist_directory=persist_dir,
            )
        else:
            db.add_texts(batch_texts, metadatas=batch_metadatas)

    print(f"Indexed {db._collection.count()} documents into {persist_dir}")
    return db


def load_chroma_index(persist_dir=CHROMA_PERSIST_DIR, embedding_model: str = EMBEDDING_MODEL) -> Chroma:
    """Load an already-built Chroma collection (does not re-embed anything)."""
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=OllamaEmbeddings(model=embedding_model),
    )


if __name__ == "__main__":
    docs = load_tagged_documents()
    build_chroma_index(docs)
