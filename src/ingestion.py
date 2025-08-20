import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sys

SUPPORTED_EXTENSIONS = ['.txt']

CHUNK_SIZE = 1000

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="documents")

def ingest_file(filepath: str) -> Dict[str, Any]:
    """
    Ingest a single text file, generate its embedding(s), and store in ChromaDB.
    Supports incremental updates and chunking for large files.
    Args:
        filepath (str): Path to the file to ingest.
    Returns:
        dict: Metadata and embedding(s) for the ingested file.
    Raises:
        ValueError: If file extension is not supported.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    metadata = {
        'filename': os.path.basename(filepath),
        'extension': ext,
        'size': os.path.getsize(filepath)
    }
    chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    embeddings = model.encode(chunks)
    ids = [f"{metadata['filename']}_chunk{i}" for i in range(len(chunks))]
    existing = collection.get(ids=ids)
    if existing and existing['ids']:
        collection.delete(ids=existing['ids'])
    collection.add(
        documents=chunks,
        metadatas=[metadata]*len(chunks),
        embeddings=embeddings,
        ids=ids
    )
    return {'metadata': metadata, 'embeddings': embeddings, 'ids': ids}

def ingest_files(filepaths: List[str]) -> List[Dict[str, Any]]:
    """
    Ingest multiple text files and store their embeddings in ChromaDB.
    Args:
        filepaths (List[str]): List of file paths to ingest.
    Returns:
        List[dict]: List of metadata and embeddings for each ingested file.
    """
    return [ingest_file(fp) for fp in filepaths]

if __name__ == "__main__":
    """
    Command-line interface for ingesting files.
    Usage:
        python ingestion.py <file1.txt> <file2.txt> ...
    """
    for fp in sys.argv[1:]:
        print(ingest_file(fp))
