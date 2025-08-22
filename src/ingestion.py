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
collection = chroma_client.get_or_create_collection(name="knowledge_base")


def get_shard_name_from_path(filepath: str) -> str:
    """
    Extract the shard/collection name from the file path.
    Assumes structure: docs/shard_name/filename.txt
    Args:
        filepath (str): Path to the file.
    Returns:
        str: Shard/collection name.
    """
    parts = os.path.normpath(filepath).split(os.sep)
    if 'docs' in parts:
        docs_index = parts.index('docs')
        if docs_index + 1 < len(parts):
            return parts[docs_index + 1]
    return os.path.basename(os.path.dirname(filepath))

def ingest_file(filepath: str) -> Dict[str, Any]:
    """
    Ingest a single text file, generate its embedding(s), and store in the correct ChromaDB shard/collection.
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
    shard_name = get_shard_name_from_path(filepath)
    shard_collection = chroma_client.get_or_create_collection(name=shard_name)
    existing = shard_collection.get(ids=ids)
    if existing and existing['ids']:
        shard_collection.delete(ids=existing['ids'])
    shard_collection.add(
        documents=chunks,
        metadatas=[metadata]*len(chunks),
        embeddings=embeddings,
        ids=ids
    )
    return {'metadata': metadata, 'embeddings': embeddings, 'ids': ids}

def ingest_files(filepaths: List[str]) -> List[Dict[str, Any]]:
    """
    Ingest multiple text files and store their embeddings in ChromaDB shards/collections.
    Args:
        filepaths (List[str]): List of file paths to ingest.
    Returns:
        List[dict]: List of metadata and embeddings for each ingested file.
    """
    return [ingest_file(fp) for fp in filepaths]

if __name__ == "__main__":
    for fp in sys.argv[1:]:
        print(ingest_file(fp))
