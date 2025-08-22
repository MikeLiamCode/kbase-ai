import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_shard_names():
    """
    Discover all shard/collection names by listing subfolders in the docs directory.
    Returns:
        List[str]: List of shard names (subfolder names).
    """
    docs_dir = os.path.join("tests", "docs")
    return [name for name in os.listdir(docs_dir) if os.path.isdir(os.path.join(docs_dir, name))]

def search_shard(shard_name, query_embeddings, top_k):
    """
    Search a single shard/collection for the query.
    Args:
        shard_name (str): Name of the shard/collection.
        query_embeddings: Embedding for the query.
        top_k (int): Number of top results to return.
    Returns:
        dict: ChromaDB query result for the shard.
    """
    shard_collection = chroma_client.get_or_create_collection(name=shard_name)
    return shard_collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "embeddings", "distances"]
    )

def semantic_search(query: str, top_k: int = 5):
    """
    Perform semantic search across all shards/collections in parallel.
    Args:
        query (str): The search query.
        top_k (int): Number of top results to return.
    Returns:
        dict: Merged and sorted top results from all shards.
    """
    query_embeddings = model.encode([query])
    shard_names = get_shard_names()
    with ThreadPoolExecutor() as executor:
        results_list = list(executor.map(lambda shard: search_shard(shard, query_embeddings, top_k), shard_names))
    merged = {"document": [], "metadata": [], "embedding": [], "distance": []}
    for results in results_list:
        merged["document"].extend(results["documents"][0])
        merged["metadata"].extend(results["metadatas"][0])
        merged["embedding"].extend(results["embeddings"][0])
        merged["distance"].extend(results["distances"][0])
    sorted_indices = sorted(range(len(merged["distance"])), key=lambda i: merged["distance"][i])
    top_indices = sorted_indices[:top_k]
    return {
        "document": [merged["document"][i] for i in top_indices],
        "metadata": [merged["metadata"][i] for i in top_indices],
        "embedding": [merged["embedding"][i] for i in top_indices],
        "distance": [merged["distance"][i] for i in top_indices]
    }
