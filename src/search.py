
import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
model = SentenceTransformer('all-MiniLM-L6-v2')


def check_completeness(query: str, threshold: float = 0.7) -> dict:
    """
    Check if the knowledge base covers the given query and return coverage info.
    Args:
        query (str): The query to check.
        threshold (float): Coverage threshold.
    Returns:
        dict: {"covered": bool, "coverage_score": float}
    """
    try:
        results = semantic_search(query, top_k=1)
        if results:
            score = 1.0 - results[0]["distance"]
            covered = score > threshold
        else:
            score = 0.0
            covered = False
        return {"covered": covered, "coverage_score": score}
    except Exception:
        return {"covered": False, "coverage_score": 0.0}



def get_shard_names():
    """
    Return all shard/collection names by listing subfolders in the docs directory.
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
    Perform semantic search across all shards/collections in parallel and return top results.
    Args:
        query (str): The search query.
        top_k (int): Number of top results to return.
    Returns:
        List[dict]: List of top matching results, each with document, metadata, embedding, and distance.
    """
    query_embeddings = model.encode([query])
    shard_names = get_shard_names()
    with ThreadPoolExecutor() as executor:
        results_list = list(executor.map(lambda shard: search_shard(shard, query_embeddings, top_k), shard_names))
    merged = []
    for results in results_list:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        embeds = results["embeddings"][0]
        dists = results["distances"][0]
        merged.extend([
            {"document": doc, "metadata": meta, "embedding": embed, "distance": dist}
            for doc, meta, embed, dist in zip(docs, metas, embeds, dists)
        ])
    merged.sort(key=lambda x: x["distance"])
    return merged[:top_k]
