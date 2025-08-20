import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="documents")
model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_search(query: str, top_k: int = 5):
    """
    Perform semantic search over the indexed document chunks.
    Args:
        query (str): The search query.
        top_k (int): Number of top results to return.
    Returns:
        List[dict]: List of top matching chunks with metadata and score.
    """
    query_embeddings = model.encode([query])
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "embeddings", "distances"]
    )
    # Format results
    matches = []
    for i in range(len(results["documents"][0])):
        matches.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "embedding": results["embeddings"][0][i],
            "distance": results["distances"][0][i]
        })
    return matches
