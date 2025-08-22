# KBaseAI: AI-Powered Knowledge Base Search & Enrichment

## Overview
This project implements a backend system for ingesting documents, generating vector embeddings, and providing semantic search and Q&A capabilities. Designed for rapid prototyping and demonstration.

## Features
- Document ingestion pipeline (raw + vector embeddings, sharded by subfolder)
- Semantic search service (parallel search across all shards)
- REST API for Q&A and completeness check
- Efficient queries across thousands of documents
- Incremental indexing and large file support (skips updates for unchanged documents)

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API server:
   ```bash
   uvicorn src.main:app --reload
   ```
3. See API docs at `/docs` when server is running.

## Document Ingestion
Automatic ingestion occurs at API server startup, scanning all `.txt` files in all subfolders of `tests/docs/` and calling `ingest_file` for each file.

Supported file types: Only `.txt` files are accepted. Attempting to ingest other file types will raise a `ValueError`.

Chunk size: Each document is split into chunks of 1000 characters for embedding and storage. This value can be modified in the code.
**Note:**
- All `.txt` files in all subfolders of `tests/docs/` are ingested recursively at API server startup.
- Each subfolder is treated as a separate ChromaDB collection (shard).

## Semantic Search
The internal search results include the embedding vector for each match, though this is not exposed in the public API response.
To perform semantic search over your ingested documents, use the `semantic_search` function in `src/search.py`:

```python
from src import search
results = search.semantic_search("your query here", top_k=5)
for match in results:
    print(match["document"], match["metadata"], match["distance"])
```
This will return the top matching document chunks, their metadata, and similarity scores.

## API Endpoints

### `/search` (POST)
Results are paginated after semantic search. The API returns only the requested page of results, based on the `page` and `page_size` parameters.
Semantic search over the knowledge base, with pagination and batching support.

**Request:**
```json
{
  "query": "your question here",
  "top_k": 10,
  "page": 1,
  "page_size": 5
}
```

**Response:**

The response is a list of objects, each with scalar fields:
```json
[
  {
    "document": "...",
    "metadata": { "filename": "...", ... },
    "distance": 0.123
  },
  ...
]
```

**Notes:**
- Use `page` and `page_size` to paginate large result sets.
- The backend supports batching for future multi-query extensions.
- Search is performed in parallel across all shards (subfolders in `tests/docs/`).

### `/completeness` (GET)
Coverage score is calculated as `1.0 - distance` of the top search result. Coverage is determined by whether the score exceeds a threshold (default: 0.7).
Check if the knowledge base covers a query.

**Request:**
`/completeness?query=your+question+here`

**Response:**
```json
{
  "covered": true,
  "coverage_score": 0.85
}
```

See interactive docs at `/docs` when the server is running.

## Design Decisions
- In-memory vector DB (Chroma) for speed and simplicity
- Pre-trained embedding models (OpenAI or HuggingFace)
- Minimal chunking for large files
- Focus on core flows, limited edge case handling
- Sharding by subfolder for scalability and parallel search

## Testing
Run tests with:
```bash
pytest tests/
```
Tests cover:
- Document ingestion (including chunking, incremental updates, and error cases)
- Semantic search (including parallel search and sharding)
- API endpoints (including pagination and completeness checks)

## Trade-offs

### HuggingFace Embeddings

- Open-source and free for local use, avoiding API costs and rate limits.
- Slightly lower accuracy and performance compared to proprietary OpenAI models, but sufficient for prototyping and most use cases.
- No need for API keys or internet access, enabling fully local development and deployment.
- Larger resource requirements for running models locally (CPU/GPU).

### In-Memory ChromaDB

- Extremely fast for prototyping and small/medium datasets.
- No persistence: data is lost on restart, not suitable for production or large-scale deployments.
- Simple setup, minimal configuration required.
- Limited scalability; for large datasets or production, a persistent vector DB is recommended.
- Sharding by subfolder enables parallel search and better scalability for thousands of documents.

### Ingestion Flow

- Current ingestion is performed both via a simple script and automatically at app server startup. This dual approach ensures the knowledge base is always initialized and up-to-date when the API server runs, making the demo experience seamless and reducing manual steps for users. Script-based ingestion is fast and flexible for prototyping, while server startup ingestion guarantees the backend is ready for queries immediately after launch.
- Importantly, because the vector database is in-memory, running ingestion in a separate process (via script) would not populate the data for the API server process—each would have its own isolated memory. Therefore, ingestion at server startup is required to ensure the API has access to the ingested data in the same process.
- For production or large-scale systems, a job-based ingestion flow (e.g., background worker, queue, or scheduled job) is recommended for reliability, scalability, and monitoring.
- The choice depends on requirements: scripts are fast and easy, jobs are robust and production-ready.