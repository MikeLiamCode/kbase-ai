# KBaseAI: AI-Powered Knowledge Base Search & Enrichment

## Overview
This project implements a backend system for ingesting documents, generating vector embeddings, and providing semantic search and Q&A capabilities. Designed for rapid prototyping and demonstration.

## Features
- Document ingestion pipeline (raw + vector embeddings)
- Semantic search service
- REST API for Q&A and completeness check
- Efficient queries across thousands of documents
- Incremental indexing and large file support

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
To ingest documents and generate embeddings, run:
```bash
python src/ingestion.py <file1.txt> <file2.txt> ...
```
This will process each file, store its metadata and embedding in the vector database, and print the results.

## Semantic Search
To perform semantic search over your ingested documents, use the `semantic_search` function in `src/search.py`:

```python
from src import search
results = search.semantic_search("your query here", top_k=5)
for match in results:
    print(match["document"], match["metadata"], match["distance"])
```
This will return the top matching document chunks, their metadata, and similarity scores.

## Design Decisions & Trade-offs
- In-memory vector DB (Chroma) for speed and simplicity
- Pre-trained embedding models (OpenAI)
- Minimal chunking for large files
- Focus on core flows, limited edge case handling

## Testing
Run tests with:
```bash
pytest tests/
```

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

### Ingestion Flow
 - Current ingestion is performed via a simple script for rapid prototyping and demonstration.
 - For production or large-scale systems, a job-based ingestion flow (e.g., background worker, queue, or scheduled job) is recommended for reliability, scalability, and monitoring.
 - The choice depends on requirements: scripts are fast and easy, jobs are robust and production-ready.