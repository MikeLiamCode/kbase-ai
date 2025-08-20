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