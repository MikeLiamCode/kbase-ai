
import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from src import search, ingestion



app = FastAPI(
    title="KBaseAI Knowledge Base API",
    description="API for semantic search and completeness checking over the KBaseAI knowledge base."
)

TEST_DOCS = [
    "tests/test1.txt",
    "tests/test2.txt",
    "tests/test3.txt",
    "tests/test4.txt",
    "tests/test5.txt"
]

for doc_path in TEST_DOCS:
    if os.path.exists(doc_path):
        try:
            ingestion.ingest_file(doc_path)
        except Exception:
            pass

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    page: int = 1
    page_size: int = 5

class SearchResult(BaseModel):
    document: str
    metadata: dict
    distance: float

@app.post(
    "/search",
    response_model=List[SearchResult],
    summary="Semantic search over the knowledge base",
    description="Returns the most relevant documents for a given query using semantic search.",
    tags=["Search"]
)
def search_endpoint(request: SearchRequest) -> List[SearchResult]:
    """
    Perform semantic search for the given query and return top-k results.
    """
    start = (request.page - 1) * request.page_size
    end = start + request.page_size
    results = search.semantic_search(request.query, top_k=request.top_k)
    paginated = results[start:end]
    return [SearchResult(
        document=r["document"],
        metadata=r["metadata"],
        distance=r["distance"]
    ) for r in paginated]

class CompletenessResponse(BaseModel):
    covered: bool
    coverage_score: float

@app.get(
    "/completeness",
    response_model=CompletenessResponse,
    summary="Check completeness of knowledge base for a query",
    description="Evaluates if the knowledge base sufficiently covers the given query.",
    tags=["Completeness"]
)
def completeness_endpoint(query: str = Query(..., description="Query to check coverage")) -> CompletenessResponse:
    """
    Check if the knowledge base covers the given query and return a coverage score.
    """
    try:
        results = search.semantic_search(query, top_k=1)
        if results:
            score = 1.0 - results[0]["distance"]
            covered = score > 0.7
        else:
            score = 0.0
            covered = False
        return CompletenessResponse(covered=covered, coverage_score=score)
    except Exception as e:
        # Basic error handling for unexpected issues
        return CompletenessResponse(covered=False, coverage_score=0.0)
