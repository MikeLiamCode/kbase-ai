import os
import glob
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from src import search, ingestion



app = FastAPI(
    title="KBaseAI Knowledge Base API",
    description="API for semantic search and completeness checking over the KBaseAI knowledge base."
)

TEST_DOCS_DIR = os.path.join("tests", "docs")
TEST_DOCS = glob.glob(os.path.join(TEST_DOCS_DIR, "**", "*.txt"), recursive=True)

for doc_path in TEST_DOCS:
    if os.path.exists(doc_path):
        try:
            ingestion.ingest_file(doc_path)
        except Exception:
            pass

class SearchRequest(BaseModel):
    """
    Request model for semantic search endpoint.
    Args:
        query (str): The search query.
        top_k (int): Number of top results to return.
        page (int): Page number for pagination.
        page_size (int): Number of results per page.
    """
    query: str
    top_k: int = 5
    page: int = 1
    page_size: int = 5

class SearchResult(BaseModel):
    """
    Response model for a single semantic search result.
    Args:
        document (str): The document text.
        metadata (dict): Metadata for the document.
        distance (float): Similarity distance.
    """
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
    Args:
        request (SearchRequest): The search request.
    Returns:
        List[SearchResult]: List of search results for the query.
    """
    results = search.semantic_search(request.query, top_k=request.top_k)
    start = (request.page - 1) * request.page_size
    end = start + request.page_size
    paginated = results[start:end]
    return [
        SearchResult(
            document=item["document"],
            metadata=item["metadata"],
            embedding=item["embedding"],
            distance=item["distance"]
        ) for item in paginated
    ]

class CompletenessResponse(BaseModel):
    """
    Response model for completeness check endpoint.
    Args:
        covered (bool): Whether the query is covered.
        coverage_score (float): Coverage score for the query.
    """
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
    Args:
        query (str): The query to check coverage for.
    Returns:
        CompletenessResponse: Coverage information for the query.
    """
    result = search.check_completeness(query)
    return CompletenessResponse(**result)
