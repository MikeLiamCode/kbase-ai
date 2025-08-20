import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_search_endpoint_basic():
    response = client.post("/search", json={"query": "test document", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any("test document" in r["document"] for r in data)

def test_search_endpoint_empty():
    response = client.post("/search", json={"query": "nonexistent query", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_completeness_endpoint_covered():
    response = client.get("/completeness", params={"query": "test document"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["covered"], bool)
    assert isinstance(data["coverage_score"], float)

def test_completeness_endpoint_not_covered():
    response = client.get("/completeness", params={"query": "unrelated query"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["covered"], bool)
    assert isinstance(data["coverage_score"], float)
