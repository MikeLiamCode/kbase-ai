from src import ingestion, search
import pytest


def test_semantic_search_basic(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc1.write_text("This is a test about cats.")
    doc2.write_text("This is a test about dogs.")
    ingestion.ingest_file(str(doc1))
    ingestion.ingest_file(str(doc2))

    results = search.semantic_search("cats", top_k=2)
    assert len(results) >= 1
    assert any("cats" in r["document"] for r in results)


def test_semantic_search_no_match(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("This is about birds.")
    ingestion.ingest_file(str(doc))

    results = search.semantic_search("unicorns", top_k=2)
    assert all("unicorns" not in r["document"] for r in results)


def test_semantic_search_empty(tmp_path):
    results = search.semantic_search("anything", top_k=2)
    assert isinstance(results, list)
