import pytest
from src import ingestion

LARGE_TEXT = "A" * (ingestion.CHUNK_SIZE * 2 + 100)


def test_chunking(tmp_path):
    test_file = tmp_path / "large.txt"
    test_file.write_text(LARGE_TEXT)
    result = ingestion.ingest_file(str(test_file))
    assert len(result['embeddings']) == 3
    assert len(result['ids']) == 3
    for i in range(3):
        assert result['ids'][i] == f"large.txt_chunk{i}"


def test_incremental_update(tmp_path):
    test_file = tmp_path / "update.txt"
    test_file.write_text("First version" * 100)
    result1 = ingestion.ingest_file(str(test_file))
    ids1 = set(result1['ids'])
    test_file.write_text("Second version" * 100)
    result2 = ingestion.ingest_file(str(test_file))
    ids2 = set(result2['ids'])
    assert ids1 == ids2
    assert any((e1 != e2).any() for e1, e2 in zip(result1['embeddings'], result2['embeddings']))


def test_chunking_and_update(tmp_path):
    test_file = tmp_path / "mix.txt"
    test_file.write_text("A" * ingestion.CHUNK_SIZE + "B" * ingestion.CHUNK_SIZE)
    result1 = ingestion.ingest_file(str(test_file))
    assert len(result1['embeddings']) == 2
    test_file.write_text("C" * (ingestion.CHUNK_SIZE // 2))
    result2 = ingestion.ingest_file(str(test_file))
    assert len(result2['embeddings']) == 1
