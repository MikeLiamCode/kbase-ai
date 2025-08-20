import pytest
from src import ingestion

def test_ingest_file_success(tmp_path):
    test_file = tmp_path / "test.txt"
    test_content = "Unit test document for ingestion."
    test_file.write_text(test_content)

    result = ingestion.ingest_file(str(test_file))
    assert result['metadata']['filename'] == "test.txt"
    assert result['metadata']['extension'] == ".txt"
    assert result['metadata']['size'] == len(test_content)
    assert result['embedding'] is not None
    assert hasattr(result['embedding'], 'shape')


def test_ingest_file_unsupported(tmp_path):
    test_file = tmp_path / "test.pdf"
    test_file.write_text("PDF content")
    with pytest.raises(ValueError):
        ingestion.ingest_file(str(test_file))


def test_ingest_files_multiple(tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"doc{i}.txt"
        f.write_text(f"Document {i}")
        files.append(str(f))
    results = ingestion.ingest_files(files)
    assert len(results) == 3
    for i, res in enumerate(results):
        assert res['metadata']['filename'] == f"doc{i}.txt"
        assert res['embedding'] is not None
