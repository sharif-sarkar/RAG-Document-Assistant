"""
Pytest fixtures for RAG application tests.
Provides common test data and mocked components.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, MagicMock

from langchain_core.documents import Document


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text(
        "This is a sample text file for testing. It contains multiple sentences. " * 10
    )
    return file_path


@pytest.fixture
def sample_md_file(temp_dir):
    """Create a sample markdown file for testing."""
    file_path = temp_dir / "sample.md"
    content = (
        """# Sample Markdown
    
## Section 1
This is a sample markdown file.

## Section 2
It contains multiple sections for testing.
"""
        * 5
    )
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing."""
    return [
        Document(
            page_content="This is the first sample document. It contains information about testing.",
            metadata={"source": "doc1.txt", "page": 0},
        ),
        Document(
            page_content="This is the second sample document. It has different content for variety.",
            metadata={"source": "doc2.txt", "page": 0},
        ),
        Document(
            page_content="This is the third sample document. It provides more test data.",
            metadata={"source": "doc3.txt", "page": 0},
        ),
    ]


@pytest.fixture
def long_document():
    """Create a long document for testing text splitting."""
    content = " ".join(
        [
            f"This is sentence number {i}. It contains some information."
            for i in range(100)
        ]
    )
    return Document(
        page_content=content, metadata={"source": "long_doc.txt", "page": 0}
    )


@pytest.fixture
def mock_ollama_embedding():
    """Mock Ollama embeddings."""
    mock_embedding = MagicMock()
    mock_embedding.embed_documents = Mock(return_value=[[0.1] * 768 for _ in range(3)])
    mock_embedding.embed_query = Mock(return_value=[0.1] * 768)
    return mock_embedding


@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke = Mock(return_value="This is a mocked response from the LLM.")
    return mock_llm


@pytest.fixture
def mock_vector_store():
    """Mock FAISS vector store."""
    mock_store = MagicMock()
    mock_store.similarity_search = Mock(
        return_value=[
            Document(
                page_content="Relevant document 1", metadata={"source": "doc1.txt"}
            ),
            Document(
                page_content="Relevant document 2", metadata={"source": "doc2.txt"}
            ),
        ]
    )
    mock_store.as_retriever = Mock(return_value=MagicMock())
    return mock_store


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    from src.config.settings import Settings

    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    return Settings(
        INPUT_DATA_DIR=input_dir,
        OUTPUT_DATA_DIR=output_dir,
        LLM_MODEL="test_model",
        EMBEDDING_MODEL="test_embedding",
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=50,
    )
