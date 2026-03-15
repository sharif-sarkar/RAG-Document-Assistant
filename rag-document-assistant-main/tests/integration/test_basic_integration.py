"""
Basic integration test for the RAG application.
Tests end-to-end document ingestion and query workflow.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.settings import get_settings
from src.ingestion.loader import DocumentIngestor
from src.processing.splitter import DocumentSplitter


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for end-to-end workflows."""

    def test_document_ingestion_workflow(self, temp_dir, sample_text_file):
        """Test complete document ingestion workflow."""
        # Setup
        settings = get_settings(
            INPUT_DATA_DIR=temp_dir,
            OUTPUT_DATA_DIR=temp_dir / "output"
        )

        # Load documents
        ingestor = DocumentIngestor(settings)
        documents = ingestor.load_documents(sample_text_file)

        # Verify documents loaded
        assert len(documents) > 0

        # Split documents
        splitter = DocumentSplitter(settings)
        chunks = splitter.split_documents(documents)

        # Verify chunks created
        assert len(chunks) >= len(documents)
        for chunk in chunks:
            assert hasattr(chunk, 'page_content')
            assert hasattr(chunk, 'metadata')

    def test_directory_loading_and_splitting(self, temp_dir):
        """Test loading from directory and splitting."""
        # Create test files
        (temp_dir / "test1.txt").write_text("Test document 1. " * 50)
        (temp_dir / "test2.md").write_text("# Test Document 2\n\n" + "Content. " * 50)

        settings = get_settings(
            INPUT_DATA_DIR=temp_dir,
            OUTPUT_DATA_DIR=temp_dir / "output",
            CHUNK_SIZE=200,
            CHUNK_OVERLAP=50
        )

        # Load all documents from directory
        ingestor = DocumentIngestor(settings)
        documents = ingestor.load_documents(temp_dir)

        assert len(documents) >= 2

        # Split documents
        splitter = DocumentSplitter(settings)
        chunks = splitter.split_documents(documents)

        # Should have multiple chunks due to small chunk size
        assert len(chunks) > len(documents)

    def test_settings_propagation_through_pipeline(self, temp_dir):
        """Test that settings correctly propagate through the pipeline."""
        custom_chunk_size = 300
        custom_overlap = 75

        settings = get_settings(
            INPUT_DATA_DIR=temp_dir,
            OUTPUT_DATA_DIR=temp_dir / "output",
            CHUNK_SIZE=custom_chunk_size,
            CHUNK_OVERLAP=custom_overlap,
            LLM_MODEL="custom_llm",
            EMBEDDING_MODEL="custom_embedding"
        )

        # Verify settings
        assert settings.CHUNK_SIZE == custom_chunk_size
        assert settings.CHUNK_OVERLAP == custom_overlap
        assert settings.LLM_MODEL == "custom_llm"
        assert settings.EMBEDDING_MODEL == "custom_embedding"

        # Verify splitter uses correct settings
        splitter = DocumentSplitter(settings)
        assert splitter.settings.CHUNK_SIZE == custom_chunk_size
        assert splitter.settings.CHUNK_OVERLAP == custom_overlap
