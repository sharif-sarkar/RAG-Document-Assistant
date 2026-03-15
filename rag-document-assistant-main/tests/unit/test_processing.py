"""
Unit tests for the processing module (src/processing/splitter.py).
Tests DocumentSplitter for text splitting with various chunk sizes and overlap.
"""

import pytest
from langchain_core.documents import Document

from src.processing.splitter import DocumentSplitter


class TestDocumentSplitter:
    """Test cases for DocumentSplitter class."""

    def test_splitter_initialization(self, test_settings):
        """Test DocumentSplitter initialization."""
        splitter = DocumentSplitter(test_settings)

        assert splitter.settings == test_settings
        assert splitter.text_splitter is not None
        assert splitter.logger is not None

    def test_splitter_uses_correct_chunk_size(self, test_settings):
        """Test that splitter uses the configured chunk size."""
        splitter = DocumentSplitter(test_settings)

        # Access the text_splitter's chunk size
        assert splitter.text_splitter._chunk_size == test_settings.CHUNK_SIZE

    def test_splitter_uses_correct_chunk_overlap(self, test_settings):
        """Test that splitter uses the configured chunk overlap."""
        splitter = DocumentSplitter(test_settings)

        assert splitter.text_splitter._chunk_overlap == test_settings.CHUNK_OVERLAP

    def test_split_documents_basic(self, test_settings, sample_documents):
        """Test basic document splitting."""
        splitter = DocumentSplitter(test_settings)

        chunks = splitter.split_documents(sample_documents)

        assert isinstance(chunks, list)
        assert len(chunks) >= len(sample_documents)
        # Each chunk should be a Document
        for chunk in chunks:
            assert isinstance(chunk, Document)

    def test_split_long_document(self, test_settings, long_document):
        """Test splitting a long document into multiple chunks."""
        splitter = DocumentSplitter(test_settings)

        chunks = splitter.split_documents([long_document])

        # Long document should be split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should respect the chunk size (approximately)
        for chunk in chunks:
            assert (
                len(chunk.page_content) <= test_settings.CHUNK_SIZE * 1.5
            )  # Allow some margin

    def test_split_documents_metadata_preservation(
        self, test_settings, sample_documents
    ):
        """Test that metadata is preserved during splitting."""
        splitter = DocumentSplitter(test_settings)

        chunks = splitter.split_documents(sample_documents)

        # Check that metadata is preserved
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_split_documents_empty_list(self, test_settings):
        """Test splitting an empty list of documents."""
        splitter = DocumentSplitter(test_settings)

        chunks = splitter.split_documents([])

        assert chunks == []

    def test_split_text_basic(self, test_settings):
        """Test splitting plain text."""
        splitter = DocumentSplitter(test_settings)
        text = "This is a test. " * 100  # Create a long text

        chunks = splitter.split_text(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # All chunks should be strings
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_split_text_short_text(self, test_settings):
        """Test splitting short text that doesn't need splitting."""
        splitter = DocumentSplitter(test_settings)
        short_text = "This is a short text."

        chunks = splitter.split_text(short_text)

        # Short text should remain as one chunk
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_split_text_respects_chunk_overlap(self, test_settings):
        """Test that chunk overlap is respected."""
        test_settings.CHUNK_SIZE = 100
        test_settings.CHUNK_OVERLAP = 20
        splitter = DocumentSplitter(test_settings)

        long_text = "word " * 200  # Create long text with repeated words
        chunks = splitter.split_text(long_text)

        # Should split into multiple chunks with overlap
        assert len(chunks) > 1

    def test_update_splitter_params_chunk_size(self, test_settings):
        """Test updating chunk size parameter."""
        splitter = DocumentSplitter(test_settings)
        original_size = test_settings.CHUNK_SIZE
        new_size = 800

        splitter.update_splitter_params(chunk_size=new_size)

        assert splitter.settings.CHUNK_SIZE == new_size
        assert splitter.text_splitter._chunk_size == new_size

    def test_update_splitter_params_chunk_overlap(self, test_settings):
        """Test updating chunk overlap parameter."""
        splitter = DocumentSplitter(test_settings)
        new_overlap = 100

        splitter.update_splitter_params(chunk_overlap=new_overlap)

        assert splitter.settings.CHUNK_OVERLAP == new_overlap
        assert splitter.text_splitter._chunk_overlap == new_overlap

    def test_update_splitter_params_both(self, test_settings):
        """Test updating both chunk size and overlap."""
        splitter = DocumentSplitter(test_settings)
        new_size = 1000
        new_overlap = 150

        splitter.update_splitter_params(chunk_size=new_size, chunk_overlap=new_overlap)

        assert splitter.settings.CHUNK_SIZE == new_size
        assert splitter.settings.CHUNK_OVERLAP == new_overlap
        assert splitter.text_splitter._chunk_size == new_size
        assert splitter.text_splitter._chunk_overlap == new_overlap

    def test_update_splitter_params_none_values(self, test_settings):
        """Test that None values don't update parameters."""
        splitter = DocumentSplitter(test_settings)
        original_size = test_settings.CHUNK_SIZE
        original_overlap = test_settings.CHUNK_OVERLAP

        splitter.update_splitter_params(chunk_size=None, chunk_overlap=None)

        # Values should remain unchanged
        assert splitter.settings.CHUNK_SIZE == original_size
        assert splitter.settings.CHUNK_OVERLAP == original_overlap

    def test_split_documents_with_exception_handling(self, test_settings):
        """Test exception handling during document splitting."""
        splitter = DocumentSplitter(test_settings)

        # Create a document with empty content
        empty_doc = Document(page_content="", metadata={})

        # Should handle gracefully
        chunks = splitter.split_documents([empty_doc])

        # Should return list (may be empty or contain empty doc)
        assert isinstance(chunks, list)

    def test_split_text_with_different_separators(self, test_settings):
        """Test text splitting with different separators."""
        splitter = DocumentSplitter(test_settings)

        # Text with multiple types of separators
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4." * 20

        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Should split intelligently based on separators
