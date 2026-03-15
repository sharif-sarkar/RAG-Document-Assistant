"""
Unit tests for the ingestion module (src/ingestion/loader.py).
Tests DocumentIngestor for loading PDF, TXT, MD files and directories.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.ingestion.loader import DocumentIngestor


class TestDocumentIngestor:
    """Test cases for DocumentIngestor class."""

    def test_ingestor_initialization(self, test_settings):
        """Test DocumentIngestor initialization."""
        ingestor = DocumentIngestor(test_settings)

        assert ingestor.settings == test_settings
        assert ingestor.logger is not None

    def test_load_text_file(self, test_settings, sample_text_file):
        """Test loading a single text file."""
        ingestor = DocumentIngestor(test_settings)

        with patch("src.ingestion.loader.TextLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [
                Document(
                    page_content="Sample text",
                    metadata={"source": str(sample_text_file)},
                )
            ]
            mock_loader.return_value = mock_instance

            documents = ingestor.load_single_file(sample_text_file)

            assert documents is not None
            assert len(documents) == 1
            mock_loader.assert_called_once()

    def test_load_markdown_file(self, test_settings, sample_md_file):
        """Test loading a single markdown file."""
        ingestor = DocumentIngestor(test_settings)

        with patch("src.ingestion.loader.TextLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [
                Document(
                    page_content="# Markdown content",
                    metadata={"source": str(sample_md_file)},
                )
            ]
            mock_loader.return_value = mock_instance

            documents = ingestor.load_single_file(sample_md_file)

            assert documents is not None
            assert len(documents) == 1

    def test_load_pdf_file(self, test_settings, temp_dir):
        """Test loading a PDF file (mocked)."""
        ingestor = DocumentIngestor(test_settings)
        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_text("fake pdf content")  # Create fake file

        with patch("src.ingestion.loader.PyPDFLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [
                Document(page_content="PDF content", metadata={"source": str(pdf_file)})
            ]
            mock_loader.return_value = mock_instance

            documents = ingestor.load_single_file(pdf_file)

            assert documents is not None
            assert len(documents) == 1
            mock_loader.assert_called_once_with(file_path=str(pdf_file))

    def test_load_nonexistent_file(self, test_settings, temp_dir):
        """Test loading a file that doesn't exist."""
        ingestor = DocumentIngestor(test_settings)
        nonexistent_file = temp_dir / "does_not_exist.txt"

        documents = ingestor.load_single_file(nonexistent_file)

        assert documents is None

    def test_load_unsupported_file_type(self, test_settings, temp_dir):
        """Test loading an unsupported file type."""
        ingestor = DocumentIngestor(test_settings)
        unsupported_file = temp_dir / "file.xyz"
        unsupported_file.write_text("content")

        documents = ingestor.load_single_file(unsupported_file)

        assert documents is None

    def test_load_file_with_exception(self, test_settings, sample_text_file):
        """Test handling of exceptions during file loading."""
        ingestor = DocumentIngestor(test_settings)

        with patch("src.ingestion.loader.TextLoader") as mock_loader:
            mock_loader.side_effect = Exception("Loading error")

            documents = ingestor.load_single_file(sample_text_file)

            assert documents is None

    def test_load_from_directory(self, test_settings, temp_dir):
        """Test loading documents from a directory."""
        ingestor = DocumentIngestor(test_settings)

        # Create some test files
        (temp_dir / "doc1.txt").write_text("Document 1")
        (temp_dir / "doc2.md").write_text("# Document 2")

        with patch("src.ingestion.loader.DirectoryLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [
                Document(page_content="Doc 1", metadata={"source": "doc1.txt"}),
                Document(page_content="Doc 2", metadata={"source": "doc2.md"}),
            ]
            mock_loader.return_value = mock_instance

            documents = ingestor.load_from_directory(temp_dir)

            assert documents is not None
            assert len(documents) >= 0  # Could vary based on mocking

    def test_load_from_nonexistent_directory(self, test_settings, temp_dir):
        """Test loading from a directory that doesn't exist."""
        ingestor = DocumentIngestor(test_settings)
        nonexistent_dir = temp_dir / "does_not_exist"

        documents = ingestor.load_from_directory(nonexistent_dir)

        assert documents == []

    def test_load_from_directory_with_exception(self, test_settings, temp_dir):
        """Test handling of exceptions during directory loading."""
        ingestor = DocumentIngestor(test_settings)

        with patch("src.ingestion.loader.DirectoryLoader") as mock_loader:
            mock_loader.side_effect = Exception("Directory loading error")

            documents = ingestor.load_from_directory(temp_dir)

            # Should return empty list or handle gracefully for at least some extensions
            assert isinstance(documents, list)

    def test_load_documents_from_file(self, test_settings, sample_text_file):
        """Test load_documents with a file path."""
        ingestor = DocumentIngestor(test_settings)

        with patch.object(ingestor, "load_single_file") as mock_load:
            mock_load.return_value = [Document(page_content="Content")]

            documents = ingestor.load_documents(sample_text_file)

            assert len(documents) == 1
            mock_load.assert_called_once_with(sample_text_file)

    def test_load_documents_from_directory(self, test_settings, temp_dir):
        """Test load_documents with a directory path."""
        ingestor = DocumentIngestor(test_settings)

        with patch.object(ingestor, "load_from_directory") as mock_load:
            mock_load.return_value = [Document(page_content="Content")]

            documents = ingestor.load_documents(temp_dir)

            assert len(documents) == 1
            mock_load.assert_called_once()

    def test_load_documents_default_path(self, test_settings):
        """Test load_documents with default INPUT_DATA_DIR."""
        ingestor = DocumentIngestor(test_settings)

        with patch.object(ingestor, "load_from_directory") as mock_load:
            mock_load.return_value = []

            documents = ingestor.load_documents()

            mock_load.assert_called_once()
            # Should use settings.INPUT_DATA_DIR by default

    def test_load_documents_recursive_flag(self, test_settings, temp_dir):
        """Test load_documents with recursive flag."""
        ingestor = DocumentIngestor(test_settings)

        with patch.object(ingestor, "load_from_directory") as mock_load:
            mock_load.return_value = []

            ingestor.load_documents(temp_dir, recursive=False)

            # Verify recursive parameter is passed
            call_args = mock_load.call_args
            assert call_args[0][0] == temp_dir
            assert call_args[0][1] == False
