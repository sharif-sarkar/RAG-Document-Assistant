"""
Unit tests for the vector store module (src/vector_store/vector_store.py).
Tests VectorStoreManager for creating, saving, loading, and deleting vector stores.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.vector_store.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """Test cases for VectorStoreManager class."""

    def test_manager_initialization(self, test_settings):
        """Test VectorStoreManager initialization."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)

            assert manager.settings == test_settings
            assert manager.logger is not None
            assert manager.vector_store is None

    def test_create_vector_store_success(self, test_settings, sample_documents):
        """Test  successful vector store creation."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"), patch(
            "src.vector_store.vector_store.FAISS"
        ) as mock_faiss:

            mock_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_store

            manager = VectorStoreManager(test_settings)
            result = manager.create_vector_store(sample_documents, save=False)

            assert result == mock_store
            assert manager.vector_store == mock_store
            mock_faiss.from_documents.assert_called_once()

    def test_create_vector_store_with_save(self, test_settings, sample_documents):
        """Test vector store creation with automatic save."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"), patch(
            "src.vector_store.vector_store.FAISS"
        ) as mock_faiss:

            mock_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_store

            manager = VectorStoreManager(test_settings)

            with patch.object(manager, "save_vector_store") as mock_save:
                manager.create_vector_store(sample_documents, save=True)
                mock_save.assert_called_once()

    def test_create_vector_store_empty_documents(self, test_settings):
        """Test vector store creation with empty documents list."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)

            with pytest.raises(ValueError, match="empty documents"):
                manager.create_vector_store([], save=False)

    def test_load_vector_store_success(self, test_settings):
        """Test successful vector store loading."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"), patch(
            "src.vector_store.vector_store.FAISS"
        ) as mock_faiss:

            mock_store = MagicMock()
            mock_faiss.load_local.return_value = mock_store

            # Create vector store path
            vector_store_path = test_settings.OUTPUT_DATA_DIR / "test_store"
            vector_store_path.mkdir(parents=True, exist_ok=True)

            manager = VectorStoreManager(test_settings)
            result = manager.load_vector_store(vector_store_path)

            assert result == mock_store
            assert manager.vector_store == mock_store

    def test_load_vector_store_nonexistent_path(self, test_settings):
        """Test loading vector store from nonexistent path."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)
            nonexistent_path = test_settings.OUTPUT_DATA_DIR / "nonexistent"

            result = manager.load_vector_store(nonexistent_path)

            assert result is None

    def test_load_vector_store_with_exception(self, test_settings):
        """Test error handling during vector store loading."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"), patch(
            "src.vector_store.vector_store.FAISS"
        ) as mock_faiss:

            mock_faiss.load_local.side_effect = Exception("Loading error")

            vector_store_path = test_settings.OUTPUT_DATA_DIR / "test_store"
            vector_store_path.mkdir(parents=True, exist_ok=True)

            manager = VectorStoreManager(test_settings)
            result = manager.load_vector_store(vector_store_path)

            assert result is None

    def test_save_vector_store_success(self, test_settings):
        """Test successful vector store saving."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)
            manager.vector_store = MagicMock()

            save_path = test_settings.OUTPUT_DATA_DIR / "saved_store"
            manager.save_vector_store(save_path)

            manager.vector_store.save_local.assert_called_once()

    def test_save_vector_store_no_store(self, test_settings):
        """Test saving when no vector store is loaded."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)
            manager.vector_store = None

            # Should handle gracefully, not raise
            manager.save_vector_store()

    def test_get_or_create_vector_store_loads_existing(self, test_settings):
        """Test get_or_create loads existing store."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)

            mock_store = MagicMock()
            with patch.object(manager, "load_vector_store", return_value=mock_store):
                result = manager.get_or_create_vector_store(force_create=False)

                assert result == mock_store

    def test_get_or_create_vector_store_creates_new(
        self, test_settings, sample_documents
    ):
        """Test get_or_create creates new store when none exists."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)

            mock_store = MagicMock()
            with patch.object(
                manager, "load_vector_store", return_value=None
            ), patch.object(manager, "create_vector_store", return_value=mock_store):

                result = manager.get_or_create_vector_store(documents=sample_documents)

                assert result == mock_store

    def test_get_or_create_vector_store_force_create(
        self, test_settings, sample_documents
    ):
        """Test get_or_create with force_create flag."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)

            mock_store = MagicMock()
            with patch.object(manager, "create_vector_store", return_value=mock_store):
                result = manager.get_or_create_vector_store(
                    documents=sample_documents, force_create=True
                )

                assert result == mock_store

    def test_get_retriever_success(self, test_settings):
        """Test getting a retriever from vector store."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)
            manager.vector_store = MagicMock()

            retriever = manager.get_retriever()

            manager.vector_store.as_retriever.assert_called_once()

    def test_get_retriever_no_store(self, test_settings):
        """Test getting retriever when no store is loaded."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)
            manager.vector_store = None

            with pytest.raises(ValueError, match="not initialized"):
                manager.get_retriever()

    def test_get_retriever_custom_search_kwargs(self, test_settings):
        """Test getting retriever with custom search parameters."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)
            manager.vector_store = MagicMock()

            custom_kwargs = {"k": 10, "score_threshold": 0.5}
            manager.get_retriever(search_kwargs=custom_kwargs)

            manager.vector_store.as_retriever.assert_called_once_with(
                search_kwargs=custom_kwargs
            )

    def test_update_embedding_model(self, test_settings):
        """Test updating the embedding model."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings") as mock_embedding:
            manager = VectorStoreManager(test_settings)
            new_model = "new_embedding_model"

            manager.update_embedding_model(new_model)

            assert manager.settings.EMBEDDING_MODEL == new_model
            # Should create new embedding instance
            assert mock_embedding.call_count >= 2  # Initial + update

    def test_delete_vector_store_success(self, test_settings):
        """Test successful vector store deletion."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)

            store_path = test_settings.OUTPUT_DATA_DIR / "to_delete"
            store_path.mkdir(parents=True, exist_ok=True)

            manager.delete_vector_store(store_path)

            assert not store_path.exists()

    def test_delete_vector_store_nonexistent(self, test_settings):
        """Test deleting nonexistent vector store."""
        with patch("src.vector_store.vector_store.OllamaEmbeddings"):
            manager = VectorStoreManager(test_settings)

            nonexistent_path = test_settings.OUTPUT_DATA_DIR / "nonexistent"

            # Should handle gracefully
            manager.delete_vector_store(nonexistent_path)
