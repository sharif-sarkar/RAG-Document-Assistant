"""
Unit tests for the retrieval module (src/retrieval/retriever.py and rag_chain.py).
Tests RetrieverFactory and RAGChainFactory for creating different retrieval strategies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.retrieval.retriever import RetrieverFactory
from src.retrieval.rag_chain import RAGChainFactory


class TestRetrieverFactory:
    """Test cases for RetrieverFactory class."""

    def test_factory_initialization(self, test_settings):
        """Test RetrieverFactory initialization."""
        factory = RetrieverFactory(test_settings)

        assert factory.settings == test_settings
        assert factory.logger is not None

    def test_create_basic_retriever(self, test_settings):
        """Test creating a basic retriever."""
        factory = RetrieverFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        result = factory.create_retriever(mock_retriever, mock_llm, mode="basic")

        # Basic mode should return the same retriever
        assert result == mock_retriever

    def test_create_multi_query_retriever(self, test_settings):
        """Test creating a multi-query retriever."""
        factory = RetrieverFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        with patch("src.retrieval.retriever.MultiQueryRetriever") as mock_mqr:
            mock_instance = MagicMock()
            mock_mqr.from_llm.return_value = mock_instance

            result = factory.create_retriever(
                mock_retriever, mock_llm, mode="multi_query"
            )

            assert result == mock_instance
            mock_mqr.from_llm.assert_called_once()

    def test_create_retriever_default_mode(self, test_settings):
        """Test creating retriever with default mode from settings."""
        test_settings.RETRIEVAL_MODE = "multi_query"
        factory = RetrieverFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        with patch("src.retrieval.retriever.MultiQueryRetriever") as mock_mqr:
            mock_mqr.from_llm.return_value = MagicMock()
            factory.create_retriever(mock_retriever, mock_llm)
            mock_mqr.from_llm.assert_called_once()

    def test_create_retriever_invalid_mode(self, test_settings):
        """Test creating retriever with invalid mode falls back to basic."""
        factory = RetrieverFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        result = factory.create_retriever(mock_retriever, mock_llm, mode="invalid_mode")

        # Should fall back to basic retriever
        assert result == mock_retriever

    def test_multi_query_retriever_exception_fallback(self, test_settings):
        """Test that multi-query retriever falls back to basic on error."""
        factory = RetrieverFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        with patch("src.retrieval.retriever.MultiQueryRetriever") as mock_mqr:
            mock_mqr.from_llm.side_effect = Exception("MultiQuery error")

            result = factory.create_retriever(
                mock_retriever, mock_llm, mode="multi_query"
            )

            # Should fall back to basic retriever
            assert result == mock_retriever

    def test_update_retrieval_mode(self, test_settings):
        """Test updating the retrieval mode."""
        factory = RetrieverFactory(test_settings)

        factory.update_retrieval_mode("basic")

        assert factory.settings.RETRIEVAL_MODE == "basic"

    def test_get_available_modes(self, test_settings):
        """Test getting available retrieval modes."""
        factory = RetrieverFactory(test_settings)

        modes = factory.get_available_modes()

        assert "basic" in modes
        assert "multi_query" in modes
        assert len(modes) == 2


class TestRAGChainFactory:
    """Test cases for RAGChainFactory class."""

    def test_rag_factory_initialization(self, test_settings):
        """Test RAGChainFactory initialization."""
        factory = RAGChainFactory(test_settings)

        assert factory.settings == test_settings
        assert factory.logger is not None

    def test_create_basic_rag_chain(self, test_settings):
        """Test creating a basic RAG chain."""
        factory = RAGChainFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        chain = factory.create_basic_rag_chain(mock_retriever, mock_llm)

        assert chain is not None

    def test_create_basic_rag_chain_custom_template(self, test_settings):
        """Test creating basic RAG chain with custom template."""
        factory = RAGChainFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        custom_template = "Custom: {context}\nQ: {question}"

        chain = factory.create_basic_rag_chain(
            mock_retriever, mock_llm, custom_template=custom_template
        )

        assert chain is not None

    def test_create_conversational_rag_chain(self, test_settings):
        """Test creating a conversational RAG chain."""
        factory = RAGChainFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_rephrase_llm = MagicMock()

        chain = factory.create_conversational_rag_chain(
            mock_retriever, mock_llm, mock_rephrase_llm
        )

        assert chain is not None

    def test_create_rag_chain_basic_mode(self, test_settings):
        """Test create_rag_chain with basic mode."""
        factory = RAGChainFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        chain = factory.create_rag_chain(mock_retriever, mock_llm, mode="basic")

        assert chain is not None

    def test_create_rag_chain_conversational_mode(self, test_settings):
        """Test create_rag_chain with conversational mode."""
        factory = RAGChainFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        chain = factory.create_rag_chain(
            mock_retriever, mock_llm, mode="conversational"
        )

        assert chain is not None

    def test_create_rag_chain_conversational_default_rephrase_llm(self, test_settings):
        """Test conversational mode uses main LLM if no rephrase LLM provided."""
        factory = RAGChainFactory(test_settings)
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        # Should use main LLM for rephrasing if not provided
        chain = factory.create_rag_chain(
            mock_retriever,
            mock_llm,
            mode="conversational",
            question_rephrasing_llm=None,
        )

        assert chain is not None

    def test_update_template(self, test_settings):
        """Test updating the RAG template."""
        factory = RAGChainFactory(test_settings)
        new_template = "New template: {context}\nQuestion: {question}"

        factory.update_template(new_template)

        assert factory.DEFAULT_RAG_TEMPLATE == new_template

    def test_default_templates_exist(self, test_settings):
        """Test that default templates are defined."""
        factory = RAGChainFactory(test_settings)

        assert factory.DEFAULT_RAG_TEMPLATE is not None
        assert factory.CONVERSATIONAL_RAG_TEMPLATE is not None
        assert factory.ANSWER_TEMPLATE is not None

        # Templates should contain expected placeholders
        assert "{context}" in factory.DEFAULT_RAG_TEMPLATE
        assert "{question}" in factory.DEFAULT_RAG_TEMPLATE
