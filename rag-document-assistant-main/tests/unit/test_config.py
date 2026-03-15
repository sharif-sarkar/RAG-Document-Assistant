"""
Unit tests for the configuration module (src/config/settings.py).
Tests Settings class initialization, validation, and environment variable loading.
"""

import pytest
from pathlib import Path
import os
from src.config.settings import Settings, get_settings


class TestSettings:
    """Test cases for Settings class."""

    def test_settings_initialization_with_defaults(self):
        """Test Settings initialization with default values."""
        settings = Settings()

        assert settings.LLM_MODEL == "llama3.2:3b"
        assert settings.EMBEDDING_MODEL == "nomic-embed-text:v1.5"
        assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
        assert settings.CHUNK_SIZE == 1200
        assert settings.CHUNK_OVERLAP == 300
        assert settings.RETRIEVAL_MODE == "multi_query"
        assert settings.NUM_RETRIEVED_DOCS == 4
        assert settings.APP_MODE == "cli"
        assert settings.LOG_LEVEL == "INFO"

    def test_settings_initialization_with_custom_values(self):
        """Test Settings initialization with custom values."""
        settings = Settings(
            LLM_MODEL="custom_model",
            EMBEDDING_MODEL="custom_embedding",
            CHUNK_SIZE=1000,
            CHUNK_OVERLAP=200,
        )

        assert settings.LLM_MODEL == "custom_model"
        assert settings.EMBEDDING_MODEL == "custom_embedding"
        assert settings.CHUNK_SIZE == 1000
        assert settings.CHUNK_OVERLAP == 200

    def test_settings_path_resolution(self, temp_dir):
        """Test that paths are properly resolved."""
        settings = Settings(
            INPUT_DATA_DIR=temp_dir / "input",
            OUTPUT_DATA_DIR=temp_dir / "output",
        )

        assert settings.INPUT_DATA_DIR.is_absolute()
        assert settings.OUTPUT_DATA_DIR.is_absolute()
        assert settings.VECTOR_STORE_PATH.is_absolute()

    def test_settings_vector_store_path_derivation(self, temp_dir):
        """Test that VECTOR_STORE_PATH is derived from OUTPUT_DATA_DIR."""
        output_dir = temp_dir / "output"
        settings = Settings(OUTPUT_DATA_DIR=output_dir)

        expected_path = output_dir / "faiss_vector_store"
        assert settings.VECTOR_STORE_PATH == expected_path

    def test_settings_relative_path_resolution(self):
        """Test that relative paths are resolved relative to BASE_DIR."""
        settings = Settings(
            INPUT_DATA_DIR=Path("input_data"),
            OUTPUT_DATA_DIR=Path("output_data"),
        )

        # Paths should be absolute after post_init
        assert settings.INPUT_DATA_DIR.is_absolute()
        assert settings.OUTPUT_DATA_DIR.is_absolute()
        assert str(settings.INPUT_DATA_DIR).endswith("input_data")
        assert str(settings.OUTPUT_DATA_DIR).endswith("output_data")

    def test_settings_supported_extensions(self):
        """Test that supported extensions are correctly set."""
        settings = Settings()

        assert settings.SUPPORTED_EXTENSIONS == [".pdf", ".txt", ".md"]

    def test_get_settings_factory_function(self):
        """Test get_settings factory function."""
        settings = get_settings()

        assert isinstance(settings, Settings)
        assert settings.LLM_MODEL == "llama3.2:3b"

    def test_get_settings_with_overrides(self):
        """Test get_settings with runtime overrides."""
        settings = get_settings(
            LLM_MODEL="override_model",
            CHUNK_SIZE=800,
        )

        assert settings.LLM_MODEL == "override_model"
        assert settings.CHUNK_SIZE == 800
        # Default values should remain
        assert settings.EMBEDDING_MODEL == "nomic-embed-text:v1.5"

    def test_settings_extra_allow(self):
        """Test that extra fields are allowed."""
        settings = Settings(CUSTOM_FIELD="custom_value")

        # Should not raise an error due to extra="allow" in Config
        assert hasattr(settings, "CUSTOM_FIELD")
        assert settings.CUSTOM_FIELD == "custom_value"

    def test_settings_environment_variable_integration(self, monkeypatch):
        """Test that environment variables are loaded correctly."""
        monkeypatch.setenv("LLM_MODEL", "env_model")
        monkeypatch.setenv("CHUNK_SIZE", "2000")

        settings = Settings()

        assert settings.LLM_MODEL == "env_model"
        assert settings.CHUNK_SIZE == 2000
