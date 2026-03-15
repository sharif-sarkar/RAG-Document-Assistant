import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration settings with environment variable support.
    All settings can be overridden using environment variables or runtime parameters.
    """

    # Directory paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    INPUT_DATA_DIR: Path = Field(default_factory=lambda: Path("input_data"))
    OUTPUT_DATA_DIR: Path = Field(default_factory=lambda: Path("output_data"))

    # Model configuration
    LLM_MODEL: str = "llama3.2:3b"
    EMBEDDING_MODEL: str = "nomic-embed-text:v1.5"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Document processing
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300
    SUPPORTED_EXTENSIONS: list = Field(default_factory=lambda: [".pdf", ".txt", ".md"])

    # Vector store
    VECTOR_STORE_NAME: str = "faiss_vector_store"
    VECTOR_STORE_PATH: Optional[Path] = None

    # Retrieval
    RETRIEVAL_MODE: str = "multi_query"
    NUM_RETRIEVED_DOCS: int = 4
    RETRIEVAL_KWARGS: dict = Field(default_factory=dict)

    # Application mode (cli or streamlit)
    APP_MODE: str = "cli"

    # Logging
    LOG_LEVEL: str = "INFO"

    def model_post_init(self, __context):
        """
        Post-initialization hook to set derived paths.
        """
        if self.VECTOR_STORE_PATH is None:
            self.VECTOR_STORE_PATH = self.OUTPUT_DATA_DIR / self.VECTOR_STORE_NAME

        if not self.INPUT_DATA_DIR.is_absolute():
            self.INPUT_DATA_DIR = self.BASE_DIR / self.INPUT_DATA_DIR

        if not self.OUTPUT_DATA_DIR.is_absolute():
            self.OUTPUT_DATA_DIR = self.BASE_DIR / self.OUTPUT_DATA_DIR

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


def get_settings(**kwargs) -> Settings:
    """
    Factory function to get Settings instance with optional runtime overrides.

    Args:
        **kwargs: Runtime parameter overrides

    Returns:
        Settings: Configured settings instance
    """
    base_settings = Settings()

    for key, value in kwargs.items():
        if hasattr(base_settings, key):
            setattr(base_settings, key, value)

    return base_settings
