import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from src.config.settings import Settings


class VectorStoreManager:
    """
    Manages vector database operations including creation, saving, and loading.
    Uses FAISS as the vector store backend with Ollama embeddings.
    """

    def __init__(self, settings: Settings):
        """
        Initialize vector store manager with configuration.

        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.embedding = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
        )

        self.vector_store: Optional[FAISS] = None

    def create_vector_store(
        self, documents: List[Document], save: bool = True
    ) -> FAISS:
        """
        Create a new vector store from documents.

        Args:
            documents: List of document chunks to embed
            save: Whether to save the vector store to disk

        Returns:
            Created FAISS vector store
        """
        if not documents:
            self.logger.error("No documents provided to create vector store")
            raise ValueError("Cannot create vector store from empty documents")

        try:
            self.vector_store = FAISS.from_documents(
                documents=documents, embedding=self.embedding
            )
            self.logger.info(f"Vector store created with {len(documents)} documents")

            if save:
                self.save_vector_store()

            return self.vector_store

        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    def load_vector_store(
        self, vector_store_path: Optional[Path] = None
    ) -> Optional[FAISS]:
        """
        Load an existing vector store from disk.

        Args:
            vector_store_path: Path to the saved vector store

        Returns:
            Loaded FAISS vector store or None if loading fails
        """
        if vector_store_path is None:
            vector_store_path = self.settings.VECTOR_STORE_PATH

        if vector_store_path is None or not vector_store_path.exists():
            self.logger.warning(f"Vector store not found at {vector_store_path}")
            return None

        try:
            self.vector_store = FAISS.load_local(
                folder_path=str(vector_store_path),
                embeddings=self.embedding,
                allow_dangerous_deserialization=True,
            )
            self.logger.info(f"Vector store loaded from {vector_store_path}")
            return self.vector_store

        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return None

    def save_vector_store(self, vector_store_path: Optional[Path] = None) -> None:
        """
        Save the current vector store to disk.

        Args:
            vector_store_path: Path where to save the vector store
        """
        if self.vector_store is None:
            self.logger.error("No vector store to save")
            return

        if vector_store_path is None:
            vector_store_path = self.settings.VECTOR_STORE_PATH

        if vector_store_path is None:
            self.logger.error("No vector store path specified")
            return

        try:
            vector_store_path = Path(vector_store_path)
            vector_store_path.mkdir(parents=True, exist_ok=True)

            self.vector_store.save_local(folder_path=str(vector_store_path))
            self.logger.info(f"Vector store saved to {vector_store_path}")

        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")

    def get_or_create_vector_store(
        self, documents: Optional[List[Document]] = None, force_create: bool = False
    ) -> FAISS:
        """
        Get existing vector store or create a new one.

        Args:
            documents: Documents to use for creation (required if creating)
            force_create: Force creation even if existing store exists

        Returns:
            FAISS vector store
        """
        if not force_create:
            existing_store = self.load_vector_store()
            if existing_store is not None:
                return existing_store

        if documents is None:
            raise ValueError(
                "Documents must be provided when creating a new vector store"
            )

        return self.create_vector_store(documents, save=True)

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever from the current vector store.

        Args:
            search_kwargs: Additional search parameters for the retriever

        Returns:
            Vector store retriever
        """
        if self.vector_store is None:
            self.logger.error("No vector store loaded. Load or create one first.")
            raise ValueError("Vector store not initialized")

        if search_kwargs is None:
            search_kwargs = {"k": self.settings.NUM_RETRIEVED_DOCS}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def update_embedding_model(self, model: str) -> None:
        """
        Update the embedding model.

        Args:
            model: New embedding model name
        """
        self.settings.EMBEDDING_MODEL = model
        self.embedding = OllamaEmbeddings(
            model=model, base_url=self.settings.OLLAMA_BASE_URL
        )
        self.logger.info(f"Embedding model updated to {model}")

    def delete_vector_store(self, vector_store_path: Optional[Path] = None) -> None:
        """
        Delete the saved vector store from disk.

        Args:
            vector_store_path: Path to the vector store to delete
        """
        if vector_store_path is None:
            vector_store_path = self.settings.VECTOR_STORE_PATH

        if vector_store_path is None:
            self.logger.error("No vector store path specified")
            return

        try:
            import shutil

            if vector_store_path.exists():
                shutil.rmtree(vector_store_path)
                self.logger.info(f"Vector store deleted from {vector_store_path}")
            else:
                self.logger.warning(f"Vector store not found at {vector_store_path}")

        except Exception as e:
            self.logger.error(f"Error deleting vector store: {str(e)}")
