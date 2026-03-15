import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import Settings


class DocumentSplitter:
    """
    Handles splitting of documents into smaller, manageable chunks for processing.
    Uses RecursiveCharacterTextSplitter for intelligent chunking.
    """

    def __init__(self, settings: Settings):
        """
        Initialize document splitter with configuration.

        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of split document chunks
        """
        if not documents:
            self.logger.warning("No documents provided for splitting")
            return []

        try:
            chunks = self.text_splitter.split_documents(documents)
            self.logger.info(
                f"Split {len(documents)} documents into {len(chunks)} chunks"
            )
            return chunks

        except Exception as e:
            self.logger.error(f"Error splitting documents: {str(e)}")
            return []

    def split_text(self, text: str) -> List[str]:
        """
        Split a text string into chunks.

        Args:
            text: Text string to split

        Returns:
            List of text chunks
        """
        try:
            chunks = self.text_splitter.split_text(text)
            self.logger.info(f"Split text into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            return []

    def update_splitter_params(
        self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ) -> None:
        """
        Update the splitter parameters at runtime.

        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        if chunk_size is not None:
            self.settings.CHUNK_SIZE = chunk_size
        if chunk_overlap is not None:
            self.settings.CHUNK_OVERLAP = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        self.logger.info(
            f"Updated splitter: chunk_size={self.settings.CHUNK_SIZE}, "
            f"chunk_overlap={self.settings.CHUNK_OVERLAP}"
        )
