import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

from src.config.settings import Settings


class DocumentIngestor:
    """
    Handles loading of documents from various file formats.
    Supports PDF, TXT, and MD file types.
    """

    def __init__(self, settings: Settings):
        """
        Initialize document ingestor with configuration.

        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def load_single_file(self, file_path: Union[str, Path]) -> Optional[List[Document]]:
        """
        Load a single document file based on its extension.

        Args:
            file_path: Path to the document file

        Returns:
            List of loaded documents or None if loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        file_extension = file_path.suffix.lower()

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path=str(file_path))
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(file_path=str(file_path))
            else:
                self.logger.warning(f"Unsupported file extension: {file_extension}")
                return None

            documents = loader.load()
            self.logger.info(
                f"Successfully loaded {len(documents)} documents from {file_path.name}"
            )
            return documents

        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def load_from_directory(
        self, directory: Optional[Union[str, Path]] = None, recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Directory path (defaults to INPUT_DATA_DIR from settings)
            recursive: Whether to search subdirectories

        Returns:
            List of all loaded documents
        """
        if directory is None:
            directory = self.settings.INPUT_DATA_DIR

        directory = Path(directory)

        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []

        all_documents = []

        for ext in self.settings.SUPPORTED_EXTENSIONS:
            try:
                # For PDF files, use PyPDFLoader directly to avoid heavy
                # unstructured dependency
                if ext == ".pdf":
                    # Find all PDF files using glob
                    pattern = "**/*{}" if recursive else "*{}"
                    pdf_files = list(directory.glob(pattern.format(ext)))

                    self.logger.info(f"Found {len(pdf_files)} PDF files")

                    # Load each PDF file with PyPDFLoader
                    for pdf_file in pdf_files:
                        try:
                            loader = PyPDFLoader(file_path=str(pdf_file))
                            docs = loader.load()
                            all_documents.extend(docs)
                            self.logger.debug(
                                f"Loaded {len(docs)} pages from "
                                f"{pdf_file.name}"
                            )
                        except Exception as pdf_error:
                            self.logger.error(
                                f"Error loading PDF {pdf_file.name}: "
                                f"{str(pdf_error)}"
                            )

                    pdf_count = len([
                        d for d in all_documents
                        if d.metadata.get('source', '').endswith('.pdf')
                    ])
                    self.logger.info(
                        f"Loaded {pdf_count} documents from {ext} files"
                    )
                else:
                    # For text files, use DirectoryLoader (works fine)
                    loader = DirectoryLoader(
                        path=str(directory),
                        glob=f"**/*{ext}" if recursive else f"*{ext}",
                        show_progress=True,
                    )
                    documents = loader.load()
                    all_documents.extend(documents)
                    self.logger.info(
                        f"Loaded {len(documents)} documents from {ext} files"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error loading {ext} files from {directory}: {str(e)}"
                )

        self.logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

    def load_documents(
        self, path: Optional[Union[str, Path]] = None, recursive: bool = True
    ) -> List[Document]:
        """
        Load documents from a file or directory.

        Args:
            path: Path to file or directory (defaults to INPUT_DATA_DIR)
            recursive: Whether to search subdirectories (only for directories)

        Returns:
            List of loaded documents
        """
        if path is None:
            path = self.settings.INPUT_DATA_DIR

        path = Path(path)

        if path.is_file():
            docs = self.load_single_file(path)
            return docs if docs else []

        elif path.is_dir():
            return self.load_from_directory(path, recursive)

        else:
            self.logger.error(f"Path not found: {path}")
            return []
