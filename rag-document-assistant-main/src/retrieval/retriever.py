import logging
from typing import Optional

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

from src.config.settings import Settings


class RetrieverFactory:
    """
    Factory class for creating different types of retrievers.
    Supports basic and multi-query retrieval strategies.
    """

    # Multi-query prompt template for generating alternative questions
    MULTI_QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    def __init__(self, settings: Settings):
        """
        Initialize retriever factory with configuration.

        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def create_retriever(self, vector_store_retriever, llm, mode: Optional[str] = None):
        """
        Create a retriever based on the specified mode.

        Args:
            vector_store_retriever: Base vector store retriever
            llm: Language model instance (required for multi_query mode)
            mode: Retrieval mode ('basic' or 'multi_query')

        Returns:
            Configured retriever
        """
        if mode is None:
            mode = self.settings.RETRIEVAL_MODE

        if mode not in ["basic", "multi_query"]:
            self.logger.warning(
                f"Unknown retrieval mode: {mode}, falling back to basic"
            )
            mode = "basic"

        if mode == "multi_query":
            return self._create_multi_query_retriever(vector_store_retriever, llm)
        elif mode == "basic":
            return self._create_basic_retriever(vector_store_retriever)
        else:
            self.logger.warning(
                f"Unknown retrieval mode: {mode}, falling back to basic"
            )
            return self._create_basic_retriever(vector_store_retriever)

    def _create_basic_retriever(self, vector_store_retriever):
        """
        Create a basic vector store retriever.

        Args:
            vector_store_retriever: Base vector store retriever

        Returns:
            Basic retriever
        """
        self.logger.info("Created basic retriever")
        return vector_store_retriever

    def _create_multi_query_retriever(self, vector_store_retriever, llm):
        """
        Create a multi-query retriever that generates alternative questions.

        Args:
            vector_store_retriever: Base vector store retriever
            llm: Language model for generating queries

        Returns:
            Multi-query retriever
        """
        try:
            retriever = MultiQueryRetriever.from_llm(
                retriever=vector_store_retriever,
                llm=llm,
                prompt=self.MULTI_QUERY_PROMPT,
            )
            self.logger.info("Created multi-query retriever")
            return retriever

        except Exception as e:
            self.logger.error(f"Error creating multi-query retriever: {str(e)}")
            self.logger.info("Falling back to basic retriever")
            return self._create_basic_retriever(vector_store_retriever)

    def update_retrieval_mode(self, mode: str) -> None:
        """
        Update default retrieval mode.

        Args:
            mode: New retrieval mode
        """
        self.settings.RETRIEVAL_MODE = mode
        self.logger.info(f"Retrieval mode updated to {mode}")

    def get_available_modes(self) -> list:
        """
        Get list of available retrieval modes.

        Returns:
            List of available mode names
        """
        return ["basic", "multi_query"]
