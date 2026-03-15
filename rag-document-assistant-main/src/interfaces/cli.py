import argparse
import logging
import sys
from typing import Optional

from langchain_ollama import ChatOllama

from src.config.settings import get_settings
from src.ingestion.loader import DocumentIngestor
from src.processing.splitter import DocumentSplitter
from src.vector_store.vector_store import VectorStoreManager
from src.retrieval.retriever import RetrieverFactory
from src.retrieval.rag_chain import RAGChainFactory


class CLIInterface:
    """
    Command-line interface for the RAG application.
    Supports document processing and interactive querying.
    """

    def __init__(self, settings):
        """
        Initialize CLI interface.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.ingestor = DocumentIngestor(settings)
        self.splitter = DocumentSplitter(settings)
        self.vector_store_manager = VectorStoreManager(settings)
        self.retriever_factory = RetrieverFactory(settings)
        self.chain_factory = RAGChainFactory(settings)

        self.llm = ChatOllama(
            model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL
        )

        self.chain = None

    def initialize_vector_store(self, force_recreate: bool = False) -> bool:
        """
        Initialize the vector store by loading existing or creating new.

        Args:
            force_recreate: Force recreation of vector store

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing vector store...")

            if not force_recreate:
                existing_store = self.vector_store_manager.load_vector_store()
                if existing_store is not None:
                    self.logger.info("Loaded existing vector store")
                    return True

            self.logger.info("Creating new vector store...")
            documents = self.ingestor.load_documents()

            if not documents:
                self.logger.error("No documents loaded. Cannot create vector store.")
                return False

            chunks = self.splitter.split_documents(documents)
            self.vector_store_manager.create_vector_store(chunks, save=True)

            self.logger.info("Vector store created and saved")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            return False

    def initialize_chain(
        self, retrieval_mode: Optional[str] = None, chain_mode: str = "basic"
    ) -> bool:
        """
        Initialize the RAG chain.

        Args:
            retrieval_mode: Retrieval mode ('basic' or 'multi_query')
            chain_mode: Chain mode ('basic' or 'conversational')

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing RAG chain...")

            vector_retriever = self.vector_store_manager.get_retriever()
            retriever = self.retriever_factory.create_retriever(
                vector_retriever, self.llm, mode=retrieval_mode
            )

            self.chain = self.chain_factory.create_rag_chain(
                retriever, self.llm, mode=chain_mode
            )

            self.logger.info("RAG chain initialized")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing chain: {str(e)}")
            return False

    def query(self, question: str) -> str:
        """
        Query the RAG system.

        Args:
            question: User question

        Returns:
            Response from the RAG system
        """
        try:
            if self.chain is None:
                return "Error: Chain not initialized. Run initialization first."

            response = self.chain.invoke(question)
            return response

        except Exception as e:
            self.logger.error(f"Error querying: {str(e)}")
            return f"Error: {str(e)}"

    def interactive_mode(self):
        """
        Run interactive CLI mode for querying.
        """
        print("\n=== RAG Interactive Mode ===")
        print("Type 'exit' or 'quit' to exit")
        print("Type 'help' for available commands")
        print("=" + "=" * 30)

        while True:
            try:
                question = input("\nYour question: ").strip()

                if question.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if question.lower() == "help":
                    print("\nAvailable commands:")
                    print("  - Type your question to query the RAG system")
                    print("  - 'exit' or 'quit': Exit the program")
                    print("  - 'help': Show this help message")
                    continue

                if not question:
                    continue

                print("\nThinking...")
                response = self.query(question)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    def single_query_mode(self, question: str):
        """
        Run single query mode.

        Args:
            question: User question
        """
        response = self.query(question)
        print(response)


def main():
    """
    Main entry point for CLI interface.
    """
    parser = argparse.ArgumentParser(
        description="RAG Application - Command Line Interface"
    )
    parser.add_argument(
        "--cli-mode",
        choices=["interactive", "query"],
        default="interactive",
        help="Mode of operation",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (required for 'query' mode)"
    )
    parser.add_argument(
        "--input-dir", type=str, help="Input directory containing documents"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for vector store"
    )
    parser.add_argument(
        "--llm-model", type=str, help="LLM model name (default: llama3.2:3b)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model name (default: nomic-embed-text:v1.5)",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["basic", "multi_query"],
        default="multi_query",
        help="Retrieval mode",
    )
    parser.add_argument(
        "--chain-mode",
        choices=["basic", "conversational"],
        default="basic",
        help="Chain mode",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of vector store"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get settings with overrides
    settings_kwargs = {}
    if args.input_dir:
        settings_kwargs["INPUT_DATA_DIR"] = args.input_dir
    if args.output_dir:
        settings_kwargs["OUTPUT_DATA_DIR"] = args.output_dir
    if args.llm_model:
        settings_kwargs["LLM_MODEL"] = args.llm_model
    if args.embedding_model:
        settings_kwargs["EMBEDDING_MODEL"] = args.embedding_model
    if args.retrieval_mode:
        settings_kwargs["RETRIEVAL_MODE"] = args.retrieval_mode

    settings = get_settings(**settings_kwargs)

    # Initialize CLI interface
    cli = CLIInterface(settings)

    # Initialize vector store
    if not cli.initialize_vector_store(force_recreate=args.force_recreate):
        print("Failed to initialize vector store. Exiting.")
        sys.exit(1)

    # Initialize chain
    if not cli.initialize_chain(
        retrieval_mode=args.retrieval_mode, chain_mode=args.chain_mode
    ):
        print("Failed to initialize chain. Exiting.")
        sys.exit(1)

    # Run mode
    if args.cli_mode == "query":
        if not args.question:
            print("Error: --question is required for query mode")
            sys.exit(1)
        cli.single_query_mode(args.question)
    else:
        cli.interactive_mode()


if __name__ == "__main__":
    main()
