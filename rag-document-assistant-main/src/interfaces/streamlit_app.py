import logging
from typing import Optional

import streamlit as st
from langchain_ollama import ChatOllama

from src.config.settings import get_settings
from src.ingestion.loader import DocumentIngestor
from src.processing.splitter import DocumentSplitter
from src.vector_store.vector_store import VectorStoreManager
from src.retrieval.retriever import RetrieverFactory
from src.retrieval.rag_chain import RAGChainFactory


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    session_vars = ["initialized", "vector_store_loaded", "chat_history"]

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = False if var != "chat_history" else []


def setup_logging(log_level: str = "INFO"):
    """
    Configure logging for the application.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@st.cache_resource
def get_components(settings):
    """
    Initialize and cache application components.

    Args:
        settings: Application settings

    Returns:
        Tuple of (ingestor, splitter, vector_store_manager, retriever_factory, chain_factory)
    """
    ingestor = DocumentIngestor(settings)
    splitter = DocumentSplitter(settings)
    vector_store_manager = VectorStoreManager(settings)
    retriever_factory = RetrieverFactory(settings)
    chain_factory = RAGChainFactory(settings)

    return ingestor, splitter, vector_store_manager, retriever_factory, chain_factory


def initialize_vector_store(
    vector_store_manager, ingestor, splitter, force_recreate=False
):
    """
    Initialize or load the vector store.

    Args:
        vector_store_manager: VectorStoreManager instance
        ingestor: DocumentIngestor instance
        splitter: DocumentSplitter instance
        force_recreate: Force recreation of vector store

    Returns:
        True if successful, False otherwise
    """
    try:
        if not force_recreate:
            existing_store = vector_store_manager.load_vector_store()
            if existing_store is not None:
                return True

        documents = ingestor.load_documents()
        if not documents:
            st.error("No documents found in input directory.")
            return False

        chunks = splitter.split_documents(documents)
        vector_store_manager.create_vector_store(chunks, save=True)
        return True

    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return False


def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(page_title="Document Assistant", page_icon="📚", layout="wide")

    st.title("📚 Document Assistant")
    st.write("Ask questions about your documents using AI-powered RAG")

    initialize_session_state()

    with st.sidebar:
        st.header("Settings")

        input_dir = st.text_input(
            "Input Directory",
            value="input_data",
            help="Directory containing your documents",
        )

        output_dir = st.text_input(
            "Output Directory",
            value="output_data",
            help="Directory for vector store output",
        )

        llm_model = st.text_input(
            "LLM Model", value="llama3.2:3b", help="Ollama model name"
        )

        embedding_model = st.text_input(
            "Embedding Model",
            value="nomic-embed-text:v1.5",
            help="Ollama embedding model name",
        )

        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=["basic", "multi_query"],
            index=1,
            help="Strategy for retrieving documents",
        )

        chain_mode = st.selectbox(
            "Chain Mode",
            options=["basic", "conversational"],
            index=0,
            help="Type of RAG chain",
        )

        force_recreate = st.checkbox(
            "Force Vector Store Recreation",
            value=False,
            help="Delete existing vector store and create a new one",
        )

        st.divider()

        if st.button("Initialize System", use_container_width=True):
            setup_logging()

            settings = get_settings(
                INPUT_DATA_DIR=input_dir,
                OUTPUT_DATA_DIR=output_dir,
                LLM_MODEL=llm_model,
                EMBEDDING_MODEL=embedding_model,
                RETRIEVAL_MODE=retrieval_mode,
            )

            with st.spinner("Initializing system..."):
                (
                    ingestor,
                    splitter,
                    vector_store_manager,
                    retriever_factory,
                    chain_factory,
                ) = get_components(settings)

                if initialize_vector_store(
                    vector_store_manager, ingestor, splitter, force_recreate
                ):
                    llm = ChatOllama(model=settings.LLM_MODEL)
                    vector_retriever = vector_store_manager.get_retriever()
                    retriever = retriever_factory.create_retriever(
                        vector_retriever, llm
                    )
                    chain = chain_factory.create_rag_chain(
                        retriever, llm, mode=chain_mode
                    )

                    st.session_state.ingestor = ingestor
                    st.session_state.splitter = splitter
                    st.session_state.vector_store_manager = vector_store_manager
                    st.session_state.retriever_factory = retriever_factory
                    st.session_state.chain_factory = chain_factory
                    st.session_state.llm = llm
                    st.session_state.chain = chain
                    st.session_state.initialized = True
                    st.session_state.vector_store_loaded = True

                    st.success("System initialized successfully!")
                else:
                    st.error("Failed to initialize system.")

        st.divider()

        st.header("System Status")

        if st.session_state.initialized:
            st.success("✓ System initialized")
            if st.session_state.vector_store_loaded:
                st.success("✓ Vector store loaded")
        else:
            st.warning("⚠ System not initialized")

    if not st.session_state.initialized:
        st.info("👈 Configure settings and initialize the system to get started.")
        return

    if not st.session_state.vector_store_loaded:
        st.warning("Vector store not loaded. Please initialize the system.")
        return

    user_input = st.chat_input("Enter your question:")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Generating response..."):
            try:
                chain = st.session_state.chain
                response = chain.invoke(user_input)

                with st.chat_message("assistant"):
                    st.write(response)

                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
