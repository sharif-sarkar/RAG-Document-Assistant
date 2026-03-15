# RAG Document Assistant - Architecture Documentation

This document provides detailed information about project structure, modules, and architecture of the RAG Document Assistant.

## Project Overview

A modular RAG-powered document assistant for AI-powered question answering. Supports CLI and Streamlit interfaces with Ollama, LangChain, and FAISS vector search.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Entry Point                     │
│                           main.py                          │
│              (Mode Switching: CLI/Streamlit)                 │
└─────────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Configuration Layer                      │
│                    src/config/settings.py                    │
│         (Environment Variables, Runtime Overrides)              │
└─────────────────────────┬─────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐   ┌──────────┐
    │ Ingestion│    │Processing│   │Vector    │
    │          │    │          │   │Store     │
    │loader.py │    │splitter  │   │vector_   │
    └──────────┘    │.py      │   │store.py  │
                    └──────────┘   └──────────┘
                          │               │
                          └───────┬───────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │  Retrieval   │
                          │             │
                          │retriever.py │
                          │rag_chain.py │
                          └──────┬──────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
           ┌──────────┐    ┌──────────┐   ┌──────────┐
           │   CLI    │    │Streamlit │   │  Future  │
           │          │    │   App    │   │Interfaces│
           │  cli.py  │    │          │   │          │
           └──────────┘    │streamlit │   └──────────┘
                           │_app.py  │
                           └──────────┘
```

## Module Descriptions

### Core Modules

#### 1. Configuration Module (`src/config/settings.py`)

**Purpose**: Centralized configuration management with environment variable support.

**Key Classes**:

- `Settings`: Pydantic model for application settings

**Responsibilities**:

- Load configuration from environment variables
- Provide default values for all settings
- Support runtime overrides
- Handle path resolution for input/output directories

**Key Settings**:

- Directory paths (input/output)
- Model configuration (LLM, embeddings)
- Document processing parameters (chunk size, overlap)
- Retrieval settings (mode, number of documents)

#### 2. Document Ingestion Module (`src/ingestion/loader.py`)

**Purpose**: Load documents from various file formats.

**Key Classes**:

- `DocumentIngestor`: Handles document loading operations

**Responsibilities**:

- Load PDF files using PyPDFLoader
- Load text and markdown files using TextLoader
- Support batch loading from directories
- Handle file extension validation
- Provide logging for loading operations

**Methods**:

- `load_single_file()`: Load a single document
- `load_from_directory()`: Load all supported files from directory
- `load_documents()`: Unified loading interface

#### 3. Document Processing Module (`src/processing/splitter.py`)

**Purpose**: Split documents into manageable chunks for embedding.

**Key Classes**:

- `DocumentSplitter`: Handles text splitting operations

**Responsibilities**:

- Use RecursiveCharacterTextSplitter for intelligent chunking
- Support configurable chunk size and overlap
- Split both document objects and raw text
- Allow runtime parameter updates

**Methods**:

- `split_documents()`: Split LangChain Document objects
- `split_text()`: Split raw text strings
- `update_splitter_params()`: Update chunking parameters

#### 4. Vector Store Module (`src/vector_store/vector_store.py`)

**Purpose**: Manage vector database operations (FAISS).

**Key Classes**:

- `VectorStoreManager`: Manages FAISS vector store lifecycle

**Responsibilities**:

- Create vector stores from documents
- Save vector stores to disk
- Load existing vector stores
- Provide retriever instances
- Support embedding model updates

**Methods**:

- `create_vector_store()`: Create new vector store
- `load_vector_store()`: Load existing vector store
- `save_vector_store()`: Save to disk
- `get_or_create_vector_store()`: Smart load/create
- `get_retriever()`: Get retriever instance
- `delete_vector_store()`: Remove stored vector database

#### 5. Retrieval Module (`src/retrieval/retriever.py`)

**Purpose**: Create and configure document retrievers.

**Key Classes**:

- `RetrieverFactory`: Factory for creating retriever instances

**Responsibilities**:

- Create basic vector store retrievers
- Create multi-query retrievers with question expansion
- Support runtime mode switching
- Provide available retrieval modes

**Retrieval Strategies**:

- `basic`: Direct similarity search
- `multi_query`: Generate multiple question variants for better retrieval

**Methods**:

- `create_retriever()`: Factory method for retriever creation
- `_create_basic_retriever()`: Basic retriever implementation
- `_create_multi_query_retriever()`: Multi-query retriever implementation
- `update_retrieval_mode()`: Change retrieval strategy
- `get_available_modes()`: List available strategies

#### 6. RAG Chain Module (`src/retrieval/rag_chain.py`)

**Purpose**: Create RAG chains for question answering.

**Key Classes**:

- `RAGChainFactory`: Factory for creating RAG chains

**Responsibilities**:

- Create basic RAG chains
- Create conversational RAG chains with chat history
- Support custom prompt templates
- Provide different chain modes

**Chain Modes**:

- `basic`: Simple question-answer without history
- `conversational`: Question-answer with chat history support

**Methods**:

- `create_basic_rag_chain()`: Create basic RAG chain
- `create_conversational_rag_chain()`: Create conversational chain
- `create_rag_chain()`: Unified chain factory method
- `update_template()`: Change prompt templates

### Interface Modules

#### 7. CLI Interface (`src/interfaces/cli.py`)

**Purpose**: Command-line interface for the application.

**Key Classes**:

- `CLIInterface`: Manages CLI operations

**Responsibilities**:

- Parse command-line arguments
- Initialize system components
- Provide interactive query mode
- Support single query mode
- Handle vector store management

**Modes**:

- `interactive`: Continuous questioning session
- `query`: Single question and exit

**Command-line Arguments**:

- `--cli-mode`: Operation mode (interactive/query)
- `--question`: Question text (for query mode)
- `--input-dir`: Custom input directory
- `--output-dir`: Custom output directory
- `--llm-model`: Custom LLM model
- `--embedding-model`: Custom embedding model
- `--retrieval-mode`: Retrieval strategy
- `--chain-mode`: Chain type
- `--force-recreate`: Rebuild vector store
- `--log-level`: Logging verbosity

#### 8. Streamlit Interface (`src/interfaces/streamlit_app.py`)

**Purpose**: Web-based interface using Streamlit.

**Key Components**:

- Sidebar configuration panel
- Chat interface for querying
- System status display

**Responsibilities**:

- Provide intuitive web UI
- Support runtime configuration changes
- Display chat history
- Show system initialization status
- Cache components for performance

**Features**:

- Dynamic settings adjustment
- Vector store initialization
- Real-time query processing
- Chat history management

## Design Patterns

### 1. Factory Pattern

Used in `RetrieverFactory` and `RAGChainFactory` for creating different types of retrievers and chains.

### 2. Dependency Injection

Components receive `Settings` instances for configuration, allowing easy testing and configuration changes.

### 3. Strategy Pattern

Different retrieval strategies (basic, multi-query) can be swapped at runtime.

### 4. Singleton/Caching

Streamlit uses `@st.cache_resource` for expensive component initialization.

## Data Flow

### Document Processing Flow

```
Input Files → DocumentIngestor → DocumentSplitter → VectorStoreManager → Saved Vector Store
```

### Query Processing Flow

```
User Question → Interface → RetrieverFactory → VectorStore → RAGChain → Response
```

## Configuration Management

### Environment Variables

All settings can be overridden using environment variables (see README.md).

### Runtime Overrides

Both CLI and Streamlit interfaces allow runtime parameter overrides.

## Security Considerations

1. **No Hardcoded Secrets**: All credentials externalized
2. **Input Validation**: File paths validated before processing
3. **Error Handling**: Comprehensive error handling and logging
4. **Safe Defaults**: Conservative default configurations

## Future Extensions

1. Additional document loaders (DOCX, images with OCR)
2. More vector store backends (Chroma, Pinecone)
3. Additional retrieval strategies (Hybrid, Re-ranking)
4. Advanced RAG techniques (Citations, Sources)
5. Authentication and multi-tenancy support
6. API server mode (FastAPI/Flask)

## Development Guidelines

### Adding New Document Loaders

1. Add loader to `DocumentIngestor.load_single_file()`
2. Update `SUPPORTED_EXTENSIONS` in settings
3. Test with sample files

### Adding New Retrieval Strategies

1. Create method in `RetrieverFactory`
2. Update `create_retriever()` to handle new mode
3. Update `get_available_modes()`

### Adding New Chain Types

1. Create method in `RAGChainFactory`
2. Update `create_rag_chain()` to handle new mode
3. Add corresponding CLI/Streamlit options

## Testing Strategy

### Unit Tests

- Test individual module methods
- Mock external dependencies (Ollama, FAISS)

### Integration Tests

- Test document ingestion through vector creation
- Test query processing flow

### End-to-End Tests

- Test CLI with sample documents
- Test Streamlit with sample queries

## Performance Considerations

1. **Vector Store Caching**: Load once, reuse multiple times
2. **Chunk Size Tuning**: Balance context window vs. retrieval precision
3. **Retrieval Optimization**: Multi-query improves but costs more tokens
4. **Model Selection**: Smaller models for faster inference

## Logging

All modules use Python's logging module with configurable levels:

- `DEBUG`: Detailed information for debugging
- `INFO`: General information about operations
- `WARNING`: Warning messages for non-critical issues
- `ERROR`: Error messages for critical failures
