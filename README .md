# RAG Document Assistant
A modular RAG-powered document assistant for AI-powered question answering. Supports CLI and Streamlit interfaces with Ollama, LangChain, and FAISS vector search.

## Features

- **Multiple Interfaces**: Command-line and web-based (Streamlit) interfaces
- **Flexible Document Loading**: Supports PDF, TXT, and MD file formats
- **Configurable Retrieval**: Basic and multi-query retrieval strategies
- **Vector Store Management**: Automatic save/load of FAISS vector databases
- **Runtime Configuration**: Switch between modes and parameters at runtime
- **Modular Architecture**: Clean separation of concerns for easy maintenance

## Prerequisites

- Python 3.9 or higher
- Ollama with at least one model (e.g., `llama3.2:3b`)
- Ollama embedding model (e.g., `nomic-embed-text:v1.5`)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd rag_document_assistant
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install and start Ollama:

```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text:v1.5
```

4. Ensure Ollama is running:

```bash
ollama serve
```

## Project Structure

```
rag_document_assistant/
├── __init__.py            # Package initialization
├── main.py                # Main entry point with mode switching
├── requirements.txt        # Python dependencies
├── pytest.ini             # Pytest configuration
├── LICENSE                # MIT License
├── .gitignore             # Git ignore rules
├── .env.example           # Example environment variables
├── README.md              # This file
├── AGENTS.md              # Architecture documentation
├── TEST_REPORT.md         # Testing documentation
├── input_data/            # Input directory for documents
├── output_data/           # Output directory for vector stores
├── src/                   # Source code
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py    # Configuration management
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py      # Document loading utilities
│   ├── processing/
│   │   ├── __init__.py
│   │   └── splitter.py    # Text splitting
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── vector_store.py # Vector database management
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py   # Retriever factory
│   │   └── rag_chain.py  # RAG chain factory
│   └── interfaces/
│       ├── __init__.py
│       ├── cli.py         # CLI interface
│       └── streamlit_app.py # Streamlit web interface
└── tests/                 # Test suite
    ├── __init__.py
    ├── conftest.py        # Shared fixtures and test utilities
    ├── fixtures/          # Test fixtures directory
    ├── unit/              # Unit tests
    │   ├── __init__.py
    │   ├── test_config.py
    │   ├── test_ingestion.py
    │   ├── test_processing.py
    │   ├── test_vector_store.py
    │   └── test_retrieval.py
    └── integration/       # Integration tests
        ├── __init__.py
        └── test_basic_integration.py
```

## Usage

### CLI Mode

Run the application in CLI mode (default):

```bash
python main.py --mode cli
```

Interactive mode (default):

````bash
```bash
python main.py --mode cli --cli-mode interactive
````

Single query mode:

```bash
python main.py --mode cli --cli-mode query --question "Your question here"
```

CLI Options:

- `--cli-mode {interactive,query}`: Operation mode
- `--question TEXT`: Question to ask (for query mode)
- `--input-dir PATH`: Input directory for documents
- `--output-dir PATH`: Output directory for vector store
- `--llm-model MODEL`: LLM model name (default: llama3.2:3b)
- `--embedding-model MODEL`: Embedding model name (default: nomic-embed-text:v1.5)
- `--retrieval-mode {basic,multi_query}`: Retrieval strategy
- `--chain-mode {basic,conversational}`: Chain type
- `--force-recreate`: Force recreation of vector store
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level

### Streamlit Mode

Run the application in Streamlit mode:

```bash
python main.py --mode streamlit
```

This will launch a web interface at `http://localhost:8501`

## Configuration

### Environment Variables

Create a `.env` file in the project root to override defaults:

```env
INPUT_DATA_DIR=input_data
OUTPUT_DATA_DIR=output_data
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text:v1.5
OLLAMA_BASE_URL=http://localhost:11434
RETRIEVAL_MODE=multi_query
CHUNK_SIZE=1200
CHUNK_OVERLAP=300
LOG_LEVEL=INFO
```

### Runtime Configuration

Both CLI and Streamlit modes allow runtime configuration:

- **Input/Output Directories**: Specify custom paths for documents and vector stores
- **Model Selection**: Choose different Ollama models for LLM and embeddings
- **Retrieval Mode**: Switch between basic and multi-query retrieval
- **Chain Mode**: Choose between basic and conversational RAG chains
- **Chunk Settings**: Adjust chunk size and overlap

## Development

### Environment Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running Tests

The project includes a comprehensive test suite with 78 tests covering all core modules.

**Run all tests:**

```bash
pytest tests/ -v
```

**Run unit tests only:**

```bash
pytest tests/unit/ -v
```

**Run integration tests:**

```bash
pytest tests/integration/ -v
```

**Run with coverage:**

```bash
pytest tests/ --cov=src --cov-report=html
# View coverage: open htmlcov/index.html
```

- Comprehensive unit and integration tests included
- Code coverage reporting available

See [TEST_REPORT.md](TEST_REPORT.md) for detailed test documentation.

### Linting

Format code:

```bash
black src/ tests/
```

Lint code:

```bash
ruff check src/ tests/
```

Type checking:

```bash
mypy src/
```

## Architecture

See `AGENTS.md` for detailed architecture documentation including module descriptions and design patterns.

## Security

This application is designed to be safe for public repositories:

- No hardcoded secrets or credentials
- All sensitive data is externalized to environment variables
- `.gitignore` excludes vector stores, logs, and configuration files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Ollama connection issues

- Ensure Ollama is running: `ollama serve`
- Check the base URL in settings (default: http://localhost:11434)

### Vector store errors

- Delete the vector store and recreate: `--force-recreate`
- Check disk space availability

### Memory issues

- Reduce chunk size in settings
- Use smaller models (e.g., `llama3.2:1b` instead of `llama3.2:3b`)
