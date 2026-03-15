import argparse
import sys

from src.interfaces.cli import main as cli_main


def main():
    """
    Main entry point for the RAG application.
    Supports switching between CLI and Streamlit modes at runtime.
    """
    parser = argparse.ArgumentParser(
        description="A modular RAG-powered document assistant for AI-powered question answering. Supports CLI and Streamlit interfaces with Ollama, LangChain, and FAISS vector search."
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "streamlit"],
        default="cli",
        help="Interface mode: 'cli' for command-line, 'streamlit' for web UI",
    )

    args, remaining_args = parser.parse_known_args()

    if args.mode == "cli":
        sys.argv = ["cli"] + remaining_args
        cli_main()
    elif args.mode == "streamlit":
        import subprocess
        import os

        # Add current directory to PYTHONPATH so src can be imported
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        # Use os.getcwd() to ensure we get the absolute path of the project root
        env["PYTHONPATH"] = f"{os.getcwd()}{os.pathsep}{python_path}"

        subprocess.run(
            ["streamlit", "run", "src/interfaces/streamlit_app.py"] + remaining_args,
            env=env,
        )
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
