# RAG Pipeline with LangChain, FastAPI, and Streamlit 

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, FastAPI, and Streamlit. It provides a user-friendly interface for document ingestion and querying, supporting multiple document formats and different vector store options, with added user authentication.

Documents sourced from https://staffsquared.com/free-hr-documents/

## Features

- User authentication with FastAPI backend
- Document ingestion supporting multiple file uploads (txt, pdf, docx)
- Querying interface for asking questions based on ingested documents
- Support for OpenAI and Amazon Bedrock as LLM and embedding providers
- Integration with Chroma and Amazon Bedrock Knowledge Base as vector stores
- User-friendly Streamlit interface for both ingestion and querying

## Prerequisites

- Python 3.9 or higher
- Poetry for dependency management

## Installation

1. Clone this repository:
   ```
   git clone git@github.com:octure-ai/hr_rag_demo.git
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Create a `.env` file in the project root with the following content:
   ```
   SECRET_KEY=your_secret_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Activate the Poetry environment:
   ```
   poetry shell
   ```

2. Run the FastAPI backend:
   ```
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

3. In a new terminal, run the Streamlit frontend:
   ```
   poetry run streamlit run streamlit_app.py
   ```

4. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

5. In the Streamlit interface:
   - Register a new user account or log in with existing credentials.
   - Configure the provider (OpenAI or Bedrock) and vector store type (Chroma or Bedrock Knowledge Base) in the sidebar.
   - For OpenAI, enter your API key if not set in the .env file.
   - Upload documents for ingestion.
   - Use the query interface to ask questions based on the ingested documents.

## Configuration

- For OpenAI: Enter your API key in the .env file or in the Streamlit sidebar.
- For Amazon Bedrock: Ensure your AWS credentials are set up in the AWS CLI or as environment variables.
- For Bedrock Knowledge Base: You'll need to handle document ingestion through AWS tools, but the app provides a querying interface.

## Project Structure

- `api.py`: FastAPI backend for user authentication and management
- `models.py`: Database models and shared utilities
- `streamlit_app.py`: Streamlit frontend for document processing and querying
- `pyproject.toml`: Poetry configuration file with project dependencies

## Notes

- The app uses a local SQLite database (`sql_app.db`) for user management.
- The app uses a local directory named "vector_store" to persist the Chroma vector store. Ensure this directory is writable.
- For production use, consider implementing proper security measures, especially for API key handling and user authentication.
- The current setup uses SQLite, which is not ideal for high-concurrency scenarios. For production with multiple users, consider switching to a more robust database like PostgreSQL.
