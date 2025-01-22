"""
Module for initializing core AI components.

This module sets up and initializes key components required for the application, 
including a generative AI model, a vector database, and a BigQuery manager.

Components Initialized:
    - ChatGoogleGenerativeAI: Provides interaction with Google's Gemini model.
    - Chroma: Vector store for document embedding and retrieval.
    - BigQueryManager: Handles interaction with Google BigQuery.

Environment Variables:
    - PROJECT_ID: The Google Cloud project ID.
    - DATASET_ID: The BigQuery dataset ID.
    - GEMINI_API_KEY: API key for accessing the Gemini AI services.

Functions:
    - initialize_components(): Initializes and returns the LLM, vector store, and BigQuery manager.

Example Usage:
    ```python
    from module_name import initialize_components
    llm, vector_store, bq_manager = await initialize_components()
    ```
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.big_query_manager import BigQueryManager


async def initialize_components():
    """
    Initializes the necessary components for the application.
    Returns:
        llm: ChatGoogleGenerativeAI instance.
        vector_store: Chroma vector store instance.
        bq_manager: BigQueryManager instance.
    """
    # Load environment variables
    load_dotenv()

    # BigQuery configuration
    project_id = os.getenv("PROJECT_ID")
    dataset_id = os.getenv("DATASET_ID")
    bq_manager = BigQueryManager(project_id=project_id, dataset_id=dataset_id)

    # Gemini API Key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=gemini_api_key)

    # Initialize vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key,
        task_type="retrieval_document",
    )
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    return llm, vector_store, bq_manager
