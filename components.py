import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from big_query_manager import BigQueryManager

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
    PROJECT_ID = os.getenv("PROJECT_ID")
    DATASET_ID = os.getenv("DATASET_ID")
    bq_manager = BigQueryManager(project_id=PROJECT_ID, dataset_id=DATASET_ID)

    # Gemini API Key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GEMINI_API_KEY
    )

    # Initialize vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
        task_type="retrieval_document"
    )
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    return llm, vector_store, bq_manager