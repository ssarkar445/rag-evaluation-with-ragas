from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # Application
    app_name: str = "RAG Application"
    environment: str = "development"
    debug: bool = False

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # Vector Database
    chroma_persist_directory: str = "./chroma_db"
    collection_name: str = "finance-documents"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 75
    top_k: int = 5

    # Data Path
    corpus: Path = (
        BASE_DIR
        / "src"
        / "rag_evaluation_with_ragas"
        / "finance_rag_data"
        / "corpus"
    )

    golden: Path = (
        BASE_DIR
        / "src"
        / "rag_evaluation_with_ragas"
        / "finance_rag_data"
        / "golden"
    )

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()