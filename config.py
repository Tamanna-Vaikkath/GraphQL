"""
config.py — Central configuration loader.
Reads .env and exposes all credentials/settings as a typed dataclass.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Azure OpenAI (GPT)
    openai_endpoint: str
    openai_api_key: str
    openai_api_version: str
    openai_deployment_name: str
    openai_model_name: str

    # Azure OpenAI Embeddings
    embedding_endpoint: str
    embedding_api_key: str
    embedding_model: str
    embedding_deployment: str

    # Neo4j
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str

    # SQLite
    sqlite_db_path: str


def load_config() -> Config:
    """Load and validate all config from environment variables."""
    required = {
        "OPENAI_DEPLOYMENT_ENDPOINT": os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_EMBEDDING_API_KEY": os.getenv("OPENAI_EMBEDDING_API_KEY"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")

    return Config(
        openai_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_version=os.getenv("OPENAI_API_VERSION", "2025-04-01-preview"),
        openai_deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
        openai_model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        embedding_endpoint=os.getenv(
            "OPENAI_EMBEDDING_ENDPOINT",
            "https://openai-rag-ai-search.openai.azure.com/"
        ),
        embedding_api_key=os.getenv("OPENAI_EMBEDDING_API_KEY"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_deployment=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", "database/insurance_demo.db"),
    )
