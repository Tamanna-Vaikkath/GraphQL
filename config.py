"""
config.py — Central configuration loader.
Reads .env and exposes all credentials/settings as a typed dataclass.

DATABRICKS MIGRATION NOTE
-------------------------
This version adds Databricks support alongside the original SQLite path —
via a **SQL Warehouse** connection (databricks-sql-connector), NOT Spark /
Databricks Connect. Set DATA_BACKEND=databricks in your .env (or in the
Databricks App's environment) to point the SQL swim-lane at a Unity
Catalog catalog/schema through a running SQL Warehouse, instead of the
local insurance_demo.db file. The Cypher/Neo4j swim lane is unaffected
either way — it always talks to Neo4j AuraDB directly.

Set DATA_BACKEND=sqlite (or leave unset) to keep the original local-demo
behavior — this is what lets you keep developing/testing on your laptop
exactly as before while the Databricks side is being stood up.

DEV COPY NOTE
-------------
This is the DEV copy of the app. Its default SQLite file is
database/insurance_dev.db (the extended dev data), and APP_ENV defaults to
"dev". See DEV_SETUP_GUIDE.md and utils/safety.py.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Which SQL backend the pipeline talks to: "sqlite" or "databricks"
    data_backend: str

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

    # SQLite (legacy / local dev)
    sqlite_db_path: str

    # Databricks (Unity Catalog, accessed via a SQL Warehouse connection —
    # databricks-sql-connector, no Spark/Databricks Connect involved)
    databricks_catalog: str = ""
    databricks_schema: str = ""
    databricks_server_hostname: str = ""
    databricks_http_path: str = ""
    databricks_token: str = ""


def load_config() -> Config:
    """Load and validate all config from environment variables."""
    data_backend = os.getenv("DATA_BACKEND", "sqlite").strip().lower()

    required = {
        "OPENAI_DEPLOYMENT_ENDPOINT": os.getenv("OPENAI_DEPLOYMENT_ENDPOINT"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_EMBEDDING_API_KEY": os.getenv("OPENAI_EMBEDDING_API_KEY"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
    }

    if data_backend == "databricks":
        required.update({
            "DATABRICKS_CATALOG": os.getenv("DATABRICKS_CATALOG"),
            "DATABRICKS_SCHEMA": os.getenv("DATABRICKS_SCHEMA"),
            "DATABRICKS_SERVER_HOSTNAME": os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            "DATABRICKS_HTTP_PATH": os.getenv("DATABRICKS_HTTP_PATH"),
        })
        # DATABRICKS_TOKEN is intentionally NOT required: when running as a
        # Databricks App, auth is automatic via the app's own
        # service-principal credentials (see utils/db_connector.py).

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")

    return Config(
        data_backend=data_backend,
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
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", "database/gl_claims.db"),
        databricks_catalog=os.getenv("DATABRICKS_CATALOG", ""),
        databricks_schema=os.getenv("DATABRICKS_SCHEMA", ""),
        databricks_server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME", ""),
        databricks_http_path=os.getenv("DATABRICKS_HTTP_PATH", ""),
        databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
    )
