"""
kg/build_graph.py — Phase 2 of the implementation playbook.

Reads the SQLite schema, generates LLM descriptions for every column,
embeds them, and writes Table nodes, Column nodes, Embedding vectors,
and :REFERENCES edges into Neo4j AuraDB.

Run once (or on schema change):
    python kg/build_graph.py
"""
from __future__ import annotations
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

from config import load_config
from utils.llm_client import (
    get_gpt_client, get_embedding_client,
    chat_complete, embed_batch,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── P&C schema metadata (mirrors Section 2 of the concept paper) ──────────────
SCHEMA_META: Dict[str, Dict[str, Any]] = {
    "CLAIMS": {
        "description": (
            "Stores one record per insurance claim filed against an active policy. "
            "Central hub linking claimants, policies, and payments. Used for claim "
            "lifecycle tracking, adjuster assignment, and loss reporting."
        ),
        "fks": {
            "POLICY_ID":   ("POLICY",   "POLICY_ID",   "CLAIMS.POLICY_ID = POLICY.POLICY_ID",   "MANY_TO_ONE", "Many claims per policy"),
            "CLAIMANT_ID": ("CLAIMANT", "CLAIMANT_ID", "CLAIMS.CLAIMANT_ID = CLAIMANT.CLAIMANT_ID", "MANY_TO_ONE", "Many claims per claimant"),
        },
    },
    "POLICY": {
        "description": (
            "Stores one row per insurance policy issued. Defines coverage terms, "
            "premium amounts, effective dates, and line of business."
        ),
        "fks": {},
    },
    "PAYMENT": {
        "description": (
            "Stores individual payment transactions issued against a claim. "
            "One claim can have many payments across indemnity, medical, and expense types."
        ),
        "fks": {
            "CLAIM_ID": ("CLAIMS", "CLAIM_ID", "PAYMENT.CLAIM_ID = CLAIMS.CLAIM_ID", "MANY_TO_ONE", "One claim can have many payments"),
        },
    },
    "CLAIMANT": {
        "description": (
            "Stores profile information about individuals who have filed claims. "
            "Includes demographics, attorney representation flag, and fraud risk score."
        ),
        "fks": {},
    },
}

DESCRIPTION_SYSTEM = """You are a P&C insurance data dictionary expert.
Write a 2-4 sentence description for the database column below.
Include:
1. What it stores in plain business English
2. Valid values and their meanings (if provided)
3. Which business questions it helps answer
4. Whether any other columns in other tables share a similar name but different meaning — explicitly call this out to prevent disambiguation errors.

Be precise. Avoid generic filler. Output only the description text, no preamble."""


def get_sqlite_schema(db_path: str) -> Dict[str, List[Dict]]:
    """Extract column info and sample values from SQLite."""
    conn = sqlite3.connect(db_path)
    tables = {}
    for table in SCHEMA_META:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = []
        for row in cursor.fetchall():
            col_name = row[1]
            col_type = row[2]
            is_pk = bool(row[5])
            # fetch sample values
            try:
                samples = conn.execute(
                    f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 5"
                ).fetchall()
                sample_vals = [str(r[0]) for r in samples]
            except Exception:
                sample_vals = []
            # null pct
            try:
                total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                nulls = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL"
                ).fetchone()[0]
                null_pct = round(nulls / total * 100, 1) if total else 0.0
            except Exception:
                null_pct = 0.0

            is_fk = col_name in SCHEMA_META[table]["fks"]
            columns.append({
                "name": col_name,
                "type": col_type,
                "is_pk": is_pk,
                "is_fk": is_fk,
                "sample_values": sample_vals,
                "null_pct": null_pct,
            })
        tables[table] = columns
    conn.close()
    return tables


def generate_column_description(gpt_client, deployment: str, table: str, col: Dict) -> str:
    user_prompt = (
        f"Column: {col['name']} | Table: {table} | Type: {col['type']}\n"
        f"Is PK: {col['is_pk']} | Is FK: {col['is_fk']}\n"
        f"Sample values: {', '.join(col['sample_values']) if col['sample_values'] else 'N/A'}\n"
        f"Null %: {col['null_pct']}\n"
        f"Other tables in schema: {[t for t in SCHEMA_META if t != table]}"
    )
    return chat_complete(gpt_client, deployment, DESCRIPTION_SYSTEM, user_prompt)


def build_graph(db_path: str):
    cfg = load_config()
    gpt_client = get_gpt_client(cfg)
    emb_client = get_embedding_client(cfg)
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))

    logger.info("Extracting schema from SQLite...")
    schema = get_sqlite_schema(db_path)

    with driver.session(database=cfg.neo4j_database) as session:
        # Clear existing graph (dev convenience)
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing graph.")

        # ── Step 1: Create Table nodes ────────────────────────────────────────
        for table, meta in SCHEMA_META.items():
            session.run(
                "MERGE (t:Table {name: $name}) SET t.description = $desc",
                name=table, desc=meta["description"]
            )
        logger.info("Table nodes created.")

        # ── Step 2: Generate descriptions and create Column nodes ─────────────
        all_col_records: List[Tuple[str, str, str]] = []  # (table, col_name, description)
        for table, columns in schema.items():
            logger.info("Generating descriptions for %s (%d cols)...", table, len(columns))
            for col in columns:
                desc = generate_column_description(
                    gpt_client, cfg.openai_deployment_name, table, col
                )
                session.run("""
                    MERGE (c:Column {name: $name, table: $table})
                    SET c.data_type    = $dtype,
                        c.description  = $desc,
                        c.is_pk        = $is_pk,
                        c.is_fk        = $is_fk,
                        c.sample_values = $samples,
                        c.null_pct     = $null_pct
                """,
                    name=col["name"], table=table,
                    dtype=col["type"], desc=desc,
                    is_pk=col["is_pk"], is_fk=col["is_fk"],
                    samples=json.dumps(col["sample_values"]),
                    null_pct=col["null_pct"]
                )
                all_col_records.append((table, col["name"], desc))
                # Link Column → Table
                session.run("""
                    MATCH (t:Table {name: $table}), (c:Column {name: $col, table: $table})
                    MERGE (c)-[:BELONGS_TO]->(t)
                """, table=table, col=col["name"])

        logger.info("Column nodes created (%d total).", len(all_col_records))

        # ── Step 3: Embed all descriptions in one batch ───────────────────────
        logger.info("Embedding %d column descriptions...", len(all_col_records))
        descriptions = [r[2] for r in all_col_records]
        vectors = embed_batch(emb_client, cfg.embedding_deployment, descriptions)

        for (table, col_name, _), vector in zip(all_col_records, vectors):
            session.run("""
                MATCH (c:Column {name: $col, table: $table})
                SET c.embedding = $vec
            """, col=col_name, table=table, vec=vector)
        logger.info("Embeddings stored.")

        # ── Step 4: Create Vector Index ───────────────────────────────────────
        try:
            session.run("""
                CREATE VECTOR INDEX column_embeddings IF NOT EXISTS
                FOR (c:Column) ON (c.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 1536,
                                       `vector.similarity_function`: 'cosine'}}
            """)
            logger.info("Vector index created/confirmed.")
        except Exception as e:
            logger.warning("Vector index creation: %s", e)

        # ── Step 5: Create :REFERENCES edges from FK map ──────────────────────
        for table, meta in SCHEMA_META.items():
            for fk_col, (ref_table, ref_col, join_cond, cardinality, rel_desc) in meta["fks"].items():
                session.run("""
                    MATCH (src:Column {name: $fk_col, table: $src_table}),
                          (tgt:Column {name: $ref_col, table: $ref_table})
                    MERGE (src)-[r:REFERENCES]->(tgt)
                    SET r.join_condition     = $join_cond,
                        r.relationship_type  = $cardinality,
                        r.description        = $rel_desc
                """,
                    fk_col=fk_col, src_table=table,
                    ref_col=ref_col, ref_table=ref_table,
                    join_cond=join_cond, cardinality=cardinality,
                    rel_desc=rel_desc
                )
        logger.info("FK :REFERENCES edges created.")

    driver.close()
    logger.info("Knowledge Graph build complete.")


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "database/insurance_demo.db"
    if not Path(db).exists():
        print(f"Database not found at {db}. Run: python database/seed_db.py first.")
        sys.exit(1)
    build_graph(db)
