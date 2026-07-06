"""
kg/build_graph.py

Reads the live SQLite schema via SchemaRegistry, generates LLM descriptions
for every table and column, embeds them, and writes:
  - Table nodes          (with description + embedding)
  - Column nodes         (with description + embedding + metadata)
  - :BELONGS_TO edges    (Column → Table)
  - :REFERENCES edges    (FK Column → referenced Column)
  - table_embeddings     vector index  (for Stage-0 schema scoping)
  - column_embeddings    vector index  (for Stage-2 KG retrieval)

SCALABILITY CHANGES vs original
--------------------------------
  BEFORE                              AFTER
  ─────────────────────────────────   ────────────────────────────────────────
  SCHEMA_META hardcoded dict (4 tbls) SchemaRegistry.load(cfg) — any # tables
  FK map hardcoded per table          FKEdge list from registry._fk_edges
  DESCRIPTION_SYSTEM hardcoded prompt Dynamic prompt — table count in header
  Table nodes had no embeddings       Table nodes get embeddings (Stage-0 index)

Run once (or on schema change):
    python kg/build_graph.py [path/to/db]
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

from config import load_config
from kg.schema_registry import SchemaRegistry
from kg.schema_retriever import build_table_embeddings
from utils.llm_client import (
    get_gpt_client, get_embedding_client,
    chat_complete, embed_batch,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── LLM prompt for column descriptions ───────────────────────────────────────

def _make_description_system(table_names: List[str]) -> str:
    """
    Build the system prompt for column description generation.
    Mentions all table names so the model can flag cross-table disambiguation.
    """
    tables_str = ", ".join(table_names)
    return (
        "You are a database data dictionary expert.\n"
        f"The database has the following tables: {tables_str}.\n\n"
        "Write a 2-4 sentence description for the database column below.\n"
        "Include:\n"
        "1. What it stores in plain business English\n"
        "2. Valid values and their meanings (if provided)\n"
        "3. Which business questions it helps answer\n"
        "4. Whether any other columns in other tables share a similar name but "
        "different meaning — explicitly call this out to prevent disambiguation errors.\n\n"
        "Be precise. Avoid generic filler. Output only the description text, no preamble."
    )


def _make_table_description_system() -> str:
    return (
        "You are a database data dictionary expert.\n"
        "Write a 2-3 sentence description for the database table below.\n"
        "Include: what business entity/process the table represents, "
        "how many rows typically exist, and which other tables it joins to.\n"
        "Be precise. Output only the description text, no preamble."
    )


# ── Column description generator ─────────────────────────────────────────────

def generate_column_description(
    gpt_client, deployment: str, description_system: str,
    table: str, col_meta,
) -> str:
    """
    Generate a rich LLM description for one column.
    col_meta is a ColumnMeta from the registry.
    """
    user_prompt = (
        f"Column: {col_meta.name} | Table: {table} | Type: {col_meta.data_type}\n"
        f"Is PK: {col_meta.is_pk} | Is FK: {col_meta.is_fk}\n"
        f"References: {col_meta.references or 'N/A'}\n"
        f"Sample values: {', '.join(col_meta.sample_values) if col_meta.sample_values else 'N/A'}\n"
        f"Null %: {col_meta.null_pct}"
    )
    return chat_complete(gpt_client, deployment, description_system, user_prompt)


def generate_table_description(
    gpt_client, deployment: str, table_name: str,
    table_meta, fk_targets: List[str],
) -> str:
    """Generate a rich LLM description for one table."""
    col_names = list(table_meta.columns.keys())
    user_prompt = (
        f"Table: {table_name}\n"
        f"Columns ({len(col_names)}): {', '.join(col_names)}\n"
        f"FK references to: {', '.join(fk_targets) if fk_targets else 'None'}\n"
        f"Current description: {table_meta.description}"
    )
    return chat_complete(gpt_client, deployment, _make_table_description_system(), user_prompt)


# ── Main graph builder ────────────────────────────────────────────────────────

def build_graph(db_path: str) -> None:
    cfg = load_config()
    gpt_client = get_gpt_client(cfg)
    emb_client = get_embedding_client(cfg)
    driver = GraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password)
    )

    # ── Step 0: Load full registry from SQLite ────────────────────────────────
    logger.info("Loading schema registry from SQLite: %s", db_path)
    registry = SchemaRegistry.load(cfg)   # internally loads from SQLite + derives rules
    table_names = registry.table_names
    logger.info("Registry loaded: %d tables, %d FK edges", len(table_names), len(registry.fk_edges))

    description_system = _make_description_system(table_names)

    with driver.session(database=cfg.neo4j_database) as session:
        # Clear existing graph (dev convenience; remove this line in production
        # if you want incremental updates)
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing graph.")

        # ── Step 1: Create Table nodes ────────────────────────────────────────
        for tname in table_names:
            tmeta = registry.tables[tname]
            # Find all tables this table's FK columns point to
            fk_targets = list({
                e.ref_table for e in registry.fk_edges if e.src_table == tname
            })
            # Generate a richer description via LLM (optional; skip if too many tables)
            try:
                rich_desc = generate_table_description(
                    gpt_client, cfg.openai_deployment_name, tname, tmeta, fk_targets
                )
            except Exception as e:
                logger.warning("Table description generation failed for %s: %s", tname, e)
                rich_desc = tmeta.description

            session.run(
                "MERGE (t:Table {name: $name}) SET t.description = $desc",
                name=tname, desc=rich_desc
            )
            # Update registry table description so embeddings use the richer text
            registry.tables[tname].description = rich_desc

        logger.info("Table nodes created (%d).", len(table_names))

        # ── Step 2: Column descriptions + Column nodes ────────────────────────
        all_col_records: List[Tuple[str, str, str]] = []  # (table, col_name, description)

        for tname in table_names:
            tmeta = registry.tables[tname]
            logger.info("Generating column descriptions for %s (%d cols)...",
                        tname, len(tmeta.columns))

            for cname, cmeta in tmeta.columns.items():
                # Generate description
                try:
                    desc = generate_column_description(
                        gpt_client, cfg.openai_deployment_name,
                        description_system, tname, cmeta
                    )
                except Exception as e:
                    logger.warning("Column description failed for %s.%s: %s", tname, cname, e)
                    desc = cmeta.description  # fall back to registry auto-description

                # Update registry so table embeddings pick up enriched text
                registry.tables[tname].columns[cname].description = desc

                session.run("""
                    MERGE (c:Column {name: $name, table: $table})
                    SET c.data_type     = $dtype,
                        c.description   = $desc,
                        c.is_pk         = $is_pk,
                        c.is_fk         = $is_fk,
                        c.sample_values = $samples,
                        c.null_pct      = $null_pct,
                        c.references    = $references
                """,
                    name=cname, table=tname,
                    dtype=cmeta.data_type, desc=desc,
                    is_pk=cmeta.is_pk, is_fk=cmeta.is_fk,
                    samples=json.dumps(cmeta.sample_values),
                    null_pct=cmeta.null_pct,
                    references=cmeta.references or "",
                )

                # Link Column → Table
                session.run("""
                    MATCH (t:Table {name: $table}), (c:Column {name: $col, table: $table})
                    MERGE (c)-[:BELONGS_TO]->(t)
                """, table=tname, col=cname)

                all_col_records.append((tname, cname, desc))

        logger.info("Column nodes created (%d total).", len(all_col_records))

        # ── Step 3: Embed all column descriptions in batch ────────────────────
        logger.info("Embedding %d column descriptions...", len(all_col_records))
        col_descriptions = [r[2] for r in all_col_records]
        col_vectors = embed_batch(emb_client, cfg.embedding_deployment, col_descriptions)

        for (tname, cname, _), vec in zip(all_col_records, col_vectors):
            session.run("""
                MATCH (c:Column {name: $col, table: $table})
                SET c.embedding = $vec
            """, col=cname, table=tname, vec=vec)
        logger.info("Column embeddings stored.")

        # ── Step 4: Column vector index ───────────────────────────────────────
        try:
            session.run("""
                CREATE VECTOR INDEX column_embeddings IF NOT EXISTS
                FOR (c:Column) ON (c.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 1536,
                                       `vector.similarity_function`: 'cosine'}}
            """)
            logger.info("column_embeddings vector index created/confirmed.")
        except Exception as e:
            logger.warning("Column vector index: %s", e)

        # ── Step 5: FK :REFERENCES edges from registry ────────────────────────
        for edge in registry.fk_edges:
            session.run("""
                MATCH (src:Column {name: $src_col, table: $src_table}),
                      (tgt:Column {name: $ref_col, table: $ref_table})
                MERGE (src)-[r:REFERENCES]->(tgt)
                SET r.join_condition    = $join_cond,
                    r.relationship_type = $cardinality,
                    r.description       = $rel_desc
            """,
                src_col=edge.src_col,   src_table=edge.src_table,
                ref_col=edge.ref_col,   ref_table=edge.ref_table,
                join_cond=edge.join_condition,
                cardinality=edge.cardinality,
                rel_desc=edge.description,
            )
        logger.info("FK :REFERENCES edges created (%d).", len(registry.fk_edges))

    # ── Step 6: Table-level embeddings + index (Stage-0 schema retrieval) ────
    logger.info("Building table-level embeddings for Stage-0 schema retrieval...")
    build_table_embeddings(cfg, registry)

    driver.close()
    logger.info("Knowledge Graph build complete.")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "database/insurance_demo.db"
    if not Path(db).exists():
        print(f"Database not found at {db}. Run: python database/seed_db.py first.")
        sys.exit(1)
    build_graph(db)