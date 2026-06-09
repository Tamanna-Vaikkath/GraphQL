"""
kg/retriever.py — Stage 2 & 3 of the query pipeline.

Stage 2: Vector similarity search → top-50 Column nodes → LLM re-rank to top-25.
Stage 3: Cypher graph traversal → FK join paths for identified columns.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from neo4j import GraphDatabase

from config import Config
from utils.llm_client import get_gpt_client, get_embedding_client, embed_text, chat_complete
from pipeline.hyde_expander import SCHEMA_MANIFEST

logger = logging.getLogger(__name__)

RERANK_SYSTEM = """You are a P&C insurance database expert.
Given a user question and a list of database column descriptions,
return ONLY a JSON array of the 25 most relevant column names (as "TABLE.COLUMN" strings).
Output only the JSON array, no explanation."""


@dataclass
class RetrievedColumn:
    name: str
    table: str
    description: str
    sample_values: List[str]
    is_pk: bool
    is_fk: bool
    score: float


@dataclass
class RetrievalResult:
    columns: List[RetrievedColumn] = field(default_factory=list)
    join_conditions: List[str] = field(default_factory=list)
    tables_involved: List[str] = field(default_factory=list)
    max_score: float = 0.0
    expanded_query: str = ""


class KGRetriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)
        self.emb = get_embedding_client(cfg)
        self.driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def retrieve(self, expanded_query: str, top_k: int = 50) -> RetrievalResult:
        """Full Stage 2+3 retrieval for a given (already-expanded) query."""
        # 1. Embed the expanded query
        query_vec = embed_text(self.emb, self.cfg.embedding_deployment, expanded_query)

        # 2. Vector similarity search → top_k candidates
        candidates = self._vector_search(query_vec, top_k)
        if not candidates:
            return RetrievalResult(expanded_query=expanded_query)

        # 2a. Schema guard — drop any column node that is not in the ground-truth
        #     manifest. This prevents stale/corrupt Neo4j nodes from flowing
        #     downstream into SQL generation as hallucinated column references.
        candidates = [
            c for c in candidates
            if c.table in SCHEMA_MANIFEST
            and c.name in SCHEMA_MANIFEST[c.table]
        ]
        if not candidates:
            return RetrievalResult(expanded_query=expanded_query)

        max_score = candidates[0].score if candidates else 0.0

        # 3. LLM re-rank to top 25
        top_25 = self._llm_rerank(expanded_query, candidates)

        # 4. Graph traversal for join paths
        col_names = [f"{c.name}" for c in top_25]
        table_names = list({c.table for c in top_25})
        join_conditions = self._get_join_paths(col_names, table_names)

        return RetrievalResult(
            columns=top_25,
            join_conditions=join_conditions,
            tables_involved=table_names,
            max_score=max_score,
            expanded_query=expanded_query,
        )

    def _vector_search(self, query_vec: List[float], top_k: int) -> List[RetrievedColumn]:
        with self.driver.session(database=self.cfg.neo4j_database) as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('column_embeddings', $k, $vec)
                YIELD node, score
                WHERE score > 0.45
                RETURN node.name        AS name,
                       node.table       AS table,
                       node.description AS description,
                       node.sample_values AS sample_values,
                       node.is_pk       AS is_pk,
                       node.is_fk       AS is_fk,
                       score
                ORDER BY score DESC
            """, k=top_k, vec=query_vec)

            columns = []
            for rec in result:
                try:
                    samples = json.loads(rec["sample_values"] or "[]")
                except Exception:
                    samples = []
                columns.append(RetrievedColumn(
                    name=rec["name"],
                    table=rec["table"],
                    description=rec["description"] or "",
                    sample_values=samples,
                    is_pk=bool(rec["is_pk"]),
                    is_fk=bool(rec["is_fk"]),
                    score=rec["score"],
                ))
            return columns

    def _llm_rerank(self, question: str, candidates: List[RetrievedColumn]) -> List[RetrievedColumn]:
        """Ask GPT to select the 25 most relevant columns from the candidates."""
        candidate_list = "\n".join(
            f"{c.table}.{c.name} (score={c.score:.3f}): {c.description[:120]}"
            for c in candidates
        )
        user_prompt = (
            f"User question: {question}\n\n"
            f"Candidate columns:\n{candidate_list}\n\n"
            "Return a JSON array of the 25 most relevant column identifiers "
            "as \"TABLE.COLUMN\" strings."
        )
        try:
            raw = chat_complete(
                self.gpt, self.cfg.openai_deployment_name,
                RERANK_SYSTEM, user_prompt, temperature=0.0, max_tokens=800
            )
            # Extract JSON array
            start = raw.find("[")
            end = raw.rfind("]") + 1
            selected = json.loads(raw[start:end])  # e.g. ["CLAIMS.CLM_STAT_CD", ...]
        except Exception as e:
            logger.warning("Re-rank parse failed (%s), using top-25 by score.", e)
            return candidates[:25]

        selected_set = set(selected)
        col_map = {f"{c.table}.{c.name}": c for c in candidates}
        reranked = [col_map[s] for s in selected if s in col_map]
        # Fill up to 25 if LLM returned fewer
        for c in candidates:
            if len(reranked) >= 25:
                break
            if f"{c.table}.{c.name}" not in selected_set:
                reranked.append(c)
        return reranked[:25]

    def _get_join_paths(self, col_names: List[str], table_names: List[str]) -> List[str]:
        """Stage 3: traverse FK :REFERENCES edges to find join conditions."""
        with self.driver.session(database=self.cfg.neo4j_database) as session:
            result = session.run("""
                MATCH path = (c1:Column)-[:REFERENCES*1..3]->(c2:Column)
                WHERE c1.table IN $tables OR c2.table IN $tables
                RETURN DISTINCT [rel IN relationships(path) | rel.join_condition]
                       AS join_conditions
            """, tables=table_names)

            joins = set()
            for rec in result:
                for jc in (rec["join_conditions"] or []):
                    if jc:
                        joins.add(jc)
            return list(joins)