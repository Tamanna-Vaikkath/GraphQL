"""
kg/retriever.py — Stage 2 & 3 of the query pipeline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from neo4j import GraphDatabase

from config import Config
from kg.schema_registry import SchemaRegistry
from kg.schema_retriever import ScopedSchema
from utils.llm_client import get_gpt_client, get_embedding_client, embed_text, chat_complete

logger = logging.getLogger(__name__)

RERANK_SYSTEM = """You are a database expert.
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
    scoped_tables: List[str] = field(default_factory=list)


class KGRetriever:
    def __init__(self, cfg: Config, registry: SchemaRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.gpt = get_gpt_client(cfg)
        self.emb = get_embedding_client(cfg)
        self.driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password)
        )
        print(f"[KGRetriever INIT] Connected to Neo4j at {cfg.neo4j_uri}")

    def close(self) -> None:
        self.driver.close()

    def retrieve_scoped(
        self,
        expanded_query: str,
        scoped_schema: ScopedSchema,
        top_k: int = 50,
    ) -> RetrievalResult:
        print(f"\n[Stage 2+3 INPUT] KGRetriever.retrieve_scoped()")
        print(f"  expanded_query  : {repr(expanded_query)}")
        print(f"  scoped tables   : {scoped_schema.relevant_tables}")
        print(f"  top_k           : {top_k}")

        scoped_tables = set(scoped_schema.relevant_tables)
        scoped_manifest = scoped_schema.scoped_manifest

        query_vec = embed_text(self.emb, self.cfg.embedding_deployment, expanded_query)
        print(f"[Stage 2] Query embedded → dim={len(query_vec)}")

        candidates = self._vector_search(query_vec, top_k, restrict_to_tables=list(scoped_tables))
        print(f"[Stage 2] Vector search → {len(candidates)} candidates before schema guard")

        if not candidates:
            print("[Stage 2] No candidates found — returning empty result")
            return RetrievalResult(expanded_query=expanded_query, scoped_tables=list(scoped_tables))

        candidates = [
            c for c in candidates
            if c.table in scoped_manifest and c.name in scoped_manifest[c.table]
        ]
        print(f"[Stage 2] After schema guard → {len(candidates)} candidates")

        if not candidates:
            return RetrievalResult(expanded_query=expanded_query, scoped_tables=list(scoped_tables))

        max_score = candidates[0].score
        print(f"[Stage 2] Max score: {max_score:.4f}")

        top_25 = self._llm_rerank(expanded_query, candidates)
        print(f"[Stage 2] After LLM rerank → {len(top_25)} columns selected")
        for c in top_25[:5]:
            print(f"  {c.table}.{c.name} score={c.score:.3f}")

        table_names = list({c.table for c in top_25})
        join_conditions = self._get_join_paths(
            col_names=[c.name for c in top_25],
            table_names=table_names,
            restrict_to_tables=list(scoped_tables),
        )
        print(f"[Stage 3] Join conditions found: {join_conditions}")

        return RetrievalResult(
            columns=top_25,
            join_conditions=join_conditions,
            tables_involved=table_names,
            max_score=max_score,
            expanded_query=expanded_query,
            scoped_tables=list(scoped_tables),
        )

    def retrieve(self, expanded_query: str, top_k: int = 50) -> RetrievalResult:
        print(f"\n[Stage 2+3 INPUT] KGRetriever.retrieve() [LEGACY full-schema]")
        print(f"  expanded_query: {repr(expanded_query)}")

        query_vec = embed_text(self.emb, self.cfg.embedding_deployment, expanded_query)
        candidates = self._vector_search(query_vec, top_k, restrict_to_tables=None)
        print(f"[Stage 2] {len(candidates)} candidates from full-schema search")
        if not candidates:
            return RetrievalResult(expanded_query=expanded_query)

        full_manifest = self.registry.manifest
        candidates = [
            c for c in candidates
            if c.table in full_manifest and c.name in full_manifest[c.table]
        ]
        if not candidates:
            return RetrievalResult(expanded_query=expanded_query)

        max_score = candidates[0].score
        top_25 = self._llm_rerank(expanded_query, candidates)
        table_names = list({c.table for c in top_25})
        join_conditions = self._get_join_paths(
            col_names=[c.name for c in top_25],
            table_names=table_names,
            restrict_to_tables=None,
        )
        return RetrievalResult(
            columns=top_25,
            join_conditions=join_conditions,
            tables_involved=table_names,
            max_score=max_score,
            expanded_query=expanded_query,
        )

    def _vector_search(
        self,
        query_vec: List[float],
        top_k: int,
        restrict_to_tables: Optional[List[str]],
    ) -> List[RetrievedColumn]:
        print(f"  [_vector_search] top_k={top_k}, scoped_tables={restrict_to_tables}")
        with self.driver.session(database=self.cfg.neo4j_database) as session:
            if restrict_to_tables:
                result = session.run("""
                    CALL db.index.vector.queryNodes('column_embeddings', $k, $vec)
                    YIELD node, score
                    WHERE score > 0.40
                      AND node.table IN $tables
                    RETURN node.name        AS name,
                           node.table       AS table,
                           node.description AS description,
                           node.sample_values AS sample_values,
                           node.is_pk       AS is_pk,
                           node.is_fk       AS is_fk,
                           score
                    ORDER BY score DESC
                """, k=top_k, vec=query_vec, tables=restrict_to_tables)
            else:
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
        top_score = f"{columns[0].score:.4f}" if columns else "N/A"
        print(f"  [_vector_search] → {len(columns)} results (top score: {top_score})")
        return columns

    def _llm_rerank(
        self, question: str, candidates: List[RetrievedColumn]
    ) -> List[RetrievedColumn]:
        print(f"  [_llm_rerank] Reranking {len(candidates)} candidates with LLM...")
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
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            selected = json.loads(raw[start:end])
            print(f"  [_llm_rerank] LLM selected {len(selected)} columns: {selected[:5]}...")
        except Exception as e:
            logger.warning("Re-rank parse failed (%s), using top-25 by score.", e)
            print(f"  [_llm_rerank] Parse failed ({e}), falling back to top-25 by score")
            return candidates[:25]

        selected_set = set(selected)
        col_map = {f"{c.table}.{c.name}": c for c in candidates}
        reranked = [col_map[s] for s in selected if s in col_map]
        for c in candidates:
            if len(reranked) >= 25:
                break
            if f"{c.table}.{c.name}" not in selected_set:
                reranked.append(c)
        return reranked[:25]

    def _get_join_paths(
        self,
        col_names: List[str],
        table_names: List[str],
        restrict_to_tables: Optional[List[str]],
    ) -> List[str]:
        print(f"  [_get_join_paths] tables={table_names}, scoped_tables={restrict_to_tables}")
        with self.driver.session(database=self.cfg.neo4j_database) as session:
            if restrict_to_tables:
                result = session.run("""
                    MATCH path = (c1:Column)-[:REFERENCES*1..3]->(c2:Column)
                    WHERE (c1.table IN $tables OR c2.table IN $tables)
                      AND c1.table IN $all_scoped
                      AND c2.table IN $all_scoped
                    RETURN DISTINCT [rel IN relationships(path) | rel.join_condition]
                           AS join_conditions
                """, tables=table_names, all_scoped=restrict_to_tables)
            else:
                result = session.run("""
                    MATCH path = (c1:Column)-[:REFERENCES*1..3]->(c2:Column)
                    WHERE c1.table IN $tables OR c2.table IN $tables
                    RETURN DISTINCT [rel IN relationships(path) | rel.join_condition]
                           AS join_conditions
                """, tables=table_names)

            joins: Set[str] = set()
            for rec in result:
                for jc in (rec["join_conditions"] or []):
                    if jc:
                        joins.add(jc)
        print(f"  [_get_join_paths] → {len(joins)} join conditions: {list(joins)}")
        return list(joins)