"""
kg/schema_retriever.py — Stage 0: Scoped Schema Retrieval.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from neo4j import GraphDatabase

from config import Config
from kg.schema_registry import SchemaRegistry, FKEdge
from utils.llm_client import get_embedding_client, embed_text

logger = logging.getLogger(__name__)


@dataclass
class ScopedSchema:
    relevant_tables: List[str] = field(default_factory=list)
    scoped_manifest: Dict[str, Dict[str, str]] = field(default_factory=dict)
    scoped_schema_block: str = ""
    scoped_domain_rules: str = ""
    scoped_fk_edges: List[FKEdge] = field(default_factory=list)
    scoped_all_columns: FrozenSet[str] = field(default_factory=frozenset)
    table_scores: Dict[str, float] = field(default_factory=dict)
    retrieval_method: str = "combined"

    def is_empty(self) -> bool:
        return len(self.relevant_tables) == 0


class SchemaRetriever:
    def __init__(
        self,
        cfg: Config,
        registry: SchemaRegistry,
        top_k_tables: int = 6,
        embedding_threshold: float = 0.40,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.top_k_tables = top_k_tables
        self.embedding_threshold = embedding_threshold
        self._emb = get_embedding_client(cfg)
        self._driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password)
        )
        self._concept_map: Dict[str, List[str]] = registry.build_concept_map()
        print(f"[SchemaRetriever INIT] top_k={top_k_tables}, threshold={embedding_threshold}")
        print(f"  Concept map entries: {len(self._concept_map)}")

    def close(self) -> None:
        self._driver.close()

    def retrieve_scoped_schema(self, question: str) -> ScopedSchema:
        print(f"\n[Stage 0 INPUT] SchemaRetriever.retrieve_scoped_schema()")
        print(f"  question: {repr(question)}")

        all_table_names = set(self.registry.table_names)

        concept_tables, concept_scores = self._concept_map_lookup(question)
        print(f"[Stage 0] Signal B (concept map) → tables={concept_tables}, scores={concept_scores}")
        logger.info("[SchemaRetriever] Concept map hit tables: %s", concept_tables)

        embed_tables, embed_scores = self._embedding_lookup(question)
        print(f"[Stage 0] Signal A (embedding)   → tables={embed_tables}, scores={ {k: round(v,3) for k,v in embed_scores.items()} }")
        logger.info("[SchemaRetriever] Embedding hit tables: %s", embed_tables)

        merged_scores: Dict[str, float] = {}
        for t in concept_tables:
            merged_scores[t] = max(merged_scores.get(t, 0.0), concept_scores.get(t, 0.5))
        for t in embed_tables:
            merged_scores[t] = max(merged_scores.get(t, 0.0), embed_scores.get(t, 0.0))

        if not merged_scores:
            logger.warning("[SchemaRetriever] No signal — falling back to full schema.")
            print("[Stage 0] No signal from any source → falling back to full schema")
            merged_scores = {t: 0.3 for t in all_table_names}

        ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"[Stage 0] Ranked merged scores: {[(t, round(s,3)) for t,s in ranked]}")
        seed_tables = [t for t, _ in ranked[:self.top_k_tables] if t in all_table_names]
        print(f"[Stage 0] Seed tables (top {self.top_k_tables}): {seed_tables}")

        scoped_tables = self._fk_closure(seed_tables)
        print(f"[Stage 0] Signal C (FK closure) → final scoped tables: {scoped_tables}")
        logger.info("[SchemaRetriever] Final scoped tables: %s", scoped_tables)

        if concept_tables and embed_tables:
            method = "combined"
        elif concept_tables:
            method = "concept_map"
        elif embed_tables:
            method = "embedding"
        else:
            method = "fallback_full"

        result = self._build_scoped_schema(scoped_tables, merged_scores, method)
        print(f"[Stage 0 OUTPUT] method={method}, tables={result.relevant_tables}")
        print(f"  scoped_all_columns count: {len(result.scoped_all_columns)}")
        print(f"  scoped FK edges          : {len(result.scoped_fk_edges)}")
        return result

    def _concept_map_lookup(self, question: str) -> Tuple[List[str], Dict[str, float]]:
        q_lower = question.lower()
        col_to_tables: Dict[str, List[str]] = self.registry.col_to_tables
        hit_tables: Set[str] = set()

        matched_concepts = []
        for concept, cols in self._concept_map.items():
            if concept in q_lower:
                matched_concepts.append(concept)
                for col in cols:
                    for tbl in col_to_tables.get(col, []):
                        hit_tables.add(tbl)

        print(f"  [_concept_map_lookup] matched concepts: {matched_concepts[:10]}")
        scores = {t: 0.50 for t in hit_tables}
        return list(hit_tables), scores

    def _embedding_lookup(self, question: str) -> Tuple[List[str], Dict[str, float]]:
        try:
            q_vec = embed_text(self._emb, self.cfg.embedding_deployment, question)
        except Exception as e:
            logger.warning("[SchemaRetriever] Embedding failed: %s", e)
            print(f"  [_embedding_lookup] Embedding failed: {e}")
            return [], {}

        table_results = self._vector_search_tables(q_vec)
        if table_results:
            filtered = {t: s for t, s in table_results.items() if s >= self.embedding_threshold}
            print(f"  [_embedding_lookup] Table-level results (filtered): {filtered}")
            return list(filtered.keys()), filtered

        col_results = self._vector_search_columns_aggregate(q_vec)
        filtered = {t: s for t, s in col_results.items() if s >= self.embedding_threshold}
        print(f"  [_embedding_lookup] Column-aggregate results (filtered): {filtered}")
        return list(filtered.keys()), filtered

    def _vector_search_tables(self, q_vec: List[float]) -> Dict[str, float]:
        try:
            with self._driver.session(database=self.cfg.neo4j_database) as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('table_embeddings', $k, $vec)
                    YIELD node, score
                    WHERE score > 0.3
                    RETURN node.name AS name, score
                    ORDER BY score DESC
                """, k=self.top_k_tables * 2, vec=q_vec)
                res = {rec["name"].upper(): rec["score"] for rec in result if rec["name"]}
                print(f"  [_vector_search_tables] raw results: { {k: round(v,3) for k,v in res.items()} }")
                return res
        except Exception as e:
            logger.debug("[SchemaRetriever] table_embeddings index query failed: %s", e)
            print(f"  [_vector_search_tables] failed (table_embeddings index absent?): {e}")
            return {}

    def _vector_search_columns_aggregate(self, q_vec: List[float]) -> Dict[str, float]:
        try:
            with self._driver.session(database=self.cfg.neo4j_database) as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('column_embeddings', $k, $vec)
                    YIELD node, score
                    WHERE score > 0.35
                    RETURN node.table AS tbl, score
                    ORDER BY score DESC
                """, k=50, vec=q_vec)
                table_scores: Dict[str, float] = {}
                for rec in result:
                    t = (rec["tbl"] or "").upper()
                    if t:
                        table_scores[t] = max(table_scores.get(t, 0.0), rec["score"])
                print(f"  [_vector_search_columns_aggregate] results: { {k: round(v,3) for k,v in table_scores.items()} }")
                return table_scores
        except Exception as e:
            logger.warning("[SchemaRetriever] Column embedding search failed: %s", e)
            print(f"  [_vector_search_columns_aggregate] failed: {e}")
            return {}

    def _fk_closure(self, seed_tables: List[str]) -> List[str]:
        scoped: Set[str] = set(seed_tables)
        registry_tables = set(self.registry.table_names)
        added = []

        for edge in self.registry.fk_edges:
            src_in_scope = edge.src_table in scoped
            ref_in_scope = edge.ref_table in scoped
            if src_in_scope and edge.ref_table in registry_tables and edge.ref_table not in scoped:
                scoped.add(edge.ref_table)
                added.append(edge.ref_table)
            elif ref_in_scope and edge.src_table in registry_tables and edge.src_table not in scoped:
                scoped.add(edge.src_table)
                added.append(edge.src_table)

        print(f"  [_fk_closure] seed={seed_tables}, FK-added={added}")
        ordered = list(seed_tables)
        for t in scoped:
            if t not in ordered:
                ordered.append(t)
        return ordered

    def _build_scoped_schema(
        self,
        tables: List[str],
        scores: Dict[str, float],
        method: str,
    ) -> ScopedSchema:
        registry_tables = self.registry.tables

        scoped_manifest: Dict[str, Dict[str, str]] = {}
        for t in tables:
            tmeta = registry_tables.get(t)
            if tmeta:
                scoped_manifest[t] = {
                    cname: cmeta.description
                    for cname, cmeta in tmeta.columns.items()
                }

        scoped_all_columns: FrozenSet[str] = frozenset(
            col for cols in scoped_manifest.values() for col in cols
        )

        scoped_fks = [
            e for e in self.registry.fk_edges
            if e.src_table in set(tables) and e.ref_table in set(tables)
        ]

        scoped_schema_block = self._render_scoped_schema_block(tables, scoped_manifest, scoped_fks)
        scoped_domain_rules = self._render_scoped_domain_rules(tables, scoped_fks)

        return ScopedSchema(
            relevant_tables=tables,
            scoped_manifest=scoped_manifest,
            scoped_schema_block=scoped_schema_block,
            scoped_domain_rules=scoped_domain_rules,
            scoped_fk_edges=scoped_fks,
            scoped_all_columns=scoped_all_columns,
            table_scores=scores,
            retrieval_method=method,
        )

    def _render_scoped_schema_block(
        self,
        tables: List[str],
        manifest: Dict[str, Dict[str, str]],
        fk_edges: List[FKEdge],
    ) -> str:
        lines: List[str] = [
            f"=== RELEVANT SCHEMA ({len(tables)} tables) ===",
            "",
        ]
        registry_tables = self.registry.tables
        for tname in tables:
            tmeta = registry_tables.get(tname)
            lines.append(f"Table: {tname}")
            if tmeta and tmeta.description:
                lines.append(f"  ({tmeta.description[:120]})")
            for col, desc in (manifest.get(tname) or {}).items():
                cmeta = (tmeta.columns.get(col) if tmeta else None)
                samples_str = ""
                if cmeta and cmeta.sample_values:
                    samples_str = f"  [e.g. {', '.join(str(s) for s in cmeta.sample_values[:4])}]"
                lines.append(f"  {col:<24} — {desc[:100]}{samples_str}")
            lines.append("")

        if fk_edges:
            lines.append("FK JOIN CONDITIONS available:")
            for e in fk_edges:
                lines.append(f"  {e.join_condition}")
            lines.append("")

        lines.append(
            "IMPORTANT: Use ONLY column names listed above. "
            "Do NOT invent or guess column names."
        )
        return "\n".join(lines)

    def _render_scoped_domain_rules(
        self, tables: List[str], fk_edges: List[FKEdge]
    ) -> str:
        scoped_set = set(tables)
        lines: List[str] = ["DOMAIN RULES (always follow):"]
        rule_num = 1

        for rule in self.registry.domain_rules:
            if rule.column:
                tbl = rule.column.split(".")[0]
                if tbl in scoped_set and rule.code_values:
                    codes_str = ", ".join(f"{k}={v}" for k, v in rule.code_values.items())
                    lines.append(f"{rule_num}. {rule.description}: {codes_str}")
                    rule_num += 1

        col_to_tables = self.registry.col_to_tables
        for col, tbl_list in col_to_tables.items():
            scoped_owners = [t for t in tbl_list if t in scoped_set]
            if len(scoped_owners) >= 2:
                for t in scoped_owners:
                    tmeta = self.registry.tables.get(t)
                    cmeta = tmeta.columns.get(col) if tmeta else None
                    short_desc = (cmeta.description[:60] if cmeta else col) if cmeta else col
                    lines.append(f"{rule_num}. Use {t}.{col} for: {short_desc}")
                    rule_num += 1

        for tname in tables:
            tmeta = self.registry.tables.get(tname)
            if not tmeta:
                continue
            gross = [c for c in tmeta.columns if "GROSS" in c and "AMT" in c]
            net   = [c for c in tmeta.columns if "NET" in c   and "AMT" in c]
            if gross and net:
                lines.append(
                    f"{rule_num}. In {tname}: {gross[0]} is before deductions; "
                    f"{net[0]} is after deductions."
                )
                rule_num += 1

        hub_tables = {e.ref_table for e in fk_edges}
        child_tables = {e.src_table for e in fk_edges}
        optional_tables = child_tables - hub_tables
        for ot in optional_tables:
            if ot in scoped_set:
                lines.append(
                    f"{rule_num}. Always use LEFT JOIN for {ot} unless explicitly filtering on its columns."
                )
                rule_num += 1

        lines.append(
            f"{rule_num}. Dates are stored as ISO strings (YYYY-MM-DD); use date() for SQLite date math."
        )
        rule_num += 1
        lines.append(
            f"{rule_num}. Generate SQLite-compatible SQL only — no Oracle syntax (ROWNUM, SYSDATE, NVL)."
        )

        return "\n".join(lines)


def build_table_embeddings(cfg: Config, registry: SchemaRegistry) -> None:
    from utils.llm_client import get_embedding_client, embed_batch
    emb_client = get_embedding_client(cfg)
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))

    table_names: List[str] = []
    texts: List[str] = []

    for tname, tmeta in registry.tables.items():
        col_descs = " ".join(
            f"{cname}: {cmeta.description}"
            for cname, cmeta in tmeta.columns.items()
        )
        full_text = f"Table {tname}: {tmeta.description}. Columns: {col_descs}"
        table_names.append(tname)
        texts.append(full_text[:2000])

    print(f"[build_table_embeddings] Embedding {len(texts)} table descriptions...")
    logger.info("Embedding %d table descriptions...", len(texts))
    vectors = embed_batch(emb_client, cfg.embedding_deployment, texts)

    with driver.session(database=cfg.neo4j_database) as session:
        for tname, vec in zip(table_names, vectors):
            session.run("""
                MATCH (t:Table {name: $name})
                SET t.embedding = $vec
            """, name=tname, vec=vec)

        try:
            session.run("""
                CREATE VECTOR INDEX table_embeddings IF NOT EXISTS
                FOR (t:Table) ON (t.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 1536,
                                       `vector.similarity_function`: 'cosine'}}
            """)
            print("[build_table_embeddings] table_embeddings vector index created/confirmed.")
            logger.info("table_embeddings vector index created/confirmed.")
        except Exception as e:
            logger.warning("Table embedding index creation: %s", e)
            print(f"[build_table_embeddings] Index creation warning: {e}")

    driver.close()
    print(f"[build_table_embeddings] Done — {len(table_names)} tables embedded.")
    logger.info("Table embeddings stored for %d tables.", len(table_names))