"""
pipeline/sql_generator.py — Stage 4: SQL Generation.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from config import Config
from kg.schema_registry import SchemaRegistry
from kg.schema_retriever import ScopedSchema
from kg.retriever import RetrievalResult
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)


class SQLColumnNotFoundError(Exception):
    def __init__(self, invalid_refs: List[str], table_names: List[str]) -> None:
        self.invalid_refs = invalid_refs
        tables_str = ", ".join(table_names) if table_names else "the schema"
        detail = (
            f"Your question references data that is not stored in this database. "
            f"The following column(s) do not exist in {tables_str}: "
            f"{', '.join(invalid_refs)}. "
            f"Please rephrase your question using available data fields."
        )
        super().__init__(detail)


_SQL_KEYWORDS: FrozenSet[str] = frozenset({
    "SELECT", "WHERE", "ON", "SET", "VALUES", "INTO", "WITH",
    "LATERAL", "UNNEST", "DUAL", "TABLE", "AS", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL",
    "NATURAL", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "BETWEEN", "LIKE", "CASE", "WHEN", "THEN", "ELSE", "END",
    "EXISTS", "DISTINCT", "ALL", "UNION", "INTERSECT", "EXCEPT",
    "LIMIT", "OFFSET", "ORDER", "GROUP", "BY", "HAVING", "ASC", "DESC",
})


def _validate_sql_columns(sql: str, manifest: Dict[str, Dict[str, str]]) -> List[str]:
    invalid: List[str] = []
    for m in re.finditer(r'\b([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\b', sql.upper()):
        tbl_or_alias, col = m.group(1), m.group(2)
        if tbl_or_alias in _SQL_KEYWORDS:
            continue
        if tbl_or_alias not in manifest:
            continue
        if col not in manifest[tbl_or_alias]:
            ref = f"{tbl_or_alias}.{col}"
            if ref not in invalid:
                invalid.append(ref)
    return invalid


_JOIN_CLAUSE_RE = re.compile(
    r'(?P<jointype>'
    r'(?:LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|INNER\s+|CROSS\s+|FULL\s+(?:OUTER\s+)?)?'
    r'JOIN\s+)'
    r'(?P<table>[A-Z_][A-Z0-9_$#]*)'
    r'(?:\s+(?:AS\s+)?(?P<alias>[A-Z_][A-Z0-9_$#]*))?'
    r'(?:\s+ON\s+(?P<on_clause>.+?))?'
    r'(?=\s+(?:LEFT|RIGHT|INNER|CROSS|FULL|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|$))',
    re.IGNORECASE | re.DOTALL,
)


def _tables_and_aliases(sql: str) -> Tuple[Dict[str, str], str]:
    alias_map: Dict[str, str] = {}
    from_m = re.search(
        r'\bFROM\s+([A-Z_][A-Z0-9_$#]*)(?:\s+(?:AS\s+)?([A-Z_][A-Z0-9_$#]*))?',
        sql, re.IGNORECASE,
    )
    from_table = ""
    if from_m:
        from_table = from_m.group(1).upper()
        if from_m.group(2) and from_m.group(2).upper() not in _SQL_KEYWORDS:
            alias_map[from_m.group(2).upper()] = from_table
    for m in _JOIN_CLAUSE_RE.finditer(sql):
        tbl   = m.group("table").upper()
        alias = (m.group("alias") or "").upper()
        if alias and alias not in _SQL_KEYWORDS and alias != tbl:
            alias_map[alias] = tbl
    return alias_map, from_table


def _col_refs_outside_joins(sql: str) -> Set[str]:
    stripped = re.sub(
        r'\bON\s+.+?(?=\s+(?:LEFT|RIGHT|INNER|CROSS|FULL|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT)|$)',
        ' ', sql, flags=re.IGNORECASE | re.DOTALL,
    )
    refs: Set[str] = set()
    for m in re.finditer(r'\b([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)\b', stripped.upper()):
        refs.add(m.group(1))
    return refs


def prune_unnecessary_joins(sql: str) -> str:
    if not sql or not sql.strip():
        return sql
    alias_map, from_table = _tables_and_aliases(sql)
    raw_refs = _col_refs_outside_joins(sql)

    contributing: Set[str] = set()
    for ref in raw_refs:
        contributing.add(alias_map.get(ref, ref))
    if from_table:
        contributing.add(from_table)

    joins = list(_JOIN_CLAUSE_RE.finditer(sql))
    if not joins:
        return sql

    surviving = [m for m in joins if m.group("table").upper() in contributing]
    if len(surviving) == len(joins):
        return sql

    first_start = joins[0].start()
    last_end    = joins[-1].end()
    prefix  = sql[:first_start].rstrip()
    suffix  = sql[last_end:].lstrip()
    middle  = "\n".join(m.group(0).strip() for m in surviving)
    pruned  = f"{prefix}\n{middle}\n{suffix}" if middle else f"{prefix}\n{suffix}"
    pruned  = re.sub(r'\n{3,}', '\n\n', pruned).strip()
    print(f"[prune_unnecessary_joins] {len(joins)} → {len(surviving)} JOINs")
    logger.info("prune_unnecessary_joins: %d → %d JOINs", len(joins), len(surviving))
    return pruned


class SQLGenerator:
    def __init__(
        self,
        cfg: Config,
        registry: SchemaRegistry,
        scoped_schema: Optional[ScopedSchema] = None,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.scoped_schema = scoped_schema
        self.gpt = get_gpt_client(cfg)

        if scoped_schema and not scoped_schema.is_empty():
            self._manifest = scoped_schema.scoped_manifest
            self._domain_rules = scoped_schema.scoped_domain_rules
            self._schema_block = scoped_schema.scoped_schema_block
            self._table_names = scoped_schema.relevant_tables
        else:
            self._manifest = registry.manifest
            self._domain_rules = registry.build_domain_rules_text()
            self._schema_block = registry.build_schema_block()
            self._table_names = registry.table_names

        self._few_shot = registry.build_fewshot_examples()
        self._sql_system = self._build_sql_system()

        print(f"[SQLGenerator INIT] tables={self._table_names}")
        print(f"  System prompt length: {len(self._sql_system)} chars")

    def update_scoped_schema(self, scoped_schema: ScopedSchema) -> None:
        self.scoped_schema = scoped_schema
        if scoped_schema and not scoped_schema.is_empty():
            self._manifest = scoped_schema.scoped_manifest
            self._domain_rules = scoped_schema.scoped_domain_rules
            self._schema_block = scoped_schema.scoped_schema_block
            self._table_names = scoped_schema.relevant_tables
        else:
            self._manifest = self.registry.manifest
            self._domain_rules = self.registry.build_domain_rules_text()
            self._schema_block = self.registry.build_schema_block()
            self._table_names = self.registry.table_names
        self._sql_system = self._build_sql_system()
        print(f"[SQLGenerator] update_scoped_schema() → tables now: {self._table_names}")

    def _build_sql_system(self) -> str:
        return (
            "You are a precise SQL generator.\n"
            "Generate SQLite-compatible SQL only. "
            "Return ONLY the SQL statement — no explanation, no markdown fences.\n\n"
            f"{self._domain_rules}\n\n"
            f"{self._few_shot}\n"
        )

    def generate(self, question: str, retrieval: RetrievalResult) -> str:
        print(f"\n[Stage 4 INPUT] SQLGenerator.generate()")
        print(f"  Question        : {question}")
        print(f"  Retrieval cols  : {len(retrieval.columns)}")
        print(f"  Join conditions : {retrieval.join_conditions}")

        columns_block = self._format_columns(retrieval)
        joins_block = (
            "\n".join(retrieval.join_conditions)
            if retrieval.join_conditions else "None identified"
        )

        user_prompt = (
            f"Relevant columns (top {len(retrieval.columns)}):\n{columns_block}\n\n"
            f"Available JOIN conditions:\n{joins_block}\n\n"
            f"Question: {question}"
        )

        print(f"[Stage 4] Sending to LLM | user_prompt ({len(user_prompt)} chars):")
        print(f"  {user_prompt[:300]}{'...' if len(user_prompt) > 300 else ''}")

        sql = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            self._sql_system, user_prompt,
            temperature=0.0, max_tokens=800,
        )
        sql = sql.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()

        print(f"[Stage 4] Raw SQL from LLM:\n{sql}")

        sql = prune_unnecessary_joins(sql)

        invalid_refs = _validate_sql_columns(sql, self._manifest)
        if invalid_refs:
            print(f"[Stage 4 VALIDATION FAILED] Invalid column refs: {invalid_refs}")
            logger.warning(
                "SQL column validation failed — invalid refs: %s\nSQL:\n%s",
                invalid_refs, sql,
            )
            raise SQLColumnNotFoundError(invalid_refs, self._table_names)

        print(f"[Stage 4 OUTPUT] Final validated SQL:\n{sql}")
        logger.debug("Generated SQL:\n%s", sql)
        return sql

    @staticmethod
    def _format_columns(retrieval: RetrievalResult) -> str:
        lines = []
        for col in retrieval.columns:
            sample_str = f"  samples={col.sample_values}" if col.sample_values else ""
            lines.append(
                f"  {col.table}.{col.name} [{col.description[:120]}]{sample_str}"
            )
        return "\n".join(lines)
