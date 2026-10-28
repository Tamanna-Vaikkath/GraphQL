"""
schema_registry.py — Dynamic Schema Registry.

Single source of truth for schema metadata, domain rules, and FK relationships.
Everything is derived at runtime from SQLite (schema) and Neo4j (enriched
column descriptions + FK edges).  

Replace all imports of:
    from pipeline.hyde_expander import SCHEMA_MANIFEST, _ALL_COLUMNS
    from kg.build_graph import SCHEMA_META
with:
    from schema_registry import SchemaRegistry
    registry = SchemaRegistry.load(cfg)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Data classes 

@dataclass
class ColumnMeta:
    name: str
    table: str
    data_type: str
    description: str               # LLM-enriched (from Neo4j) 
    is_pk: bool
    is_fk: bool
    sample_values: List[str]
    null_pct: float
    references: Optional[str] = None   # "REFTABLE.REFCOL" when is_fk=True


@dataclass
class TableMeta:
    name: str
    description: str               # LLM-enriched (from Neo4j) 
    columns: Dict[str, ColumnMeta] = field(default_factory=dict)
    fk_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    # fk_edges: list of (src_col, ref_table.ref_col, join_condition)


@dataclass
class FKEdge:
    src_table: str
    src_col: str
    ref_table: str
    ref_col: str
    join_condition: str            # e.g. "CLAIMS.POLICY_ID = POLICY.POLICY_ID"
    cardinality: str               # e.g. "MANY_TO_ONE"
    description: str


@dataclass
class DomainRule:
    rule_id: int
    description: str
    column: Optional[str] = None   # "TABLE.COLUMN" this rule applies to
    code_values: Optional[Dict[str, str]] = None  # {"O": "Open", "C": "Closed", …}


class SchemaRegistry:
    """
    Unified, lazily-loaded schema registry.

    Sources (priority order):
      1. Neo4j knowledge graph   — enriched descriptions, FK edges, embeddings
      2. SQLite PRAGMA            — column names, types, PKs, FKs, sample values
      3. Auto-generation          — fallback descriptions when KG is unavailable
    """

    def __init__(self) -> None:
        self._tables: Dict[str, TableMeta] = {}
        self._fk_edges: List[FKEdge] = []
        self._domain_rules: List[DomainRule] = []
        # Alias index: lowercase alias term → "TABLE" or "TABLE.COLUMN".
        # Populated by _generate_standard_aliases() (mechanical, derived from
        # the schema's own names) and then overlaid by _load_custom_aliases()
        # (hand-curated abbreviations / business synonyms from aliases.json).
        self._aliases: Dict[str, str] = {}
        self._custom_aliases: Dict[str, str] = {}
        self._loaded = False

    # Public API 

    @classmethod
    def load(cls, cfg) -> "SchemaRegistry":
        """
        Build a fully-populated SchemaRegistry from the live database.

        Steps
        -----
        1. Read every table from SQLite (PRAGMA table_info + FK pragma).
        2. Overlay enriched descriptions and FK edges from Neo4j if available.
        3. Derive domain rules from sample values found in status-code columns.
        4. Build the ontology concept map from column descriptions.
        """
        data_backend = getattr(cfg, "data_backend", "sqlite")
        print(f"\n{'='*60}")
        print(f"[SchemaRegistry.load] START")
        print(f"  data_backend   : {data_backend}")
        if data_backend == "databricks":
            print(f"  engine         : Databricks SQL Warehouse (databricks-sql-connector)")
            print(f"  catalog.schema : {cfg.databricks_catalog}.{cfg.databricks_schema}")
        else:
            print(f"  sqlite_db_path : {cfg.sqlite_db_path}")
        print(f"  neo4j_uri      : {cfg.neo4j_uri}")
        print(f"{'='*60}")

        reg = cls()

        if data_backend == "databricks":
            print(f"\n[SchemaRegistry.load] Step 1/3 — Loading via Databricks SQL Warehouse...")
            reg._load_from_databricks(cfg)
            print(f"[SchemaRegistry.load] SQL Warehouse load complete → {len(reg._tables)} tables, {len(reg._fk_edges)} FK edges")
        else:
            print(f"\n[SchemaRegistry.load] Step 1/3 — Loading from SQLite...")
            reg._load_from_sqlite(cfg.sqlite_db_path)
            print(f"[SchemaRegistry.load] SQLite load complete → {len(reg._tables)} tables, {len(reg._fk_edges)} FK edges")

        print(f"\n[SchemaRegistry.load] Step 2/3 — Overlaying from Neo4j...")
        try:
            reg._overlay_from_neo4j(cfg)
            print(f"[SchemaRegistry.load] Neo4j overlay complete")
        except Exception as e:
            logger.warning("Neo4j overlay skipped (%s). Using SQLite-only schema.", e)
            print(f"[SchemaRegistry.load] Neo4j overlay SKIPPED: {e}")

        print(f"\n[SchemaRegistry.load] Step 3/4 — Generating aliases...")
        reg._generate_standard_aliases()
        n_custom = reg._load_custom_aliases(getattr(cfg, "aliases_json_path", None))
        print(
            f"[SchemaRegistry.load] Aliases ready → {len(reg._aliases)} total "
            f"({len(reg._aliases) - n_custom} auto-generated, {n_custom} from aliases.json)"
        )

        print(f"\n[SchemaRegistry.load] Step 4/4 — Deriving domain rules...")
        reg._derive_domain_rules()
        print(f"[SchemaRegistry.load] Domain rules derived → {len(reg._domain_rules)} rules")

        reg._loaded = True
        print(f"\n[SchemaRegistry.load] DONE")
        print(f"  Tables       : {reg.table_names}")
        print(f"  FK edges     : {len(reg._fk_edges)}")
        print(f"  Aliases      : {len(reg._aliases)} ({n_custom} custom)")
        print(f"  Domain rules : {len(reg._domain_rules)}")
        print(f"{'='*60}\n")
        return reg

    @property
    def tables(self) -> Dict[str, TableMeta]:
        return self._tables

    @property
    def table_names(self) -> List[str]:
        return list(self._tables.keys())

    @property
    def fk_edges(self) -> List[FKEdge]:
        return self._fk_edges

    @property
    def domain_rules(self) -> List[DomainRule]:
        return self._domain_rules

    def get_column(self, table: str, column: str) -> Optional[ColumnMeta]:
        t = self._tables.get(table.upper())
        if t:
            return t.columns.get(column.upper())
        return None

    @property
    def aliases(self) -> Dict[str, str]:
        """
        Merged alias index: lowercase alias term → canonical "TABLE" or
        "TABLE.COLUMN". Combines mechanically-generated standard aliases
        (naming-convention variants) with hand-curated abbreviations /
        business synonyms loaded from aliases.json.
        """
        return dict(self._aliases)

    @property
    def custom_aliases(self) -> Dict[str, str]:
        """Only the hand-curated entries loaded from aliases.json."""
        return dict(self._custom_aliases)

    def resolve_alias(self, term: str) -> Optional[str]:
        """
        Resolve a natural-language term, abbreviation, or business synonym
        to its canonical "TABLE" or "TABLE.COLUMN" reference, or None if
        the term isn't a known alias.
        """
        return self._aliases.get(term.strip().lower())

    @property
    def manifest(self) -> Dict[str, Dict[str, str]]:
        """
        Returns { TABLE: { COL: description } } — drop-in replacement for the
        old hardcoded SCHEMA_MANIFEST dict in hyde_expander.py.
        """
        return {
            tname: {cname: c.description for cname, c in tmeta.columns.items()}
            for tname, tmeta in self._tables.items()
        }

    @property
    def all_columns(self) -> FrozenSet[str]:
        """Flat set of all column names for O(1) existence checks."""
        return frozenset(
            col
            for tmeta in self._tables.values()
            for col in tmeta.columns
        )

    @property
    def col_to_tables(self) -> Dict[str, List[str]]:
        """Reverse index: column name → list of tables containing it."""
        idx: Dict[str, List[str]] = {}
        for tname, tmeta in self._tables.items():
            for cname in tmeta.columns:
                idx.setdefault(cname, []).append(tname)
        return idx

    def build_schema_block(self) -> str:
        """
        Render the schema as a compact text block for LLM prompts.
        Automatically reflects the current table/column set — no manual updates.
        """
        print(f"\n  [build_schema_block] Building schema block for {len(self._tables)} tables: {self.table_names}")
        n_tables = len(self._tables)
        lines: List[str] = [
            f"=== GROUND-TRUTH SCHEMA ({n_tables} tables — reference ONLY these columns) ===",
            "",
        ]
        for table, tmeta in self._tables.items():
            lines.append(f"Table: {table}")
            if tmeta.description:
                lines.append(f"  ({tmeta.description})")
            for col, cmeta in tmeta.columns.items():
                lines.append(f"  {col:<24} — {cmeta.description}")
            lines.append("")

        if self._custom_aliases:
            lines.append("=== KNOWN ABBREVIATIONS / BUSINESS SYNONYMS ===")
            for alias, target in sorted(self._custom_aliases.items()):
                lines.append(f"  {alias} → {target}")
            lines.append("")

        lines.append(
            "IMPORTANT: You MUST NOT invent, guess, or use any column name not listed above. "
            "If the question implies a concept with no matching column, do not attempt to represent it."
        )
        result = "\n".join(lines)
        print(f"  [build_schema_block] Schema block built ({len(result)} chars)")
        return result

    def build_domain_rules_text(self) -> str:
        """
        Render domain rules as the DOMAIN_RULES text block for the SQL prompt.
        Derived from actual sample values in the DB — no hardcoding required.
        """
        print(f"\n  [build_domain_rules_text] Building domain rules from {len(self._domain_rules)} rules...")
        lines = ["DOMAIN RULES (always follow):"]
        rule_num = 1

        # Status-code rules derived from sample values
        for rule in self._domain_rules:
            if rule.code_values:
                codes_str = ", ".join(f"{k}={v}" for k, v in rule.code_values.items())
                lines.append(f"{rule_num}. {rule.description}: {codes_str}")
                rule_num += 1

        # Structural rules derived from FK topology and column naming patterns
        for tname, tmeta in self._tables.items():
            for cname, cmeta in tmeta.columns.items():
                # Detect STATE_CD ambiguity
                if cname == "STATE_CD" and len(self.col_to_tables.get("STATE_CD", [])) > 1:
                    other_tables = [t for t in self.col_to_tables["STATE_CD"] if t != tname]
                    lines.append(
                        f"{rule_num}. Use {tname}.STATE_CD for the {tname.lower()} state; "
                        f"use {other_tables[0]}.STATE_CD for the {other_tables[0].lower()} state."
                    )
                    rule_num += 1
                    break  

        # Amount semantics derived from column names
        for tname, tmeta in self._tables.items():
            gross = [c for c in tmeta.columns if "GROSS" in c]
            net   = [c for c in tmeta.columns if "NET" in c and "AMT" in c]
            if gross and net:
                lines.append(
                    f"{rule_num}. In {tname}: {gross[0]} is before deductions; "
                    f"{net[0]} is after deductions."
                )
                rule_num += 1

        # Join preference: LEFT JOIN when joining optional tables
        optional_tables = [
            tname for tname, tmeta in self._tables.items()
            if any(e.ref_table == tname for e in self._fk_edges)
            and not any(e.src_table == tname and not e.ref_table for e in self._fk_edges)
        ]
        for ot in optional_tables:
            lines.append(
                f"{rule_num}. Always use LEFT JOIN for {ot} unless explicitly filtering on its columns."
            )
            rule_num += 1

        # Date format (dialect-specific function names are stated in the
        # system prompt by SQLGenerator._build_sql_system, which knows the
        # active data_backend; keep this rule generic so it's correct either way)
        lines.append(f"{rule_num}. Use the target SQL dialect's native date functions for date math (see system prompt).")
        rule_num += 1
        lines.append(f"{rule_num}. No Oracle-only syntax (ROWNUM, SYSDATE, NVL) regardless of backend.")

        result = "\n".join(lines)
        print(f"  [build_domain_rules_text] Done — {rule_num} rules, {len(result)} chars")
        return result

    def build_concept_map(self) -> Dict[str, List[str]]:
        """
        Build a concept→columns mapping dynamically from column descriptions
        and names.  Replaces the hardcoded _CONCEPT_TO_COLUMNS dict.
        """
        print(f"\n  [build_concept_map] Building concept map for {len(self._tables)} tables...")
        concept_map: Dict[str, List[str]] = {}

        def add(concept: str, cols: List[str]) -> None:
            k = concept.strip().lower()
            if k:
                existing = concept_map.setdefault(k, [])
                for c in cols:
                    if c not in existing:
                        existing.append(c)

        for tname, tmeta in self._tables.items():
            tl = tname.lower()
            # Table-level concepts
            pk_cols = [c for c, m in tmeta.columns.items() if m.is_pk]
            add(tl, pk_cols or list(tmeta.columns.keys())[:2])
            add(tl + "s", pk_cols or list(tmeta.columns.keys())[:2])

            for cname, cmeta in tmeta.columns.items():
                col_cols = [cname]
                # column name itself (underscores → spaces)
                readable = cname.lower().replace("_", " ").replace(" cd", "").replace(" amt", " amount").strip()
                add(readable, col_cols)
                add(cname.lower(), col_cols)

                # Also register the "bare" business term with the trailing
                # generic DB-suffix word stripped off (e.g. "litigation flg"
                # → "litigation", "atty rep flg" → "atty rep"). Without this,
                # a natural question like "claims in litigation" or "atty
                # rep" never matches the concept map — only the fully
                # suffixed forms ("litigation flag"/"litigationflag") do —
                # and gets kicked to the stricter LLM concept-coverage check
                # for no reason, since the column is really about
                # "litigation"/"atty rep", not about a generic "flag".
                _GENERIC_SUFFIX_WORDS = (
                    "flg", "flag", "nbr", "number", "qty", "quantity",
                    "desc", "description", "pct", "percent", "stat",
                    "status", "type", "amount",
                )
                words = readable.split()
                if len(words) > 1 and words[-1] in _GENERIC_SUFFIX_WORDS:
                    stem = " ".join(words[:-1]).strip()
                    if len(stem) >= 3:
                        add(stem, col_cols)

                # status-code columns: add decoded values as concepts
                if cmeta.code_values:
                    for code, label in cmeta.code_values.items():
                        add(label.lower(), col_cols)
                        add(label.lower() + " " + tl[:-1] if tl.endswith("s") else label.lower() + " " + tl, col_cols)

                # FK columns: add parent table name
                if cmeta.is_fk and cmeta.references:
                    ref_tbl = cmeta.references.split(".")[0].lower()
                    add(ref_tbl, [cname])

        # Alias-derived concepts: both the auto-generated standard aliases
        # and the hand-curated abbreviations/business synonyms resolve here
        # to the same real column(s), so a term like "pol_id" (aliases.json)
        # or "policy id" (auto-generated) both map onto POLICY.POLICY_ID.
        for alias, target in self._aliases.items():
            if "." in target:
                tname, cname = target.split(".", 1)
                if tname in self._tables and cname in self._tables[tname].columns:
                    add(alias, [cname])
            else:
                tmeta = self._tables.get(target)
                if tmeta:
                    pk_cols = [c for c, m in tmeta.columns.items() if m.is_pk]
                    add(alias, pk_cols or list(tmeta.columns.keys())[:2])

        print(f"  [build_concept_map] Done — {len(concept_map)} concept entries across {len(self._tables)} tables")
        return concept_map

    def build_fewshot_examples(self) -> str:
        """
        Generate 2–3 SQL few-shot examples automatically from the schema topology.
        Covers: status filter, multi-table join, aggregation.
        """
        print(f"\n  [build_fewshot_examples] Generating few-shot SQL examples...")
        hub_table = self._find_hub_table()
        print(f"  [build_fewshot_examples] Hub table identified: {hub_table}")
        examples = []
        if hub_table:
            tmeta = self._tables[hub_table]
            stat_col = next(
                (c for c in tmeta.columns if c.endswith("_STAT_CD") or c.endswith("_STATUS")),
                None,
            )
            pk = next((c for c, m in tmeta.columns.items() if m.is_pk), None)
            amt_col = next((c for c in tmeta.columns if "AMT" in c or "AMOUNT" in c), None)

            if stat_col and pk:
                rule = self._domain_rules_for_col(hub_table, stat_col)
                first_code = next(iter(rule.code_values), "O") if rule and rule.code_values else "O"
                first_label = (rule.code_values or {}).get(first_code, "Open")
                cols_str = ", ".join(list(tmeta.columns.keys())[:6])
                examples.append(
                    f"Q: Show all {first_label.lower()} {hub_table.lower()}\n"
                    f"SQL:\nSELECT {cols_str}\n"
                    f"FROM {hub_table}\n"
                    f"WHERE {stat_col} = '{first_code}'\n"
                    f"ORDER BY {pk} ASC;"
                )

            # Aggregation example across an FK join
            child_edge = next(
                (e for e in self._fk_edges if e.ref_table == hub_table),
                None,
            )
            if child_edge and amt_col:
                parent_tbl = self._tables.get(child_edge.src_table)
                lob_col = next(
                    (c for c in (parent_tbl.columns if parent_tbl else {}) if "LINE" in c or "LOB" in c),
                    None,
                )
                if parent_tbl and lob_col:
                    examples.append(
                        f"Q: Total {amt_col.replace('_', ' ').lower()} by {lob_col.replace('_', ' ').lower()}\n"
                        f"SQL:\n"
                        f"SELECT p.{lob_col}, COUNT(c.{pk}) AS record_count, "
                        f"SUM(c.{amt_col}) AS total\n"
                        f"FROM {hub_table} c\n"
                        f"JOIN {child_edge.src_table} p ON {child_edge.join_condition}\n"
                        f"GROUP BY p.{lob_col}\n"
                        f"ORDER BY total DESC;"
                    )

        examples.extend(self._analytics_fewshots())

        if not examples:
            # Minimal fallback
            first_table = next(iter(self._tables), "TABLE1")
            examples.append(
                f"Q: Show all records\nSQL:\nSELECT * FROM {first_table} LIMIT 100;"
            )

        print(f"  [build_fewshot_examples] Generated {len(examples)} example(s)")
        return "\nEXAMPLES:\n\n" + "\n\n".join(examples)

    def _analytics_fewshots(self) -> List[str]:
        """Hand-written examples for the DEV analytics tables. Only emitted when
        the table exists, so the original 4-table schema is unaffected. Uses
        Databricks/Spark SQL date functions (datediff, etc.)."""
        t = self._tables
        out: List[str] = []
        if "CLAIM_EVENT" in t:
            out.append(
                "Q: Show the timeline of claim 12\nSQL:\n"
                "SELECT EVENT_SEQ, EVENT_DT, EVENT_TYPE_CD, FROM_STAT_CD, TO_STAT_CD, "
                "INCURRED_AMT_AT_EVENT, RESERVE_AMT_AT_EVENT, EVENT_NOTE\n"
                "FROM CLAIM_EVENT\nWHERE CLAIM_ID = 12\nORDER BY EVENT_SEQ;")
            out.append(
                "Q: How many days did each closed claim take from report to close\nSQL:\n"
                "SELECT CLAIM_ID, REPORT_DT, CLOSE_DT, datediff(CLOSE_DT, REPORT_DT) AS days_to_close\n"
                "FROM CLAIMS\nWHERE CLM_STAT_CD = 'C'\nORDER BY days_to_close DESC LIMIT 100;")
        if "CLAIM_OVERLAP" in t:
            out.append(
                "Q: Which claims collide or overlap (strong collisions)\nSQL:\n"
                "SELECT o.CLAIM_ID_A, o.CLAIM_ID_B, o.OVERLAP_TYPE_CD, o.DAYS_APART, o.OVERLAP_SCORE, o.OVERLAP_REASON\n"
                "FROM CLAIM_OVERLAP o\nWHERE o.COLLISION_FLG = 'Y'\nORDER BY o.OVERLAP_SCORE DESC LIMIT 100;")
        if "CLAIM_CLUSTER" in t and "CLAIM_CLUSTER_MEMBER" in t:
            out.append(
                "Q: Show the claims in each high-risk cluster\nSQL:\n"
                "SELECT cl.CLUSTER_ID, cl.CLUSTER_TYPE_CD, cl.CLAIM_CNT, m.CLAIM_ID, c.INCURRED_AMT\n"
                "FROM CLAIM_CLUSTER cl\nJOIN CLAIM_CLUSTER_MEMBER m ON m.CLUSTER_ID = cl.CLUSTER_ID\n"
                "JOIN CLAIMS c ON c.CLAIM_ID = m.CLAIM_ID\nWHERE cl.RISK_LEVEL_CD = 'HIGH'\n"
                "ORDER BY cl.CLUSTER_ID, m.CLAIM_ID;")
        if "CLAIM_LOSS_DETAIL" in t and "LOSS_CAUSE" in t:
            out.append(
                "Q: Top loss causes by total incurred\nSQL:\n"
                "SELECT lc.LOSS_CAUSE_CD, lc.LOSS_CAUSE_DESC, COUNT(*) AS claim_count, SUM(c.INCURRED_AMT) AS total_incurred\n"
                "FROM CLAIM_LOSS_DETAIL d\nJOIN LOSS_CAUSE lc ON lc.LOSS_CAUSE_CD = d.LOSS_CAUSE_CD\n"
                "JOIN CLAIMS c ON c.CLAIM_ID = d.CLAIM_ID\nGROUP BY lc.LOSS_CAUSE_CD, lc.LOSS_CAUSE_DESC\n"
                "ORDER BY total_incurred DESC;")
        if "CLAIM_SIMILARITY" in t:
            out.append(
                "Q: Show claims similar to claim 42\nSQL:\n"
                "SELECT s.SIMILARITY_RANK, s.SIMILAR_CLAIM_ID, s.SIMILARITY_SCORE, s.MATCH_BASIS_TXT\n"
                "FROM CLAIM_SIMILARITY s\nWHERE s.CLAIM_ID = 42\nORDER BY s.SIMILARITY_RANK;")
        return out

    # Private loaders 

    def _load_from_sqlite(self, db_path: str) -> None:
        """Read all tables, columns, PKs, FKs, and sample values from SQLite."""
        print(f"\n  [_load_from_sqlite] Connecting to: {db_path}")
        conn = sqlite3.connect(db_path)
        try:
            # Get all user tables
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            table_names = [r[0].upper() for r in table_rows]
            logger.info("SQLite tables found: %s", table_names)
            print(f"  [_load_from_sqlite] Tables found: {table_names}")

            # Build FK map from pragma
            fk_map: Dict[str, Dict[str, Tuple[str, str]]] = {}
            for tname in table_names:
                fk_rows = conn.execute(f"PRAGMA foreign_key_list({tname})").fetchall()
                for fk in fk_rows:
                    # (id, seq, table, from, to, on_update, on_delete, match)
                    ref_table = fk[2].upper()
                    from_col  = fk[3].upper()
                    to_col    = fk[4].upper()
                    fk_map.setdefault(tname, {})[from_col] = (ref_table, to_col)
                    self._fk_edges.append(FKEdge(
                        src_table=tname,
                        src_col=from_col,
                        ref_table=ref_table,
                        ref_col=to_col,
                        join_condition=f"{tname}.{from_col} = {ref_table}.{to_col}",
                        cardinality="MANY_TO_ONE",
                        description=f"Many {tname} per {ref_table}",
                    ))
            print(f"  [_load_from_sqlite] FK edges from PRAGMA: {len(self._fk_edges)}")
            for e in self._fk_edges:
                print(f"    {e.join_condition}  [{e.cardinality}]")

            for tname in table_names:
                col_rows = conn.execute(f"PRAGMA table_info({tname})").fetchall()
                table_fks = fk_map.get(tname, {})
                total = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                print(f"\n  [_load_from_sqlite] Table: {tname}  ({total} rows, {len(col_rows)} columns)")

                columns: Dict[str, ColumnMeta] = {}
                for row in col_rows:
                    cname   = row[1].upper()
                    ctype   = row[2]
                    is_pk   = bool(row[5])
                    is_fk   = cname in table_fks
                    ref_str = f"{table_fks[cname][0]}.{table_fks[cname][1]}" if is_fk else None

                    # Sample values
                    try:
                        samples = [
                            str(r[0]) for r in conn.execute(
                                f"SELECT DISTINCT {cname} FROM {tname} "
                                f"WHERE {cname} IS NOT NULL LIMIT 8"
                            ).fetchall()
                        ]
                    except Exception:
                        samples = []

                    # Null pct
                    try:
                        nulls = conn.execute(
                            f"SELECT COUNT(*) FROM {tname} WHERE {cname} IS NULL"
                        ).fetchone()[0]
                        null_pct = round(nulls / total * 100, 1) if total else 0.0
                    except Exception:
                        null_pct = 0.0

                    # Auto-generate a basic description from the column name
                    desc = _auto_describe(cname, tname, ctype, is_pk, is_fk, ref_str, samples)
                    # Detect code_values
                    code_values = _infer_code_values(cname, samples)

                    columns[cname] = ColumnMeta(
                        name=cname, table=tname, data_type=ctype,
                        description=desc, is_pk=is_pk, is_fk=is_fk,
                        sample_values=samples, null_pct=null_pct,
                        references=ref_str,
                    )
                    if code_values:
                        columns[cname].code_values = code_values   
                    else:
                        columns[cname].code_values = None           

                    pk_marker  = " [PK]" if is_pk  else ""
                    fk_marker  = f" [FK→{ref_str}]" if is_fk else ""
                    cv_marker  = f" codes={list(code_values.keys())}" if code_values else ""
                    null_marker = f" null={null_pct}%" if null_pct > 0 else ""
                    print(f"    {cname:<28} {ctype:<12}{pk_marker}{fk_marker}{cv_marker}{null_marker}")

                self._tables[tname] = TableMeta(
                    name=tname,
                    description=_auto_describe_table(tname, list(columns.keys())),
                    columns=columns,
                )
                print(f"  [_load_from_sqlite] Stored TableMeta for {tname} ({len(columns)} columns)")
        finally:
            conn.close()
        print(f"\n  [_load_from_sqlite] COMPLETE — {len(self._tables)} tables loaded")

    def _load_from_databricks(self, cfg) -> None:
        """
        Read all tables, columns, PKs, FKs, and sample values from a
        Databricks Unity Catalog catalog/schema via a **SQL Warehouse**
        connection (utils.db_connector.get_databricks_connection —
        databricks-sql-connector, no Spark/Databricks Connect involved).
        Unity Catalog exposes standard ANSI information_schema views over
        the warehouse connection the same way it would to Spark SQL, so
        this mirrors _load_from_sqlite() — everything downstream
        (manifest, aliases, domain rules, few-shot examples) is unchanged.
        """
        from utils.db_connector import run_rows

        catalog = cfg.databricks_catalog
        schema = cfg.databricks_schema
        print(f"\n  [_load_from_databricks] Using Databricks SQL Warehouse "
              f"connection for {catalog}.{schema}")

        def _run(query: str) -> List[Tuple]:
            """Execute a SQL statement on the warehouse, return rows as plain tuples."""
            return run_rows(cfg, query)

        # All user tables in this catalog.schema 
        rows = _run(
            f"SELECT table_name FROM {catalog}.information_schema.tables "
            f"WHERE table_schema = '{schema}'"
        )
        table_names = [r[0].upper() for r in rows]
        logger.info("SQL Warehouse tables found: %s", table_names)
        print(f"  [_load_from_databricks] Tables found: {table_names}")

        # FK edges via information_schema constraint views 
        # Unity Catalog supports declared PRIMARY KEY / FOREIGN KEY
        # constraints (informational, not enforced). If none are declared
        # on a fresh migration, this simply comes back empty and the
        # registry falls back to auto-generated join hints — declare FKs
        # in Unity Catalog (ALTER TABLE ... ADD CONSTRAINT ... FOREIGN KEY)
        # for the richest schema-grounding behavior.
        fk_map: Dict[str, Dict[str, Tuple[str, str]]] = {}
        try:
            rows = _run(f"""
                SELECT
                    fk.table_name        AS src_table,
                    fk_col.column_name    AS src_col,
                    pk.table_name         AS ref_table,
                    pk_col.column_name    AS ref_col
                FROM {catalog}.information_schema.referential_constraints rc
                JOIN {catalog}.information_schema.table_constraints fk
                    ON rc.constraint_name = fk.constraint_name
                    AND rc.constraint_schema = fk.constraint_schema
                JOIN {catalog}.information_schema.table_constraints pk
                    ON rc.unique_constraint_name = pk.constraint_name
                    AND rc.unique_constraint_schema = pk.constraint_schema
                JOIN {catalog}.information_schema.key_column_usage fk_col
                    ON fk.constraint_name = fk_col.constraint_name
                    AND fk.constraint_schema = fk_col.constraint_schema
                JOIN {catalog}.information_schema.key_column_usage pk_col
                    ON pk.constraint_name = pk_col.constraint_name
                    AND pk.constraint_schema = pk_col.constraint_schema
                WHERE fk.constraint_schema = '{schema}'
            """)
            for src_table, src_col, ref_table, ref_col in rows:
                src_table, src_col = src_table.upper(), src_col.upper()
                ref_table, ref_col = ref_table.upper(), ref_col.upper()
                fk_map.setdefault(src_table, {})[src_col] = (ref_table, ref_col)
                self._fk_edges.append(FKEdge(
                    src_table=src_table, src_col=src_col,
                    ref_table=ref_table, ref_col=ref_col,
                    join_condition=f"{src_table}.{src_col} = {ref_table}.{ref_col}",
                    cardinality="MANY_TO_ONE",
                    description=f"Many {src_table} per {ref_table}",
                ))
        except Exception as e:
            logger.warning(
                "No declared Unity Catalog FK constraints found (%s) — "
                "join hints will rely on Neo4j overlay / naming conventions.",
                e,
            )
            print(f"  [_load_from_databricks] No declared UC FK constraints: {e}")
        print(f"  [_load_from_databricks] FK edges from Unity Catalog: {len(self._fk_edges)}")

        # Columns, types, sample values, null % per table 
        for tname in table_names:
            fq_table = f"{catalog}.{schema}.{tname}"
            col_rows = _run(f"""
                SELECT column_name, data_type, ordinal_position
                FROM {catalog}.information_schema.columns
                WHERE table_schema = '{schema}' AND table_name = '{tname.lower()}'
                ORDER BY ordinal_position
            """)

            # Declared PK columns for this table (if any)
            pk_cols: set = set()
            try:
                rows = _run(f"""
                    SELECT kcu.column_name
                    FROM {catalog}.information_schema.table_constraints tc
                    JOIN {catalog}.information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.constraint_schema = kcu.constraint_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_schema = '{schema}' AND tc.table_name = '{tname.lower()}'
                """)
                pk_cols = {r[0].upper() for r in rows}
            except Exception:
                pass

            total = _run(f"SELECT COUNT(*) FROM {fq_table}")[0][0]
            table_fks = fk_map.get(tname, {})
            print(f"\n  [_load_from_databricks] Table: {tname}  ({total} rows, {len(col_rows)} columns)")

            columns: Dict[str, ColumnMeta] = {}
            for cname_raw, ctype, _ord in col_rows:
                cname = cname_raw.upper()
                is_pk = cname in pk_cols
                is_fk = cname in table_fks
                ref_str = f"{table_fks[cname][0]}.{table_fks[cname][1]}" if is_fk else None

                try:
                    sample_rows = _run(
                        f"SELECT DISTINCT {cname} FROM {fq_table} "
                        f"WHERE {cname} IS NOT NULL LIMIT 8"
                    )
                    samples = [str(r[0]) for r in sample_rows]
                except Exception:
                    samples = []

                try:
                    nulls = _run(f"SELECT COUNT(*) FROM {fq_table} WHERE {cname} IS NULL")[0][0]
                    null_pct = round(nulls / total * 100, 1) if total else 0.0
                except Exception:
                    null_pct = 0.0

                desc = _auto_describe(cname, tname, ctype, is_pk, is_fk, ref_str, samples)
                code_values = _infer_code_values(cname, samples)

                columns[cname] = ColumnMeta(
                    name=cname, table=tname, data_type=ctype,
                    description=desc, is_pk=is_pk, is_fk=is_fk,
                    sample_values=samples, null_pct=null_pct,
                    references=ref_str,
                )
                columns[cname].code_values = code_values or None  # type: ignore[attr-defined]

                pk_marker = " [PK]" if is_pk else ""
                fk_marker = f" [FK→{ref_str}]" if is_fk else ""
                print(f"    {cname:<28} {ctype:<12}{pk_marker}{fk_marker}")

            self._tables[tname] = TableMeta(
                name=tname,
                description=_auto_describe_table(tname, list(columns.keys())),
                columns=columns,
            )

        print(f"\n  [_load_from_databricks] COMPLETE — {len(self._tables)} tables loaded")

    def _overlay_from_neo4j(self, cfg) -> None:
        """
        Overlay LLM-enriched descriptions and FK relationships from Neo4j.
        Falls back gracefully if Neo4j is unavailable.
        """
        print(f"\n  [_overlay_from_neo4j] Connecting to Neo4j: {cfg.neo4j_uri}")
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
        try:
            with driver.session(database=cfg.neo4j_database) as session:
                # Column descriptions
                result = session.run("""
                    MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
                    RETURN t.name AS table, c.name AS col, c.description AS desc,
                           c.sample_values AS samples
                """)
                col_overlay_count = 0
                for rec in result:
                    tbl = (rec["table"] or "").upper()
                    col = (rec["col"]   or "").upper()
                    if tbl in self._tables and col in self._tables[tbl].columns:
                        self._tables[tbl].columns[col].description = rec["desc"] or \
                            self._tables[tbl].columns[col].description
                        try:
                            sv = json.loads(rec["samples"] or "[]")
                            if sv:
                                self._tables[tbl].columns[col].sample_values = sv
                        except Exception:
                            pass
                        col_overlay_count += 1
                print(f"  [_overlay_from_neo4j] Column descriptions overlaid: {col_overlay_count}")

                # Table descriptions
                result2 = session.run("MATCH (t:Table) RETURN t.name AS name, t.description AS desc")
                tbl_overlay_count = 0
                for rec in result2:
                    tbl = (rec["name"] or "").upper()
                    if tbl in self._tables and rec["desc"]:
                        self._tables[tbl].description = rec["desc"]
                        tbl_overlay_count += 1
                print(f"  [_overlay_from_neo4j] Table descriptions overlaid: {tbl_overlay_count}")

                # FK edges from KG 
                result3 = session.run("""
                    MATCH (c1:Column)-[r:REFERENCES]->(c2:Column)
                    RETURN c1.table AS src_tbl, c1.name AS src_col,
                           c2.table AS ref_tbl, c2.name AS ref_col,
                           r.join_condition AS jc,
                           r.relationship_type AS card,
                           r.description AS desc
                """)
                kg_edges = []
                for rec in result3:
                    kg_edges.append(FKEdge(
                        src_table=(rec["src_tbl"] or "").upper(),
                        src_col=(rec["src_col"] or "").upper(),
                        ref_table=(rec["ref_tbl"] or "").upper(),
                        ref_col=(rec["ref_col"] or "").upper(),
                        join_condition=rec["jc"] or "",
                        cardinality=rec["card"] or "MANY_TO_ONE",
                        description=rec["desc"] or "",
                    ))
                if kg_edges:
                    self._fk_edges = kg_edges   # KG edges take precedence
                    print(f"  [_overlay_from_neo4j] FK edges replaced with KG edges: {len(kg_edges)}")
                    for e in kg_edges:
                        print(f"    {e.join_condition}  [{e.cardinality}]")
                else:
                    print(f"  [_overlay_from_neo4j] No KG FK edges found — keeping PRAGMA edges")

                logger.info("Neo4j overlay complete: %d tables, %d FK edges", len(self._tables), len(self._fk_edges))
                print(f"  [_overlay_from_neo4j] COMPLETE")
        finally:
            driver.close()

    def _generate_standard_aliases(self) -> None:
        """
        Mechanically derive "standard" aliases straight from the schema's own
        table/column names — no business knowledge required, so this scales
        to 100+ tables with zero manual upkeep. Covers naming-convention
        variants only:
            snake_case  ↔  space separated  ↔  concatenated  ↔  common
            generic DB-suffix expansions (_DT → date, _AMT → amount, …)

        Special abbreviations and business synonyms (e.g. "pol_id" for
        POLICY_ID, where "pol" isn't derivable from the string "POLICY_ID"
        itself) are deliberately NOT handled here — those live in the
        external aliases.json and are merged in by _load_custom_aliases().
        """
        print(f"\n  [_generate_standard_aliases] Deriving standard aliases for "
              f"{len(self._tables)} tables...")
        self._aliases = {}

        def register(term: str, target: str) -> None:
            key = term.strip().lower()
            if key and key not in self._aliases:
                self._aliases[key] = target

        for tname, tmeta in self._tables.items():
            # Table-level variants: "policy", "policys"/"policies"-ish plural, singular
            register(tname.lower(), tname)
            register(tname.lower() + "s", tname)
            if tname.lower().endswith("s"):
                register(tname.lower()[:-1], tname)

            for cname in tmeta.columns:
                target = f"{tname}.{cname}"
                register(cname.lower(), target)                       # policy_id
                register(cname.lower().replace("_", " "), target)     # policy id
                register(cname.lower().replace("_", ""), target)      # policyid
                register(f"{tname.lower()}.{cname.lower()}", target)  # policy.policy_id
                register(f"{tname.lower()}_{cname.lower()}", target)  # policy_policy_id

                # Generic DB-suffix expansion, e.g. POL_EFF_DT → "pol eff date"
                for suffix, expansion in _SUFFIX_EXPANSIONS.items():
                    if cname.upper().endswith(suffix):
                        stem = cname[: -len(suffix)].lower().replace("_", " ").strip()
                        expanded = f"{stem} {expansion}".strip()
                        register(expanded, target)
                        register(expanded.replace(" ", ""), target)
                        break

        n_generated = len(self._aliases)
        print(f"  [_generate_standard_aliases] Generated {n_generated} standard aliases")

    def _load_custom_aliases(self, path: Optional[str] = None) -> int:
        """
        Load hand-curated abbreviations / business synonyms from an external
        aliases.json and merge them on top of the auto-generated standard
        aliases. Curated entries always win on collision, since a human
        deliberately mapped e.g. "pol_id" → POLICY.POLICY_ID.

        Expected aliases.json shape::

            {
              "tables":  { "pol": "POLICY", "clm": "CLAIMS" },
              "columns": { "pol_id": "POLICY.POLICY_ID", "dob": "CLAIMANT.DOB" }
            }

        A column target may be "TABLE.COLUMN" (unambiguous) or a bare column
        name, which is resolved against whichever table(s) actually have it.
        Returns the number of custom aliases successfully loaded.
        """
        json_path = Path(path) if path else Path(__file__).with_name("aliases.json")
        print(f"\n  [_load_custom_aliases] Looking for external aliases at: {json_path}")

        if not json_path.exists():
            print(f"  [_load_custom_aliases] Not found — skipping (standard aliases only)")
            return 0

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse %s (%s) — skipping custom aliases.", json_path, e)
            print(f"  [_load_custom_aliases] Failed to parse {json_path}: {e}")
            return 0

        self._custom_aliases = {}

        for alias, table in (data.get("tables") or {}).items():
            table_u = str(table).strip().upper()
            key = str(alias).strip().lower()
            if table_u in self._tables:
                self._aliases[key] = table_u
                self._custom_aliases[key] = table_u
            else:
                logger.warning(
                    "aliases.json: table alias '%s' → '%s' does not match any known table — skipped.",
                    alias, table,
                )
                print(f"  [_load_custom_aliases] SKIP table alias '{alias}' → '{table}' (unknown table)")

        for alias, target in (data.get("columns") or {}).items():
            key = str(alias).strip().lower()
            target_str = str(target).strip()

            if "." in target_str:
                tname, cname = (p.strip().upper() for p in target_str.split(".", 1))
                if tname in self._tables and cname in self._tables[tname].columns:
                    resolved = f"{tname}.{cname}"
                else:
                    logger.warning(
                        "aliases.json: column alias '%s' → '%s' does not resolve — skipped.",
                        alias, target,
                    )
                    print(f"  [_load_custom_aliases] SKIP column alias '{alias}' → '{target}' (not found)")
                    continue
            else:
                # Bare column name — resolve against every table that has it.
                cname = target_str.upper()
                owners = self.col_to_tables.get(cname, [])
                if not owners:
                    logger.warning(
                        "aliases.json: column alias '%s' → '%s' not found in any table — skipped.",
                        alias, target,
                    )
                    print(f"  [_load_custom_aliases] SKIP column alias '{alias}' → '{target}' (unknown column)")
                    continue
                resolved = f"{owners[0]}.{cname}"

            self._aliases[key] = resolved
            self._custom_aliases[key] = resolved

        n_loaded = len(self._custom_aliases)
        print(f"  [_load_custom_aliases] Loaded {n_loaded} custom alias(es) from {json_path}")
        return n_loaded

    def _derive_domain_rules(self) -> None:
        """
        Scan status-code columns and derive domain rules from actual sample values.
        Also attaches code_values to ColumnMeta for use in concept map building.
        """
        print(f"\n  [_derive_domain_rules] Scanning all tables for code/status columns...")
        self._domain_rules = []
        rule_id = 1
        for tname, tmeta in self._tables.items():
            for cname, cmeta in tmeta.columns.items():
                if not (cname.endswith("_STAT_CD") or cname.endswith("_STATUS")
                        or cname.endswith("_CD") or cname.endswith("_FLG")
                        or cname.endswith("_TYPE_CD") or "LINE_OF" in cname):
                    continue
                if not cmeta.sample_values:
                    continue
                # Build code→label mapping from sample values
                code_values = _infer_code_values(cname, cmeta.sample_values)
                if code_values:
                    cmeta.code_values = code_values   
                    self._domain_rules.append(DomainRule(
                        rule_id=rule_id,
                        description=f"Use {tname}.{cname} for {cname.replace('_', ' ').lower()}",
                        column=f"{tname}.{cname}",
                        code_values=code_values,
                    ))
                    print(f"  [_derive_domain_rules] Rule {rule_id}: {tname}.{cname} → {code_values}")
                    rule_id += 1
        print(f"  [_derive_domain_rules] DONE — {len(self._domain_rules)} rules derived")

    def _find_hub_table(self) -> Optional[str]:
        """Heuristic: the table referenced by the most FK edges is the hub."""
        ref_counts: Dict[str, int] = {}
        for edge in self._fk_edges:
            ref_counts[edge.ref_table] = ref_counts.get(edge.ref_table, 0) + 1
        if not ref_counts:
            return next(iter(self._tables), None)
        return max(ref_counts, key=lambda k: ref_counts[k])

    def _domain_rules_for_col(self, table: str, col: str) -> Optional[DomainRule]:
        for r in self._domain_rules:
            if r.column == f"{table}.{col}":
                return r
        return None


# Helpers 

# Generic DB-naming-convention suffix expansions — these are standard,
# schema-agnostic abbreviations (not business synonyms), so they're used to
# mechanically GENERATE standard aliases rather than being hand-curated.
# e.g. POL_EFF_DT → "pol eff date", PMT_AMT_NET → "pmt amt net amount"
_SUFFIX_EXPANSIONS: Dict[str, str] = {
    "_CD": "code", "_DT": "date", "_AMT": "amount", "_NM": "name",
    "_FLG": "flag", "_NBR": "number", "_QTY": "quantity", "_DESC": "description",
    "_PCT": "percent", "_ID": "id", "_STAT": "status", "_TYPE": "type",
}

# Known insurance code decodings — consulted when inferred labels look cryptic.
_KNOWN_CODES: Dict[str, Dict[str, str]] = {
    "CLM_STAT_CD":   {"O": "Open",     "C": "Closed",    "P": "Pending",  "D": "Denied"},
    "PMT_STAT_CD":   {"IS": "Issued",  "CL": "Cleared",  "VD": "Voided",  "PD": "Paid"},
    "POL_STAT_CD":   {"AC": "Active",  "CN": "Cancelled","EX": "Expired"},
    "PMT_TYPE_CD":   {"INDEM": "Indemnity", "MED": "Medical", "EXP": "Expense"},
    "LOSS_TYPE_CD":  {"AUTO": "Auto",  "PROP": "Property", "LIAB": "Liability",
                      "WC": "Workers Comp", "MARINE": "Marine"},
    "LINE_OF_BUSNSS":{"PERSONAL_AUTO": "Personal Auto", "HOMEOWNERS": "Homeowners",
                      "COMMERCIAL": "Commercial", "WC": "Workers Comp"},
    "GENDER_CD":     {"M": "Male", "F": "Female", "U": "Unknown"},
    "ATTY_REP_FLG":  {"Y": "Yes", "N": "No"},
    "LITIGATION_FLG":{"Y": "Yes", "N": "No"},
    # ── DEV analytics tables ──
    "EVENT_TYPE_CD": {"REPORTED": "Reported", "ASSIGNED": "Assigned", "RESERVE_CHANGE": "Reserve change",
                      "PAYMENT_ISSUED": "Payment issued", "PAYMENT_VOIDED": "Payment voided",
                      "LITIGATION_FILED": "Litigation filed", "STATUS_CHANGE": "Status change",
                      "DENIED": "Denied", "CLOSED": "Closed", "REOPENED": "Reopened"},
    "FROM_STAT_CD": {"O": "Open", "C": "Closed", "P": "Pending", "D": "Denied"},
    "TO_STAT_CD":   {"O": "Open", "C": "Closed", "P": "Pending", "D": "Denied"},
    "OVERLAP_TYPE_CD": {"DUPLICATE_SUSPECT": "Duplicate suspect", "SAME_POLICY_CLAIMANT": "Same policy and claimant",
                        "SAME_POLICY": "Same policy", "SAME_CLAIMANT": "Same claimant"},
    "COLLISION_FLG": {"Y": "Yes", "N": "No"},
    "SAME_LOSS_TYPE_FLG": {"Y": "Yes", "N": "No"},
    "ANCHOR_FLG": {"Y": "Yes", "N": "No"},
    "CLUSTER_TYPE_CD": {"CAT_EVENT": "Catastrophe event", "PAYEE_RING": "Shared-payee ring",
                        "OVERLAP_GROUP": "Overlapping claims group"},
    "RISK_LEVEL_CD": {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"},
    "SEVERITY_CD": {"MINOR": "Minor", "MODERATE": "Moderate", "MAJOR": "Major", "SEVERE": "Severe"},
    "PERIL_GROUP_CD": {"WEATHER": "Weather", "FIRE": "Fire", "WATER": "Water", "THEFT": "Theft",
                       "ACCIDENT": "Accident", "LIABILITY": "Liability", "INJURY": "Injury", "TRANSIT": "Transit"},
}


def _infer_code_values(cname: str, samples: List[str]) -> Optional[Dict[str, str]]:
    """
    Return a {code: label} dict for status/flag columns, or None.

    Priority:
    1. _KNOWN_CODES lookup (covers current schema codes exactly)
    2. Heuristic: if all samples are short (≤10 chars) and there are ≤15 distinct
       values, treat them as codes and generate label = title-cased code.
    """
    col_upper = cname.upper()
    if col_upper in _KNOWN_CODES:
        return _KNOWN_CODES[col_upper]

    is_code_column = (
        col_upper.endswith("_CD")
        or col_upper.endswith("_STAT_CD")
        or col_upper.endswith("_FLG")
        or col_upper.endswith("_STATUS")
        or "LINE_OF" in col_upper
    )
    if not is_code_column or not samples:
        return None

    # Only treat as codes if values are short enumerated strings
    if all(len(s) <= 12 for s in samples) and len(samples) <= 15:
        return {s: s.replace("_", " ").title() for s in samples}
    return None


def _auto_describe(
    cname: str, tname: str, ctype: str,
    is_pk: bool, is_fk: bool, ref_str: Optional[str],
    samples: List[str],
) -> str:
    """Generate a short description from column metadata when no LLM description is available."""
    readable = cname.replace("_", " ").title()

    if is_pk:
        return f"Primary key — unique {tname.lower()} identifier."
    if is_fk and ref_str:
        ref_tbl, ref_col = ref_str.split(".")
        return f"Foreign key → {ref_tbl}.{ref_col} — links {tname.lower()} to {ref_tbl.lower()}."

    code_values = _infer_code_values(cname, samples)
    if code_values:
        codes_str = ", ".join(f"{k}={v}" for k, v in code_values.items())
        return f"{readable}. Values: {codes_str}."

    if "DT" in cname or "DATE" in cname:
        return f"Date field: {readable.lower()}. Stored as ISO text (YYYY-MM-DD)."
    if "AMT" in cname or "AMOUNT" in cname:
        return f"Monetary amount: {readable.lower()}."
    if "FLG" in cname or "FLAG" in cname:
        return f"Y/N flag: {readable.lower()}."
    if "NM" in cname or "NAME" in cname:
        return f"Name field: {readable.lower()}."
    if "ID" in cname:
        return f"Identifier field: {readable.lower()}."

    sample_str = f" Sample values: {', '.join(samples[:4])}." if samples else ""
    return f"{readable}.{sample_str}"


def _auto_describe_table(tname: str, columns: List[str]) -> str:
    return (
        f"Stores {tname.lower()} records. "
        f"Contains {len(columns)} columns: {', '.join(columns[:6])}{'…' if len(columns) > 6 else ''}."
    )