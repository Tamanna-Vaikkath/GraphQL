"""
schema_registry.py — Dynamic Schema Registry.

Single source of truth for schema metadata, domain rules, and FK relationships.
Everything is derived at runtime from SQLite (schema) and Neo4j (enriched
column descriptions + FK edges).  Nothing is hardcoded.

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


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnMeta:
    name: str
    table: str
    data_type: str
    description: str               # LLM-enriched (from Neo4j) or raw pragma fallback
    is_pk: bool
    is_fk: bool
    sample_values: List[str]
    null_pct: float
    references: Optional[str] = None   # "REFTABLE.REFCOL" when is_fk=True


@dataclass
class TableMeta:
    name: str
    description: str               # LLM-enriched (from Neo4j) or auto-generated
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
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

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
        print(f"\n{'='*60}")
        print(f"[SchemaRegistry.load] START")
        print(f"  sqlite_db_path : {cfg.sqlite_db_path}")
        print(f"  neo4j_uri      : {cfg.neo4j_uri}")
        print(f"{'='*60}")

        reg = cls()

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

        print(f"\n[SchemaRegistry.load] Step 3/3 — Deriving domain rules...")
        reg._derive_domain_rules()
        print(f"[SchemaRegistry.load] Domain rules derived → {len(reg._domain_rules)} rules")

        reg._loaded = True
        print(f"\n[SchemaRegistry.load] DONE")
        print(f"  Tables       : {reg.table_names}")
        print(f"  FK edges     : {len(reg._fk_edges)}")
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
                    break  # Only emit once

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

        # Date format
        lines.append(f"{rule_num}. Dates are stored as ISO strings (YYYY-MM-DD); use date() for SQLite date math.")
        rule_num += 1
        lines.append(f"{rule_num}. Generate SQLite-compatible SQL only (no Oracle syntax like ROWNUM or SYSDATE).")

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

                # status-code columns: add decoded values as concepts
                if cmeta.code_values:
                    for code, label in cmeta.code_values.items():
                        add(label.lower(), col_cols)
                        add(label.lower() + " " + tl[:-1] if tl.endswith("s") else label.lower() + " " + tl, col_cols)

                # FK columns: add parent table name
                if cmeta.is_fk and cmeta.references:
                    ref_tbl = cmeta.references.split(".")[0].lower()
                    add(ref_tbl, [cname])

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

        if not examples:
            # Minimal fallback
            first_table = next(iter(self._tables), "TABLE1")
            examples.append(
                f"Q: Show all records\nSQL:\nSELECT * FROM {first_table} LIMIT 100;"
            )

        print(f"  [build_fewshot_examples] Generated {len(examples)} example(s)")
        return "\nEXAMPLES:\n\n" + "\n\n".join(examples)

    # ── Private loaders ───────────────────────────────────────────────────────

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
                        columns[cname].code_values = code_values   # type: ignore[attr-defined]
                    else:
                        columns[cname].code_values = None           # type: ignore[attr-defined]

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

                # FK edges from KG (may be richer than PRAGMA)
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
                    cmeta.code_values = code_values   # type: ignore[attr-defined]
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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