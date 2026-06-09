"""
pipeline/sql_generator.py — Stage 4: SQL Generation.

Assembles the structured prompt (top-25 columns + join paths + domain rules +
few-shot examples) and calls GPT to produce a SQLite-compatible SQL query.

"""
from __future__ import annotations
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from config import Config
from kg.retriever import RetrievalResult
from utils.llm_client import get_gpt_client, chat_complete
from pipeline.hyde_expander import SCHEMA_MANIFEST, _ALL_COLUMNS


class SQLColumnNotFoundError(Exception):
    """
    Raised when the SQL generator produces a query referencing column(s) that
    do not exist in the ground-truth schema manifest.

    Attributes
    ----------
    invalid_refs : list of str
        Each entry is a "TABLE.COLUMN" string that failed validation.
    suggestion : str or None
        Human-readable hint for the user.
    """
    def __init__(self, invalid_refs: List[str]) -> None:
        self.invalid_refs = invalid_refs
        detail = (
            f"Your question references data that is not stored in this database. "
            f"The following column(s) do not exist in the schema: "
            f"{', '.join(invalid_refs)}. "
            f"Please rephrase your question using available data fields."
        )
        super().__init__(detail)


def _validate_sql_columns(sql: str) -> List[str]:
    """
    Extract all TABLE.COLUMN references from *sql* and check each against the
    ground-truth SCHEMA_MANIFEST.

    Returns a list of invalid "TABLE.COLUMN" strings (empty list = all valid).

    This is purely deterministic — no LLM call, O(n) in SQL length.
    Only TABLE.COLUMN dot-notation is checked; bare column names in SELECT
    without a table prefix are skipped (the DB engine will catch those).
    """
    invalid: List[str] = []
    # Match TABLE.COLUMN or alias.COLUMN patterns in the SQL
    for m in re.finditer(
        r'\b([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\b',
        sql.upper(),
    ):
        tbl_or_alias, col = m.group(1), m.group(2)
        # Skip if tbl_or_alias is a known SQL keyword that can precede a dot
        if tbl_or_alias in _SQL_KEYWORDS:
            continue
        # Only validate if the left side matches a known table name exactly
        # (aliases like "c", "p" are skipped)
        if tbl_or_alias not in SCHEMA_MANIFEST:
            continue
        if col not in SCHEMA_MANIFEST[tbl_or_alias]:
            ref = f"{tbl_or_alias}.{col}"
            if ref not in invalid:
                invalid.append(ref)
    return invalid

logger = logging.getLogger(__name__)

# ── Domain Rules ───────────────────────────────
DOMAIN_RULES = """
DOMAIN RULES (always follow):
1. Use CLM_STAT_CD from CLAIMS for claim status: O=Open, C=Closed, P=Pending, D=Denied
2. Use PMT_STAT_CD from PAYMENT for payment status: IS=Issued, CL=Cleared, VD=Voided, PD=Pending
3. Use CLAIMANT.STATE_CD for claimant's state of residence (not POLICY.STATE_CD)
4. Use POLICY.STATE_CD for the policy-writing state
5. Use INCURRED_AMT for total claim cost; RESERVE_AMT for outstanding reserves only
6. For "open claims" always filter CLM_STAT_CD = 'O'; for "pending" use 'P'
7. Payment amounts: PMT_AMT_GROSS is before deductible; PMT_AMT_NET is after
8. ATTY_REP_FLG='Y' means represented by attorney; 'N' means not represented
9. LITIGATION_FLG='Y' means the claim is in active litigation
10. FRAUD_RISK_SCRE is 0-100; higher = more risk. Use > 75 for high risk
11. Always use LEFT JOIN for PAYMENT unless explicitly filtering on payment columns
12. Dates are stored as ISO strings (YYYY-MM-DD); use date() for SQLite date math
13. Generate SQLite-compatible SQL only (no Oracle syntax like ROWNUM or SYSDATE)
""".strip()

# ── Few-shot examples ─────────────────────────────────────────────────────────
FEW_SHOT = """
EXAMPLES:

Q: Show all open claims where no payment has been issued
SQL:
SELECT c.CLAIM_ID, c.CLM_STAT_CD, c.LOSS_DT, c.INCURRED_AMT, c.RESERVE_AMT, c.ADJUSTER_ID
FROM CLAIMS c
LEFT JOIN PAYMENT p ON p.CLAIM_ID = c.CLAIM_ID
WHERE c.CLM_STAT_CD = 'O' AND p.PAYMENT_ID IS NULL
ORDER BY c.LOSS_DT ASC;

Q: List claimants with fraud risk above 75 who have filed more than 2 claims and are represented by an attorney
SQL:
SELECT CLAIMANT_ID, CLAIMANT_NM, FRAUD_RISK_SCRE, CLAIM_COUNT, STATE_CD
FROM CLAIMANT
WHERE FRAUD_RISK_SCRE > 75 AND CLAIM_COUNT > 2 AND ATTY_REP_FLG = 'Y'
ORDER BY FRAUD_RISK_SCRE DESC;

Q: Total incurred amount by line of business
SQL:
SELECT p.LINE_OF_BUSNSS, COUNT(c.CLAIM_ID) AS claim_count, SUM(c.INCURRED_AMT) AS total_incurred
FROM CLAIMS c
JOIN POLICY p ON c.POLICY_ID = p.POLICY_ID
GROUP BY p.LINE_OF_BUSNSS
ORDER BY total_incurred DESC;
""".strip()

SQL_SYSTEM = f"""You are a precise SQL generator for a P&C insurance database.
Generate SQLite-compatible SQL only. Return ONLY the SQL statement, no explanation, no markdown fences.

{DOMAIN_RULES}

{FEW_SHOT}
"""


# ── Join pruning (post-generation) ────────────────────────────────────────────
_SQL_KEYWORDS = frozenset({
    "SELECT", "WHERE", "ON", "SET", "VALUES", "INTO", "WITH",
    "LATERAL", "UNNEST", "DUAL", "TABLE", "AS", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL",
    "NATURAL", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "BETWEEN", "LIKE", "CASE", "WHEN", "THEN", "ELSE", "END",
    "EXISTS", "DISTINCT", "ALL", "UNION", "INTERSECT", "EXCEPT",
    "LIMIT", "OFFSET", "ORDER", "GROUP", "BY", "HAVING", "ASC", "DESC",
})

# Matches a single JOIN clause including its ON condition.
# Captures: (join_type  table_name  optional_alias  on_clause)
_JOIN_CLAUSE_RE = re.compile(
    r'(?P<jointype>'
    r'(?:LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|INNER\s+|CROSS\s+|FULL\s+(?:OUTER\s+)?)?'
    r'JOIN\s+)'
    r'(?P<table>[A-Z_][A-Z0-9_$#]*)'        # table name
    r'(?:\s+(?:AS\s+)?(?P<alias>[A-Z_][A-Z0-9_$#]*))?'   # optional alias
    r'(?:\s+ON\s+(?P<on_clause>.+?))?'       
    r'(?=\s+(?:LEFT|RIGHT|INNER|CROSS|FULL|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|$))',
    re.IGNORECASE | re.DOTALL,
)


def _tables_and_aliases_in_sql(sql: str) -> tuple[dict[str, str], str]:
    """
    Return (alias_map, from_table) where alias_map is {alias_upper: table_upper}
    and from_table is the primary table in the FROM clause (uppercased).
    """
    alias_map: dict[str, str] = {}
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
        tbl = m.group("table").upper()
        alias = (m.group("alias") or "").upper()
        if alias and alias not in _SQL_KEYWORDS and alias != tbl:
            alias_map[alias] = tbl

    return alias_map, from_table


def _column_references_outside_join(sql: str) -> set[str]:
    """
    Return the set of *table names* (uppercased, aliases resolved later)
    that are explicitly referenced by ALIAS.COLUMN notation anywhere in the
    SQL *other than* inside JOIN … ON clauses.

    We strip the ON clauses first so we only count real SELECT / WHERE / GROUP
    BY / HAVING / ORDER BY usage — not the join predicate itself.
    """
    
    stripped = re.sub(
        r'\bON\s+.+?(?=\s+(?:LEFT|RIGHT|INNER|CROSS|FULL|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT)|$)',
        ' ',
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    refs: set[str] = set()
    for m in re.finditer(r'\b([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)\b',
                         stripped.upper()):
        refs.add(m.group(1))
    return refs


def prune_unnecessary_joins(sql: str) -> str:
    """
    Remove JOIN clauses whose joined table contributes no columns to the
    SELECT list, WHERE filter, GROUP BY, HAVING, or ORDER BY clause.

    A table is considered *contributing* when at least one ALIAS.COLUMN (or
    TABLE.COLUMN) reference that resolves to that table appears outside of
    the ON predicate of that join itself.  LEFT JOINs that are present only
    for their ON predicate (e.g. existence checks written as LEFT JOIN … IS
    NULL) are preserved because the IS-NULL test on the join key does count
    as a WHERE-level usage of the joined table.

    Algorithm
    ---------
    1. Parse alias → table mapping from FROM + all JOIN clauses.
    2. Collect ALIAS references that appear *outside* ON clauses.
    3. Resolve those aliases to canonical table names → contributing set.
    4. Walk JOIN clauses; drop any whose table is not in the contributing set
       AND whose ON clause does not reference a non-contributing alias on the
       *other* side that is itself used (avoids breaking chain joins).
    5. Re-stitch the SQL from FROM table onwards with surviving JOINs.
    """
    if not sql or not sql.strip():
        return sql

    alias_map, from_table = _tables_and_aliases_in_sql(sql)
    raw_refs = _column_references_outside_join(sql)

    # Resolve raw alias/table refs → canonical table names
    contributing: set[str] = set()
    for ref in raw_refs:
        resolved = alias_map.get(ref, ref)
        contributing.add(resolved)
    # The primary FROM table always counts
    if from_table:
        contributing.add(from_table)

    # Collect JOIN clauses in order
    joins = list(_JOIN_CLAUSE_RE.finditer(sql))
    if not joins:
        return sql  # no JOINs to prune

    # Decide which JOINs to keep
    surviving_joins: list[re.Match] = []
    for m in joins:
        tbl = m.group("table").upper()
        if tbl in contributing:
            surviving_joins.append(m)
        else:
            logger.debug("prune_unnecessary_joins: dropping JOIN %s (no column refs)", tbl)

    if len(surviving_joins) == len(joins):
        return sql  

    # Re-assemble: keep everything up to the first JOIN, insert surviving JOINs,
    # then append everything after the last JOIN (WHERE, GROUP BY, etc.)
    first_join_start = joins[0].start()
    last_join_end    = joins[-1].end()

    prefix  = sql[:first_join_start].rstrip()
    suffix  = sql[last_join_end:].lstrip()
    middle  = "\n".join(m.group(0).strip() for m in surviving_joins)

    pruned = f"{prefix}\n{middle}\n{suffix}" if middle else f"{prefix}\n{suffix}"
    # Collapse any run of blank lines introduced by removal
    pruned = re.sub(r'\n{3,}', '\n\n', pruned).strip()
    logger.info(
        "prune_unnecessary_joins: %d → %d JOINs after pruning",
        len(joins), len(surviving_joins),
    )
    return pruned


class SQLGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)

    def generate(self, question: str, retrieval: RetrievalResult) -> str:
        """Generate SQL from the question and retrieval result.

        Args:
            question: Original user question.
            retrieval: RetrievalResult with top-25 columns and join paths.
        Returns:
            Raw SQL string.
        """
        columns_block = self._format_columns(retrieval)
        joins_block = "\n".join(retrieval.join_conditions) if retrieval.join_conditions else "None identified"

        user_prompt = (
            f"Relevant columns (top {len(retrieval.columns)}):\n{columns_block}\n\n"
            f"Available JOIN conditions:\n{joins_block}\n\n"
            f"Question: {question}"
        )

        sql = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            SQL_SYSTEM, user_prompt,
            temperature=0.0, max_tokens=800
        )
        # Strip markdown fences if model adds them
        sql = sql.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()

        # ── Prune JOIN clauses for tables that contribute nothing ─────────────
        # Any table whose columns never appear outside its own ON predicate
        # (i.e. not in SELECT, WHERE, GROUP BY, HAVING, ORDER BY) is removed.
        sql = prune_unnecessary_joins(sql)

        # ── Post-generation schema validation ─────────────────────────────────
        # Check every TABLE.COLUMN reference in the generated SQL against the
        # ground-truth manifest. Raises SQLColumnNotFoundError immediately if any
        # hallucinated or non-existent column slips through, stopping execution
        # before a confusing SQLite error reaches the user.
        invalid_refs = _validate_sql_columns(sql)
        if invalid_refs:
            logger.warning(
                "SQL column validation failed — invalid refs: %s\nSQL:\n%s",
                invalid_refs, sql,
            )
            raise SQLColumnNotFoundError(invalid_refs)

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