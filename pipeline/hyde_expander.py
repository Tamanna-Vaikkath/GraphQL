"""
pipeline/hyde_expander.py — HyDE (Hypothetical Document Embedding) query expansion.

Produces a short, intent-focused hypothetical passage from the user's natural-
language question.  The passage is embedded and used for semantic retrieval
against the schema knowledge graph.

SCALABILITY CHANGES
-------------------
All hardcoded artifacts have been removed:

  BEFORE (hardcoded)                       AFTER (registry-derived)
  ─────────────────────────────────────    ──────────────────────────────────────
  SCHEMA_MANIFEST dict (4 tables)          registry.manifest           (any tables)
  _ALL_COLUMNS frozenset                   registry.all_columns        (any columns)
  _COL_TO_TABLES dict                      registry.col_to_tables      (any tables)
  _CONCEPT_TO_COLUMNS dict (hand-written)  registry.build_concept_map()  (auto-built)
  _UNMAPPABLE_INSURANCE_CONCEPTS set       LLM concept coverage check  (no static list)
  _SCHEMA_BLOCK string                     registry.build_schema_block() (auto-rendered)

Usage
-----
The module can be used in two modes:

1. Registry-injected (recommended, scalable):
       registry = SchemaRegistry.load(cfg)
       expander = LLMHyDEExpander(llm_client, registry=registry)

2. Legacy / standalone (backward-compat, falls back to module-level vars):
       expander = LLMHyDEExpander(llm_client)
   In this mode SCHEMA_MANIFEST, _ALL_COLUMNS, and _CONCEPT_TO_COLUMNS are
   still exported for backward-compat imports but they are derived from the
   registry at first use rather than hardcoded.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Protocol, Set, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# ── Schema grounding error ────────────────────────────────────────────────────

class SchemaGroundingError(Exception):
    """
    Raised by ``LLMHyDEExpander.expand_or_raise`` when the user's question
    cannot be grounded to any column in the schema.

    Attributes
    ----------
    question:       The original user question.
    unknown_tokens: Column-like tokens the LLM invented that don't exist.
    suggestion:     Optional hint pointing to the closest real column(s).
    """

    def __init__(
        self,
        question: str,
        unknown_tokens: List[str],
        suggestion: Optional[str] = None,
        table_names: Optional[List[str]] = None,
    ) -> None:
        self.question = question
        self.unknown_tokens = unknown_tokens
        self.suggestion = suggestion
        tables_str = ", ".join(table_names) if table_names else "the database schema"
        if unknown_tokens:
            tokens_str = ", ".join(f"'{t}'" for t in unknown_tokens)
            detail = (
                f"The following concept(s) in your question have no corresponding column "
                f"in {tables_str}: {tokens_str}. "
                f"The database cannot answer this question because the required data is not stored."
            )
        else:
            detail = (
                f"Your question does not correspond to any column in {tables_str}. "
                "The database cannot answer this question because the required data is not stored."
            )
        super().__init__(detail)


# ── Prompt strategies ─────────────────────────────────────────────────────────

class PromptStrategy(str, Enum):
    CONCISE           = "concise"
    VERBOSE           = "verbose"
    DOMAIN_CODES      = "domain_codes"
    ANTI_HALLUCINATION = "anti_hallucination"


_PROMPT_TEMPLATES: Dict[PromptStrategy, str] = {

    PromptStrategy.CONCISE: (
        "You are a database analyst. Given a natural-language question, "
        "write exactly 1–2 sentences that directly mirror the question's intent "
        "using precise domain terminology and the EXACT column names, "
        "table names, and code values from the schema below.\n\n"
        "{schema_block}\n\n"
        "Rules:\n"
        "- Use ONLY column names listed above — no others.\n"
        "- Name the exact tables and columns the query would touch.\n"
        "- ONLY include a status code filter if the user explicitly asks for that status.\n"
        "- Do NOT generate SQL.\n"
        "- Do NOT add any filter the user did not request."
    ),

    PromptStrategy.VERBOSE: (
        "You are a database analyst. Given a natural-language question, write 3–4 sentences "
        "describing the data required to answer it, referencing ONLY the tables and columns "
        "from the schema below.\n\n"
        "{schema_block}\n\n"
        "Rules:\n"
        "- Use ONLY column names listed above — no others.\n"
        "- Mention every table that would be needed (joins included).\n"
        "- Do NOT generate SQL.\n"
        "- Do NOT invent column names."
    ),

    PromptStrategy.DOMAIN_CODES: (
        "You are a database data dictionary. Given a natural-language question, "
        "produce 1–2 sentences that enumerate exactly which columns and code values "
        "satisfy the question, using ONLY columns from the schema below.\n\n"
        "{schema_block}\n\n"
        "Rules:\n"
        "- Lead with the primary table(s) and filter column + value ONLY if the user asked for that status.\n"
        "- List any FK joins needed.\n"
        "- Use ONLY column names listed above.\n"
        "- Do NOT generate SQL."
    ),

    PromptStrategy.ANTI_HALLUCINATION: (
        "You are a strict database schema validator. Given a question, produce "
        "1–2 sentences describing the data needed, using ONLY the column names "
        "listed in the schema below.\n\n"
        "{schema_block}\n\n"
        "Absolute rules:\n"
        "1. Every column token MUST appear verbatim in the schema above.\n"
        "2. If the question mentions a concept with no matching column, say "
        "'this concept has no corresponding column in the schema'.\n"
        "3. Do NOT generate SQL.\n"
        "4. Do NOT invent, abbreviate, or shorten column names."
    ),
}


# ── Validation report ─────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    is_clean: bool
    unknown_tokens: List[str] = field(default_factory=list)
    known_tokens: List[str] = field(default_factory=list)
    cleaned_passage: str = ""
    warnings: List[str] = field(default_factory=list)
    has_schema_coverage: bool = True


# ── Schema validator ──────────────────────────────────────────────────────────

class SchemaValidator:
    """
    Scans a HyDE passage for column-like tokens and validates against a
    given column set.  Accepts the column set as a constructor argument so
    it works with any schema size — not just the original 4-table schema.
    """

    _COL_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[A-Z]{4,})\b')

    def __init__(self, all_columns: FrozenSet[str]) -> None:
        self._all_columns = all_columns

    def validate(self, passage: str) -> ValidationReport:
        if not passage:
            return ValidationReport(is_clean=True, cleaned_passage=passage)

        candidates: Set[str] = set(self._COL_PATTERN.findall(passage))
        unknown: List[str] = []
        known: List[str] = []

        for token in sorted(candidates):
            if token in self._all_columns:
                known.append(token)
            elif "_" in token:
                unknown.append(token)

        warnings: List[str] = []
        cleaned = passage

        for token in unknown:
            suggestion = _suggest_closest(token, self._all_columns)
            msg = f"Column '{token}' does NOT exist in the schema."
            if suggestion:
                msg += f" Closest known column: '{suggestion}'."
            warnings.append(msg)
            logger.warning("HyDE hallucination guard: %s", msg)
            cleaned = re.sub(rf'\b{re.escape(token)}\b', '', cleaned)

        cleaned = re.sub(r'  +', ' ', cleaned).strip()

        return ValidationReport(
            is_clean=len(unknown) == 0,
            unknown_tokens=unknown,
            known_tokens=known,
            cleaned_passage=cleaned,
            warnings=warnings,
            has_schema_coverage=len(known) > 0,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _suggest_closest(token: str, all_columns: FrozenSet[str]) -> Optional[str]:
    token_upper = token.upper()
    best: Optional[str] = None
    best_len = 0
    for col in all_columns:
        common = 0
        for a, b in zip(token_upper, col):
            if a == b:
                common += 1
            else:
                break
        if common > best_len:
            best_len = common
            best = col
    return best if best_len >= 4 else None


def trim_hyde(expanded: str, max_sentences: int = 2) -> str:
    if not expanded:
        return expanded
    sentences = re.split(r'(?<=[.!?])\s+', expanded.strip())
    return " ".join(sentences[:max_sentences]) or expanded


# ── Expander interface ────────────────────────────────────────────────────────

@runtime_checkable
class HyDEExpander(Protocol):
    def expand(self, question: str) -> str: ...
    def expand_with_report(self, question: str) -> Tuple[str, ValidationReport]: ...


# ── Main expander ─────────────────────────────────────────────────────────────

class LLMHyDEExpander:
    """
    Schema-grounded HyDE expander.

    Parameters
    ----------
    llm_client:
        Any object with a ``complete(prompt: str) -> str`` method.
    registry:
        A ``SchemaRegistry`` instance.  When provided, the schema block,
        concept map, and column sets are derived from it dynamically — no
        hardcoded manifests.  When None, falls back to the module-level
        legacy vars (backward-compat for tests / migration period).
    max_sentences:
        Sentences to retain after trimming (default: 2).
    strategy:
        ``PromptStrategy`` variant (default: CONCISE).
    strict_validation:
        Strip hallucinated column tokens before embedding (default: True).
    """

    def __init__(
        self,
        llm_client,
        registry=None,
        max_sentences: int = 2,
        strategy: PromptStrategy = PromptStrategy.CONCISE,
        custom_system_prompt: Optional[str] = None,
        strict_validation: bool = True,
    ) -> None:
        self._client = llm_client
        self._max_sentences = max_sentences
        self._strategy = strategy
        self._strict = strict_validation
        self._registry = registry

        print(f"\n[LLMHyDEExpander INIT]")
        print(f"  strategy         : {strategy.value}")
        print(f"  max_sentences    : {max_sentences}")
        print(f"  strict_validation: {strict_validation}")
        print(f"  registry_present : {registry is not None}")

        # Resolve schema data — registry-first, then legacy module-level vars
        if registry is not None:
            self._schema_block = registry.build_schema_block()
            self._all_columns: FrozenSet[str] = registry.all_columns
            self._manifest: Dict[str, Dict[str, str]] = registry.manifest
            self._concept_map: Dict[str, List[str]] = registry.build_concept_map()
            self._table_names: List[str] = registry.table_names
            print(f"  schema source    : registry ({len(self._table_names)} tables, {len(self._all_columns)} columns)")
        else:
            # Backward-compat: use module-level legacy vars
            # (populated below at module level for standalone use)
            self._schema_block = _get_legacy_schema_block()
            self._all_columns = _get_legacy_all_columns()
            self._manifest = _get_legacy_manifest()
            self._concept_map = _get_legacy_concept_map()
            self._table_names = list(_get_legacy_manifest().keys())
            print(f"  schema source    : legacy fallback ({len(self._table_names)} tables)")

        self._validator = SchemaValidator(self._all_columns)

        if custom_system_prompt is not None:
            self._system = (
                custom_system_prompt.format(schema_block=self._schema_block)
                if "{schema_block}" in custom_system_prompt
                else custom_system_prompt
            )
        else:
            template = _PROMPT_TEMPLATES[strategy]
            self._system = template.format(schema_block=self._schema_block)

    # ── Public API ────────────────────────────────────────────────────────────

    def expand(self, question: str) -> str:
        passage, _ = self.expand_with_report(question)
        return passage

    def expand_with_report(self, question: str) -> Tuple[str, ValidationReport]:
        print(f"\n[HyDE expand_with_report] INPUT question: {repr(question)}")
        print(f"  strategy={self._strategy.value}, strict={self._strict}, max_sentences={self._max_sentences}")
        prompt = f"Question: {question}"
        raw: str = self._client.complete(self._system + "\n\n" + prompt)
        print(f"[HyDE expand_with_report] Raw LLM output: {repr(raw)}")
        report = self._validator.validate(raw)
        print(f"[HyDE expand_with_report] Validation → is_clean={report.is_clean}, known={report.known_tokens}, unknown={report.unknown_tokens}")
        base_passage = report.cleaned_passage if self._strict else raw
        trimmed = trim_hyde(base_passage, max_sentences=self._max_sentences)
        report.cleaned_passage = trimmed
        if not report.is_clean:
            logger.warning(
                "HyDE passage for %r had %d unknown token(s): %s",
                question, len(report.unknown_tokens), report.unknown_tokens,
            )
            print(f"[HyDE expand_with_report] WARNING — {len(report.unknown_tokens)} unknown token(s) stripped: {report.unknown_tokens}")
        print(f"[HyDE expand_with_report] OUTPUT passage: {repr(trimmed)}")
        return trimmed, report

    def expand_or_raise(self, question: str) -> str:
        print(f"\n[HyDE expand_or_raise] INPUT question: {repr(question)}")

        # ── Layer 0: deterministic concept precheck ───────────────────────────
        print(f"[HyDE expand_or_raise] Layer 0: ontology precheck...")
        non_db = self._ontology_precheck(question)
        if non_db is not None:
            print(f"[HyDE expand_or_raise] Layer 0 FAILED — unmappable concepts: {non_db}")
            raise SchemaGroundingError(
                question=question,
                unknown_tokens=non_db,
                suggestion=_suggest_closest(non_db[0], self._all_columns) if non_db else None,
                table_names=self._table_names,
            )
        print(f"[HyDE expand_or_raise] Layer 0 PASSED")

        passage, report = self.expand_with_report(question)

        # ── Layer 1: token presence check ────────────────────────────────────
        print(f"[HyDE expand_or_raise] Layer 1: schema coverage check → has_schema_coverage={report.has_schema_coverage}")
        if not report.has_schema_coverage:
            question_upper = question.upper()
            direct_hit = (
                any(col in question_upper for col in self._all_columns)
                or any(tbl in question_upper for tbl in self._table_names)
            )
            print(f"[HyDE expand_or_raise] Layer 1: direct_hit={direct_hit}")
            if not direct_hit:
                print(f"[HyDE expand_or_raise] Layer 1 FAILED — no schema coverage, no direct hit")
                suggestion: Optional[str] = None
                if report.unknown_tokens:
                    suggestion = _suggest_closest(report.unknown_tokens[0], self._all_columns)
                raise SchemaGroundingError(
                    question=question,
                    unknown_tokens=report.unknown_tokens,
                    suggestion=suggestion,
                    table_names=self._table_names,
                )
        print(f"[HyDE expand_or_raise] Layer 1 PASSED")

        # ── Layer 2: LLM concept coverage check ──────────────────────────────
        print(f"[HyDE expand_or_raise] Layer 2: LLM concept coverage check...")
        unmapped = self._check_concept_coverage(question)
        if unmapped:
            print(f"[HyDE expand_or_raise] Layer 2 FAILED — unmapped concepts: {unmapped}")
            raise SchemaGroundingError(
                question=question,
                unknown_tokens=unmapped,
                suggestion=_suggest_closest(unmapped[0], self._all_columns) if unmapped else None,
                table_names=self._table_names,
            )
        print(f"[HyDE expand_or_raise] Layer 2 PASSED")
        print(f"[HyDE expand_or_raise] OUTPUT passage: {repr(passage)}")
        return passage

    def set_strategy(self, strategy: PromptStrategy) -> None:
        self._strategy = strategy
        template = _PROMPT_TEMPLATES[strategy]
        self._system = template.format(schema_block=self._schema_block)
        logger.info("HyDE prompt strategy switched to: %s", strategy.value)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ontology_precheck(self, question: str) -> Optional[List[str]]:
        """
        Deterministic pre-LLM check.
        Returns None  → question looks answerable; proceed.
        Returns []    → off-topic / no domain signal.
        Returns [str] → specific concepts with no DB column.
        """
        print(f"  [_ontology_precheck] question: {repr(question)}")
        q_lower = question.lower()
        q_upper = question.upper()

        has_db_concept   = any(c in q_lower for c in self._concept_map)
        has_direct_col   = any(col in q_upper for col in self._all_columns)
        has_direct_table = any(tbl in q_upper for tbl in self._table_names)

        print(f"  [_ontology_precheck] has_db_concept={has_db_concept}, has_direct_col={has_direct_col}, has_direct_table={has_direct_table}")

        if has_direct_col or has_direct_table:
            print(f"  [_ontology_precheck] → PASS (direct schema reference found)")
            return None   # explicit schema reference → pass through

        if has_db_concept:
            print(f"  [_ontology_precheck] → PASS (concept map hit)")
            return None   # concept map hit → let LLM layers verify further

        # No schema signal at all → off-topic; let LLM concept check handle it
        print(f"  [_ontology_precheck] → PASS (no signal — deferring to LLM layers)")
        return None

    def _check_concept_coverage(self, question: str) -> List[str]:
        """
        Ask the LLM to identify queryable concepts and verify each against
        the schema.  Returns unmapped concepts (empty list = all covered).
        """
        print(f"\n  [_check_concept_coverage] question: {repr(question)}")
        q_lower = question.lower()

        # Deterministic short-circuit: if every meaningful term already matches
        # a concept in the auto-built concept map, skip the LLM call.
        matched_concepts = [c for c in self._concept_map if c in q_lower]
        print(f"  [_check_concept_coverage] Matched concept map entries: {matched_concepts[:10]}")
        if matched_concepts:
            _STOP_WORDS = frozenset({
                "show", "list", "get", "find", "display", "give", "me", "all",
                "the", "a", "an", "for", "of", "with", "and", "or", "in", "on",
                "to", "from", "by", "where", "that", "which", "are", "is", "be",
                "has", "have", "had", "do", "does", "did", "their", "its",
                "any", "some", "no", "not", "at", "as", "up", "out",
            })
            covered_text = " ".join(matched_concepts)
            remaining = [
                w for w in re.findall(r"[a-z]+", q_lower)
                if w not in _STOP_WORDS and w not in covered_text
            ]
            print(f"  [_check_concept_coverage] Remaining uncovered words: {remaining}")
            if not remaining:
                logger.info("Concept coverage: all terms matched concept map — skipping LLM check.")
                print(f"  [_check_concept_coverage] All terms matched — skipping LLM call")
                return []

        print(f"  [_check_concept_coverage] Invoking LLM for concept coverage check...")
        coverage_system = (
            "You are a strict database schema validator.\n\n"
            + self._schema_block
            + "\n\n"
            "Given a user question, identify every distinct *queryable concept* it is asking "
            "about. For each concept, determine whether there is at least one column in the "
            "schema above that can represent it WITH SUFFICIENT PRECISION.\n\n"
            "Rules:\n"
            "1. A concept is 'covered' ONLY if a real column stores or encodes that information precisely.\n"
            "2. Generic filter words ('show', 'list', 'related to') are NOT concepts.\n"
            "3. Return ONLY a JSON object with two keys:\n"
            "   - \"covered\": list of concept strings that ARE in the schema\n"
            "   - \"unmapped\": list of concept strings that have NO column representation\n"
            "4. Output only the JSON object — no explanation, no markdown fences."
        )
        user_prompt = f"Question: {question}"
        try:
            raw = self._client.complete(coverage_system + "\n\n" + user_prompt)
            print(f"  [_check_concept_coverage] LLM raw response: {repr(raw[:300])}")
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = __import__("json").loads(raw[start:end])
            unmapped: List[str] = parsed.get("unmapped", [])
            print(f"  [_check_concept_coverage] covered={parsed.get('covered', [])}, unmapped={unmapped}")
            logger.info(
                "Concept coverage for %r — covered=%s unmapped=%s",
                question, parsed.get("covered", []), unmapped,
            )
            return [str(u) for u in unmapped] if unmapped else []
        except Exception as e:
            logger.warning("Concept coverage check failed (%s) — allowing pipeline to proceed.", e)
            print(f"  [_check_concept_coverage] LLM check FAILED ({e}) — proceeding anyway")
            return []


# ── Legacy backward-compatibility shims ──────────────────────────────────────
# These are populated lazily the first time code imports
#   from pipeline.hyde_expander import SCHEMA_MANIFEST, _ALL_COLUMNS
# If a registry is never injected, they fall back to a minimal auto-built set
# derived from _BUILTIN_FALLBACK_SCHEMA (the original 4-table schema).
# In production, always inject a registry; these are only for unit tests and
# the migration period.

_BUILTIN_FALLBACK_SCHEMA: Dict[str, Dict[str, str]] = {
    "CLAIMS": {
        "CLAIM_ID": "Primary key.", "POLICY_ID": "FK→POLICY.",
        "CLAIMANT_ID": "FK→CLAIMANT.", "CLM_STAT_CD": "O=Open,C=Closed,P=Pending,D=Denied.",
        "LOSS_DT": "Loss date.", "REPORT_DT": "Report date.",
        "LOSS_TYPE_CD": "AUTO/PROP/LIAB/WC/MARINE.", "INCURRED_AMT": "Total incurred.",
        "RESERVE_AMT": "Reserve.", "ADJUSTER_ID": "Adjuster ID.",
        "CLOSE_DT": "Close date.", "LITIGATION_FLG": "Y/N litigation.",
    },
    "POLICY": {
        "POLICY_ID": "Primary key.", "POLICY_NBR": "Policy number.",
        "INSURED_NM": "Insured name.", "POL_EFF_DT": "Effective date.",
        "POL_EXP_DT": "Expiry date.", "LINE_OF_BUSNSS": "Line of business.",
        "STATE_CD": "State code.", "PREMIUM_AMT": "Premium.", "DEDUCTIBLE_AMT": "Deductible.",
        "AGENT_ID": "Agent ID.", "POL_STAT_CD": "AC=Active,CN=Cancelled,EX=Expired.",
    },
    "PAYMENT": {
        "PAYMENT_ID": "Primary key.", "CLAIM_ID": "FK→CLAIMS.",
        "PMT_DT": "Payment date.", "PMT_AMT_GROSS": "Gross amount.", "PMT_AMT_NET": "Net amount.",
        "PMT_STAT_CD": "IS=Issued,CL=Cleared,VD=Voided,PD=Paid.", "PMT_TYPE_CD": "INDEM/MED/EXP.",
        "PAYEE_NM": "Payee name.", "CHK_NBR": "Check number.", "VOID_RSN_CD": "Void reason.",
    },
    "CLAIMANT": {
        "CLAIMANT_ID": "Primary key.", "CLAIMANT_NM": "Claimant name.", "DOB": "Date of birth.",
        "GENDER_CD": "M/F/U.", "ADDRESS_LINE1": "Address.", "STATE_CD": "State.",
        "CONTACT_PHONE": "Phone.", "ATTY_REP_FLG": "Y/N attorney.", "CLAIM_COUNT": "Claim count.",
        "FRAUD_RISK_SCRE": "Fraud score 0-100.",
    },
}


def _get_legacy_manifest() -> Dict[str, Dict[str, str]]:
    return _BUILTIN_FALLBACK_SCHEMA


def _get_legacy_all_columns() -> FrozenSet[str]:
    return frozenset(col for tbl in _BUILTIN_FALLBACK_SCHEMA.values() for col in tbl)


def _get_legacy_concept_map() -> Dict[str, List[str]]:
    # Minimal fallback — same structure as registry.build_concept_map()
    concept_map: Dict[str, List[str]] = {}
    for tname, cols in _BUILTIN_FALLBACK_SCHEMA.items():
        tl = tname.lower()
        pks = [c for c in cols if "ID" in c and not c.endswith("_ID") or c == list(cols.keys())[0]]
        concept_map.setdefault(tl, pks or [list(cols.keys())[0]])
        concept_map.setdefault(tl + "s", concept_map[tl])
        for cname in cols:
            readable = cname.lower().replace("_", " ").replace(" cd", "").replace(" amt", " amount")
            concept_map.setdefault(readable, [cname])
    return concept_map


def _get_legacy_schema_block() -> str:
    lines = ["=== GROUND-TRUTH SCHEMA (4 tables) ===", ""]
    for table, cols in _BUILTIN_FALLBACK_SCHEMA.items():
        lines.append(f"Table: {table}")
        for col, desc in cols.items():
            lines.append(f"  {col:<24} — {desc}")
        lines.append("")
    return "\n".join(lines)


# ── Module-level exports for backward-compat imports ─────────────────────────
# Downstream code that does:
#   from pipeline.hyde_expander import SCHEMA_MANIFEST, _ALL_COLUMNS
# will still work.  When you have fully migrated to the registry, these can
# be deleted.

SCHEMA_MANIFEST: Dict[str, Dict[str, str]] = _get_legacy_manifest()
_ALL_COLUMNS: FrozenSet[str] = _get_legacy_all_columns()
_CONCEPT_TO_COLUMNS: Dict[str, List[str]] = _get_legacy_concept_map()


def inject_registry(registry) -> None:
    """
    Update the module-level legacy vars with data from a live registry.
    Call this once at app startup after loading the registry:

        from pipeline.hyde_expander import inject_registry
        inject_registry(SchemaRegistry.load(cfg))

    After this call, any code doing ``from pipeline.hyde_expander import
    SCHEMA_MANIFEST`` will get the registry-derived version.
    """
    global SCHEMA_MANIFEST, _ALL_COLUMNS, _CONCEPT_TO_COLUMNS
    SCHEMA_MANIFEST = registry.manifest
    _ALL_COLUMNS = registry.all_columns
    _CONCEPT_TO_COLUMNS = registry.build_concept_map()
    logger.info(
        "hyde_expander module vars refreshed from registry: %d tables, %d columns.",
        len(SCHEMA_MANIFEST), len(_ALL_COLUMNS),
    )
    print(f"[inject_registry] Module-level vars updated from registry: {len(SCHEMA_MANIFEST)} tables, {len(_ALL_COLUMNS)} columns")


# ── Introspection helpers (unchanged API) ─────────────────────────────────────

def get_table_columns(table: str) -> Dict[str, str]:
    return SCHEMA_MANIFEST[table.upper()]


def column_exists(column: str, table: Optional[str] = None) -> bool:
    if table:
        return column.upper() in SCHEMA_MANIFEST.get(table.upper(), {})
    return column.upper() in _ALL_COLUMNS


def tables_for_column(column: str) -> List[str]:
    idx: Dict[str, List[str]] = {}
    for tname, cols in SCHEMA_MANIFEST.items():
        for cname in cols:
            idx.setdefault(cname, []).append(tname)
    return idx.get(column.upper(), [])


def validate_column_list(columns: List[str]) -> ValidationReport:
    synthetic = " ".join(columns)
    validator = SchemaValidator(_ALL_COLUMNS)
    return validator.validate(synthetic)