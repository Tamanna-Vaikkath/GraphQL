"""
hyde_expander.py — HyDE (Hypothetical Document Embedding) query expansion.

Produces a short, intent-focused hypothetical passage from the user's natural-
language question. The passage is embedded and used for semantic retrieval
against the schema knowledge graph.

Key design principles
---------------------
1.  SCHEMA-GROUNDED — The expander is initialised with the complete ground-truth
    column manifest for all four tables (CLAIMS, POLICY, PAYMENT, CLAIMANT).
    The LLM prompt includes this manifest verbatim so it can ONLY reference
    columns that actually exist.

2.  HALLUCINATION GUARD — After the LLM returns a hypothetical passage,
    `SchemaValidator` scans it for column-like tokens and flags any that are
    NOT present in the ground-truth manifest.  Flagged tokens are removed from
    the passage before it reaches the embedding step, and a structured
    `ValidationReport` is returned alongside the passage so callers (app.py,
    the trace panel) can surface warnings.

3.  TRIM UPSTREAM — Trimming to ≤2 sentences happens here, before embedding,
    so the KG retrieval call always sees a sharp, focused passage and the trace
    panel shows the actual text that drove retrieval.

4.  MULTIPLE PROMPT VARIANTS — Four `PromptStrategy` values let callers choose
    between CONCISE (default), VERBOSE, DOMAIN_CODES, and ANTI_HALLUCINATION
    prompts.  Each variant is schema-aware.

5.  ZERO EXTERNAL DEPS beyond the standard library and the existing
    `utils/llm_client` contract (any object with a `.complete(prompt) -> str`
    method).
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
    cannot be grounded to *any* column in the 4-table schema.

    This happens when:
    1. The LLM's hypothetical passage contains zero known schema column tokens
       after the hallucination guard strips all unrecognised tokens, AND
    2. The original question does not reference any known column or table name
       directly.

    The pipeline should catch this, stop immediately, and surface a
    user-friendly "not found in schema" message rather than generating SQL
    that would hallucinate columns or return misleading results.

    Attributes
    ----------
    question:
        The original user question that triggered the error.
    unknown_tokens:
        Column-like tokens the LLM invented that don't exist in the schema.
    suggestion:
        Optional hint pointing to the closest real column(s), if any.
    """

    def __init__(
        self,
        question: str,
        unknown_tokens: List[str],
        suggestion: Optional[str] = None,
    ) -> None:
        self.question = question
        self.unknown_tokens = unknown_tokens
        self.suggestion = suggestion
        if unknown_tokens:
            tokens_str = ", ".join(f"'{t}'" for t in unknown_tokens)
            detail = (
                f"The following concept(s) in your question have no corresponding column "
                f"in the database schema (CLAIMS, POLICY, PAYMENT, CLAIMANT): {tokens_str}. "
                f"The database cannot answer this question because the required data is not stored."
            )
        else:
            detail = (
                "Your question does not correspond to any column in the database schema "
                "(CLAIMS, POLICY, PAYMENT, CLAIMANT). "
                "The database cannot answer this question because the required data is not stored."
            )
        super().__init__(detail)


# ── Ground-truth schema manifest ─────────────────────────────────────────────
# Single source of truth.  Keep this in sync with seed_db.py / build_graph.py.
# Structure:  { TABLE_NAME: { COL_NAME: human_readable_description } }

SCHEMA_MANIFEST: Dict[str, Dict[str, str]] = {
    "CLAIMS": {
        "CLAIM_ID":       "Primary key — unique claim identifier.",
        "POLICY_ID":      "Foreign key → POLICY.POLICY_ID — links claim to its policy.",
        "CLAIMANT_ID":    "Foreign key → CLAIMANT.CLAIMANT_ID — links claim to the person who filed it.",
        "CLM_STAT_CD":    "Claim status code: O=Open, C=Closed, P=Pending, D=Denied.",
        "LOSS_DT":        "Date the loss/incident occurred.",
        "REPORT_DT":      "Date the claim was reported to the insurer.",
        "LOSS_TYPE_CD":   "Type of loss: AUTO, PROP, LIAB, WC, MARINE.",
        "INCURRED_AMT":   "Total incurred loss amount (paid + reserve).",
        "RESERVE_AMT":    "Outstanding reserve amount; 0 for closed/denied claims.",
        "ADJUSTER_ID":    "Internal ID of the adjuster assigned to the claim.",
        "CLOSE_DT":       "Date the claim was closed; NULL for open/pending claims.",
        "LITIGATION_FLG": "Y/N flag indicating whether the claim is in litigation.",
    },
    "POLICY": {
        "POLICY_ID":      "Primary key — unique policy identifier.",
        "POLICY_NBR":     "Human-readable policy number string (e.g. PL-2021-00042).",
        "INSURED_NM":     "Name of the insured person or entity on the policy.",
        "POL_EFF_DT":     "Policy effective (start) date.",
        "POL_EXP_DT":     "Policy expiration date.",
        "LINE_OF_BUSNSS": "Line of business: PERSONAL_AUTO, HOMEOWNERS, COMMERCIAL, WC.",
        "STATE_CD":       "Two-letter US state code where the policy is written.",
        "PREMIUM_AMT":    "Annual premium amount in dollars.",
        "DEDUCTIBLE_AMT": "Policy deductible amount; common values 500, 1000, 2500, 5000.",
        "AGENT_ID":       "Internal ID of the agent who wrote the policy.",
        "POL_STAT_CD":    "Policy status: AC=Active, CN=Cancelled, EX=Expired.",
    },
    "PAYMENT": {
        "PAYMENT_ID":     "Primary key — unique payment transaction identifier.",
        "CLAIM_ID":       "Foreign key → CLAIMS.CLAIM_ID — links payment to its claim.",
        "PMT_DT":         "Date the payment was issued.",
        "PMT_AMT_GROSS":  "Gross payment amount before any deductions.",
        "PMT_AMT_NET":    "Net payment amount after deductions (≈ gross × 0.9).",
        "PMT_STAT_CD":    "Payment status: IS=Issued, CL=Cleared, VD=Voided, PD=Paid.",
        "PMT_TYPE_CD":    "Payment type: INDEM=Indemnity, MED=Medical, EXP=Expense.",
        "PAYEE_NM":       "Name of the payment recipient.",
        "CHK_NBR":        "Check number string (e.g. CHK-482910).",
        "VOID_RSN_CD":    "Void reason code when PMT_STAT_CD='VD': DUPE or ERROR.",
    },
    "CLAIMANT": {
        "CLAIMANT_ID":    "Primary key — unique claimant identifier.",
        "CLAIMANT_NM":    "Full name of the claimant.",
        "DOB":            "Date of birth of the claimant.",
        "GENDER_CD":      "Gender code: M=Male, F=Female, U=Unknown.",
        "ADDRESS_LINE1":  "Street address of the claimant.",
        "STATE_CD":       "Two-letter US state code of the claimant's address.",
        "CONTACT_PHONE":  "Claimant phone number string.",
        "ATTY_REP_FLG":   "Y/N flag indicating attorney representation.",
        "CLAIM_COUNT":    "Denormalised count of claims filed by this claimant.",
        "FRAUD_RISK_SCRE":"Fraud risk score 0–100; higher = greater risk.",
    },
}

# Build a flat set of ALL valid column names for O(1) lookup
_ALL_COLUMNS: FrozenSet[str] = frozenset(
    col
    for table_cols in SCHEMA_MANIFEST.values()
    for col in table_cols
)

# Build a reverse index:  COL_NAME → list of tables that contain it
_COL_TO_TABLES: Dict[str, List[str]] = {}
for _table, _cols in SCHEMA_MANIFEST.items():
    for _col in _cols:
        _COL_TO_TABLES.setdefault(_col, []).append(_table)


# ── Schema reference block (injected into every LLM prompt) ──────────────────

def _build_schema_block() -> str:
    """Render the full schema manifest as a compact text block for the prompt."""
    lines: List[str] = [
        "=== GROUND-TRUTH SCHEMA (4 tables — reference ONLY these columns) ===",
        "",
    ]
    for table, cols in SCHEMA_MANIFEST.items():
        lines.append(f"Table: {table}")
        for col, desc in cols.items():
            lines.append(f"  {col:<20} — {desc}")
        lines.append("")
    lines.append(
        "IMPORTANT: You MUST NOT invent, guess, or use any column name that is "
        "not listed above. If the question implies a concept that has no matching "
        "column, do not attempt to represent it."
    )
    return "\n".join(lines)


_SCHEMA_BLOCK: str = _build_schema_block()   # computed once at import time


# ── Prompt strategies ─────────────────────────────────────────────────────────

class PromptStrategy(str, Enum):
    """
    Selects which system prompt variant the LLM receives.

    CONCISE          — 1–2 tight sentences, domain codes, schema-grounded.
                       Best default for embedding quality.
    VERBOSE          — 3–4 sentences with more business context.
                       Use when retrieval recall is low on complex questions.
    DOMAIN_CODES     — Explicitly enumerates status codes and FK paths.
                       Best for questions involving filters (e.g. "open claims").
    ANTI_HALLUCINATION — Maximally restrictive; forbids any token not in the
                       schema block. Use when hallucination rate is high.
    """
    CONCISE           = "concise"
    VERBOSE           = "verbose"
    DOMAIN_CODES      = "domain_codes"
    ANTI_HALLUCINATION = "anti_hallucination"


_SYSTEM_PROMPTS: Dict[PromptStrategy, str] = {

    PromptStrategy.CONCISE: (
        "You are a P&C insurance data analyst. Given a natural-language question, "
        "write exactly 1–2 sentences that directly mirror the question's intent "
        "using precise insurance domain terminology and the EXACT column names, "
        "table names, and status codes from the schema below.\n\n"
        "{schema_block}\n\n"
        "Rules:\n"
        "- Use ONLY column names listed above — no others.\n"
        "- Name the exact tables and columns the query would touch.\n"
        "- ONLY include a status code filter (e.g. CLM_STAT_CD='O') if the user's "
        "question explicitly asks for that status (e.g. 'open claims', 'voided payments'). "
        "If the user does NOT mention a specific status, do NOT add any status filter.\n"
        "- Do NOT generate SQL. Do NOT describe generic database structure.\n"
        "- Do NOT use any column name not in the schema above.\n"
        "- Do NOT add any filter, constraint, or condition the user did not ask for.\n"
        "- Stay tightly focused on ONLY what the question is actually asking — nothing more."
    ),

    PromptStrategy.VERBOSE: (
        "You are a P&C insurance data analyst with deep knowledge of claims "
        "operations. Given a natural-language question, write 3–4 sentences that "
        "describe the data required to answer it, referencing ONLY the tables and "
        "columns from the schema below.\n\n"
        "{schema_block}\n\n"
        "Rules:\n"
        "- Use ONLY column names listed above — no others.\n"
        "- Mention every table that would be needed (joins included).\n"
        "- ONLY specify a status code filter if the user explicitly asked for that "
        "status. Do NOT add filters the user did not request.\n"
        "- Describe what the result set would look like (which columns, any "
        "aggregations, ordering).\n"
        "- Do NOT generate SQL.\n"
        "- Do NOT invent column names.\n"
        "- Do NOT add any assumption, condition, or constraint beyond what the user asked."
    ),

    PromptStrategy.DOMAIN_CODES: (
        "You are a P&C insurance data dictionary. Given a natural-language question, "
        "produce 1–2 sentences that enumerate EXACTLY which columns and status code "
        "values satisfy the question, using ONLY columns from the schema below.\n\n"
        "{schema_block}\n\n"
        "Rules:\n"
        "- Lead with the primary table(s) and the filter column + value "
        "(e.g. 'CLAIMS.CLM_STAT_CD = ''O'' for open claims') ONLY if the user asked for "
        "that specific status. Do NOT add a status filter the user did not request.\n"
        "- List any FK joins needed (e.g. 'joined to PAYMENT on CLAIMS.CLAIM_ID = "
        "PAYMENT.CLAIM_ID').\n"
        "- Available status codes for reference (use ONLY those the user's question calls for): "
        "CLM_STAT_CD (O/C/P/D), PMT_STAT_CD (IS/CL/VD/PD), "
        "POL_STAT_CD (AC/CN/EX), PMT_TYPE_CD (INDEM/MED/EXP), "
        "LOSS_TYPE_CD (AUTO/PROP/LIAB/WC/MARINE), "
        "LINE_OF_BUSNSS (PERSONAL_AUTO/HOMEOWNERS/COMMERCIAL/WC).\n"
        "- Use ONLY column names listed above — no others.\n"
        "- Do NOT generate SQL.\n"
        "- Do NOT add any filter or condition the user did not explicitly ask for."
    ),

    PromptStrategy.ANTI_HALLUCINATION: (
        "You are a strict P&C insurance schema validator. Given a question, produce "
        "1–2 sentences describing the data needed, using ONLY the column names "
        "listed in the schema below. Any word that looks like a column name but is "
        "NOT in the schema below must not appear in your response.\n\n"
        "{schema_block}\n\n"
        "Absolute rules:\n"
        "1. Every column token you write MUST appear verbatim in the schema above.\n"
        "2. If the question mentions a concept with no matching column, say "
        "'this concept has no corresponding column in the schema'.\n"
        "3. Do NOT generate SQL.\n"
        "4. Do NOT invent, abbreviate, or shorten column names.\n"
        "5. Treat this as a closed-book test — the schema block is your only "
        "allowed reference."
    ),
}


# ── Validation report ─────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """
    Result of running the hallucination guard on a HyDE passage.

    Attributes
    ----------
    is_clean:
        True if no unknown column tokens were found.
    unknown_tokens:
        Column-like tokens found in the passage that do NOT exist in the schema.
    known_tokens:
        Column-like tokens found that DO exist in the schema.
    cleaned_passage:
        The passage with unknown tokens stripped (used for embedding).
    warnings:
        Human-readable warning strings suitable for the trace panel.
    """
    is_clean: bool
    unknown_tokens: List[str] = field(default_factory=list)
    known_tokens: List[str] = field(default_factory=list)
    cleaned_passage: str = ""
    warnings: List[str] = field(default_factory=list)
    has_schema_coverage: bool = True  # False → question maps to nothing in DB


# ── Schema validator ──────────────────────────────────────────────────────────

class SchemaValidator:
    """
    Scans a HyDE passage for column-like tokens and validates them against
    the ground-truth schema manifest.

    A "column-like token" is any ALL_CAPS word (≥3 chars) optionally followed
    by _SUFFIX, OR any token that appears verbatim in ``_ALL_COLUMNS``.

    Design note
    -----------
    We deliberately use a conservative detector (ALLCAPS + underscore pattern)
    to avoid false positives on ordinary words.  The detector will catch tokens
    like ``CLM_STAT_CD``, ``INCURRED_AMT``, ``FRAUD_RISK_SCRE``, and similar
    typical DB column names while ignoring lowercase prose.
    """

    # Regex: ALL_CAPS words with at least one underscore OR pure ALLCAPS ≥4 chars
    _COL_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[A-Z]{4,})\b')

    def validate(self, passage: str) -> ValidationReport:
        """
        Validate *passage* and return a ``ValidationReport``.

        Parameters
        ----------
        passage:
            The raw HyDE passage from the LLM.

        Returns
        -------
        ValidationReport with ``cleaned_passage`` ready for embedding.
        """
        if not passage:
            return ValidationReport(
                is_clean=True,
                cleaned_passage=passage,
            )

        candidates: Set[str] = set(self._COL_PATTERN.findall(passage))

        unknown: List[str] = []
        known: List[str] = []

        for token in sorted(candidates):
            if token in _ALL_COLUMNS:
                known.append(token)
            else:
                # Only flag it if it LOOKS like an intended column reference
                # (contains underscore → almost certainly meant as a column name)
                if "_" in token:
                    unknown.append(token)
                # Pure ALLCAPS without underscore (e.g. "SQL", "KG", "P&C") — skip

        warnings: List[str] = []
        cleaned = passage

        for token in unknown:
            # Suggest closest real column by prefix match
            suggestion = _suggest_closest(token)
            msg = (
                f"Column '{token}' does NOT exist in any of the 4 tables "
                f"(CLAIMS, POLICY, PAYMENT, CLAIMANT)."
            )
            if suggestion:
                msg += f" Closest known column: '{suggestion}'."
            warnings.append(msg)
            logger.warning("HyDE hallucination guard: %s", msg)
            # Remove the unknown token from the cleaned passage to prevent
            # it from polluting the embedding
            cleaned = re.sub(rf'\b{re.escape(token)}\b', '', cleaned)

        # Collapse multiple spaces left by removals
        cleaned = re.sub(r'  +', ' ', cleaned).strip()

        return ValidationReport(
            is_clean=len(unknown) == 0,
            unknown_tokens=unknown,
            known_tokens=known,
            cleaned_passage=cleaned,
            warnings=warnings,
            has_schema_coverage=len(known) > 0,
        )



# ── Ontology pre-check (deterministic, no LLM) ───────────────────────────────
# Maps insurance business concepts → the schema column(s) that represent them.
# If a concept appears in the question but has NO entry here, it has no DB column.
# This is the authoritative ground-truth mapping; keep in sync with SCHEMA_MANIFEST.
_CONCEPT_TO_COLUMNS: Dict[str, List[str]] = {
    # CLAIMS table
    "claim id":          ["CLAIM_ID"],
    "claim status":      ["CLM_STAT_CD"],
    "open claim":        ["CLM_STAT_CD"],
    "closed claim":      ["CLM_STAT_CD"],
    "pending claim":     ["CLM_STAT_CD"],
    "denied claim":      ["CLM_STAT_CD"],
    "loss date":         ["LOSS_DT"],
    "loss type":         ["LOSS_TYPE_CD"],
    "report date":       ["REPORT_DT"],
    "incurred":          ["INCURRED_AMT"],
    "incurred amount":   ["INCURRED_AMT"],
    "reserve":           ["RESERVE_AMT"],
    "reserve amount":    ["RESERVE_AMT"],
    "adjuster":          ["ADJUSTER_ID"],
    "close date":        ["CLOSE_DT"],
    "litigation":        ["LITIGATION_FLG"],
    # POLICY table
    "policy":            ["POLICY_ID", "POLICY_NBR"],
    "policy number":     ["POLICY_NBR"],
    "insured name":      ["INSURED_NM"],
    "policy effective":  ["POL_EFF_DT"],
    "policy expiration": ["POL_EXP_DT"],
    "policy expiry":     ["POL_EXP_DT"],
    "line of business":  ["LINE_OF_BUSNSS"],
    "lob":               ["LINE_OF_BUSNSS"],
    "state":             ["STATE_CD"],
    "premium":           ["PREMIUM_AMT"],
    "deductible":        ["DEDUCTIBLE_AMT"],
    "agent":             ["AGENT_ID"],
    "policy status":     ["POL_STAT_CD"],
    # PAYMENT table
    "payment":           ["PAYMENT_ID", "PMT_AMT_GROSS"],
    "payment date":      ["PMT_DT"],
    "payment amount":    ["PMT_AMT_GROSS", "PMT_AMT_NET"],
    "gross payment":     ["PMT_AMT_GROSS"],
    "net payment":       ["PMT_AMT_NET"],
    "payment status":    ["PMT_STAT_CD"],
    "payment type":      ["PMT_TYPE_CD"],
    "payee":             ["PAYEE_NM"],
    "check number":      ["CHK_NBR"],
    "void":              ["VOID_RSN_CD", "PMT_STAT_CD"],
    "voided payment":    ["PMT_STAT_CD", "VOID_RSN_CD"],
    # CLAIMANT table
    "claimant":          ["CLAIMANT_ID", "CLAIMANT_NM"],
    "claimant name":     ["CLAIMANT_NM"],
    "date of birth":     ["DOB"],
    "dob":               ["DOB"],
    "gender":            ["GENDER_CD"],
    "address":           ["ADDRESS_LINE1"],
    "phone":             ["CONTACT_PHONE"],
    "attorney":          ["ATTY_REP_FLG"],
    "attorney representation": ["ATTY_REP_FLG"],
    "claim count":       ["CLAIM_COUNT"],
    "fraud":             ["FRAUD_RISK_SCRE"],
    "fraud risk":        ["FRAUD_RISK_SCRE"],
    "fraud score":       ["FRAUD_RISK_SCRE"],
}

# Concepts that are valid insurance domain terms but are NOT stored in the 4-table
# schema. Any question that references ONLY these (with no _CONCEPT_TO_COLUMNS
# match) should be flagged as "column not in database."
_UNMAPPABLE_INSURANCE_CONCEPTS: FrozenSet[str] = frozenset({
    "complaint", "complaints", "customer complaint",
    "nps", "net promoter score", "customer satisfaction",
    "renewal", "renewals", "renewal rate",
    "quote", "quotes", "rating",
    "underwriting score", "underwriting decision",
    "reinsurance", "reinsurer", "treaty",
    "actuarial", "actuarial reserve", "ibnr",
    "catastrophe", "cat bond",
    "broker commission", "commission",
    "medical record", "medical records",
    "incident report",
    "note", "notes", "claim notes", "adjuster notes",
    "document", "documents", "attachment",
    "email", "correspondence",
    "audit trail", "audit log",
    "sla", "turnaround time", "cycle time",
    "customer", "customer id", "customer number",
    "account", "account number",
    "naic code",
    "coverage type", "coverage limit",
    # ── Auto/liability sub-types that have NO dedicated column ────────────
    # The schema stores loss type as LOSS_TYPE_CD with values AUTO/PROP/LIAB/WC/MARINE.
    # Sub-categories like "bodily injury" or "property damage" are not stored as
    # separate columns — silently mapping them to LIAB produces hallucinated results.
    "bodily injury", "bodily injury claim", "bi claim",
    "property damage", "property damage claim", "pd claim",
    "collision", "collision claim",
    "comprehensive claim",
    "uninsured motorist", "underinsured motorist",
    "pip claim", "personal injury protection claim",
    "med pay", "medical payments",
    "umbrella claim",
    "professional liability claim", "errors and omissions claim", "e&o claim",
    "directors and officers claim", "d&o claim",
    "product liability claim",
    "general liability claim", "gl claim",
    "cyber claim", "cyber liability claim",
    "inland marine claim",
    "flood claim", "earthquake claim",
    "builder's risk claim",
    "surety claim", "fidelity claim",
    "crop claim",
    "coverage sub-type", "coverage category",
})


def _ontology_precheck(question: str) -> Optional[List[str]]:
    """
    Deterministic pre-LLM check: returns a list of unmappable concept strings if
    the question references things not in the 4-table schema, or None if the
    question looks answerable from the DB.

    Returns
    -------
    None
        → question appears answerable; proceed to LLM expansion.
    []
        → question has no insurance/DB intent at all (off-topic).
    [str, ...]
        → one or more concepts identified that have no DB column.

    Design
    ------
    1. If the question contains ANY _CONCEPT_TO_COLUMNS key → probably answerable
       (return None to let the LLM layers confirm).
    2. If the question contains any _UNMAPPABLE_INSURANCE_CONCEPTS key → flag it.
    3. If the question has no insurance signal at all (none of the above) → off-topic.

    This runs before any LLM call, so it never hallucinates and costs O(n).
    """
    q_lower = question.lower()

    # Check for concepts that DO map to DB columns
    has_db_concept = any(concept in q_lower for concept in _CONCEPT_TO_COLUMNS)

    # Check for concepts that are insurance-sounding but NOT in the DB
    found_unmappable = [
        concept for concept in _UNMAPPABLE_INSURANCE_CONCEPTS
        if concept in q_lower
    ]

    # Check for bare schema column/table names in the question (e.g. "CLM_STAT_CD")
    q_upper = question.upper()
    has_direct_column = any(col in q_upper for col in _ALL_COLUMNS)
    has_direct_table  = any(tbl in q_upper for tbl in SCHEMA_MANIFEST)

    if has_direct_column or has_direct_table:
        return None   # direct schema reference → let LLM layers handle it

    if found_unmappable and not has_db_concept:
        # Insurance-sounding but no matching column(s)
        return found_unmappable

    if has_db_concept:
        # Some concepts map to DB columns — let the LLM layers verify further
        # (they can catch sub-concept issues the simple string scan misses)
        if found_unmappable:
            # Mixed: some real columns + some unmappable — flag the unmappable ones
            return found_unmappable
        return None

    # No insurance signal at all
    # We return None here to let the existing Layer 1 + Layer 2 handle it,
    # because a question like "show me data from Texas" has no explicit concept
    # key but IS answerable (STATE_CD). 
    # The LLM layers are the right backstop for ambiguous
    # non-keyed questions.
    return None


def _suggest_closest(token: str) -> Optional[str]:
    """
    Return the schema column most similar to *token* by longest common prefix,
    or None if no reasonable match exists.
    """
    token_upper = token.upper()
    best: Optional[str] = None
    best_len = 0
    for col in _ALL_COLUMNS:
        common = 0
        for a, b in zip(token_upper, col):
            if a == b:
                common += 1
            else:
                break
        if common > best_len:
            best_len = common
            best = col
    # Only suggest if at least 4 characters match
    return best if best_len >= 4 else None


# ── Trim helper ───────────────────────────────────────────────────────────────

def trim_hyde(expanded: str, max_sentences: int = 2) -> str:
    """
    Reduce a verbose HyDE paragraph to at most *max_sentences* sentences.

    Rationale
    ---------
    Embedding models compress long texts by averaging token representations.
    A two-sentence passage that captures the query intent precisely produces a
    sharper embedding than a five-sentence paragraph with filler.  Empirically,
    the first 1–2 sentences of a hypothetical answer carry the most signal.

    Parameters
    ----------
    expanded:
        The raw hypothetical passage returned by the LLM.
    max_sentences:
        Maximum number of sentences to retain (default: 2).

    Returns
    -------
    The trimmed passage, or the original string if splitting fails.
    """
    if not expanded:
        return expanded
    sentences = re.split(r'(?<=[.!?])\s+', expanded.strip())
    trimmed = " ".join(sentences[:max_sentences])
    return trimmed or expanded


# ── Expander interface ────────────────────────────────────────────────────────

@runtime_checkable
class HyDEExpander(Protocol):
    """Minimal protocol for any HyDE expander implementation."""

    def expand(self, question: str) -> str:
        """Return the trimmed, validated hypothetical passage for *question*."""
        ...

    def expand_with_report(
        self, question: str
    ) -> Tuple[str, ValidationReport]:
        """
        Like ``expand`` but also returns the ``ValidationReport``.

        Callers that want to surface hallucination warnings in the trace panel
        should use this method instead of ``expand``.
        """
        ...


# ── Main expander implementation ──────────────────────────────────────────────

class LLMHyDEExpander:
    """
    Schema-grounded HyDE expander: calls an LLM to generate a hypothetical
    document, validates the output against the ground-truth schema manifest,
    then trims to ``max_sentences`` before returning.

    Parameters
    ----------
    llm_client:
        Any object with a ``complete(prompt: str) -> str`` method
        (e.g. an AzureOpenAI wrapper).
    max_sentences:
        Number of sentences to retain after trimming (default: 2).
    strategy:
        ``PromptStrategy`` controlling which system prompt variant to use.
        Defaults to ``PromptStrategy.CONCISE``.
    custom_system_prompt:
        Optional fully custom system prompt.  If provided, *strategy* is
        ignored, but ``{schema_block}`` will still be injected if the
        placeholder is present in the string.
    strict_validation:
        If True (default), unknown column tokens are stripped from the passage
        before embedding.  Set False to preserve the original passage while
        still logging warnings.
    """

    def __init__(
        self,
        llm_client,
        max_sentences: int = 2,
        strategy: PromptStrategy = PromptStrategy.CONCISE,
        custom_system_prompt: Optional[str] = None,
        strict_validation: bool = True,
    ) -> None:
        self._client = llm_client
        self._max_sentences = max_sentences
        self._strategy = strategy
        self._strict = strict_validation
        self._validator = SchemaValidator()

        if custom_system_prompt is not None:
            # Inject schema block if the caller left the placeholder
            self._system = custom_system_prompt.format(
                schema_block=_SCHEMA_BLOCK
            ) if "{schema_block}" in custom_system_prompt else custom_system_prompt
        else:
            template = _SYSTEM_PROMPTS[strategy]
            self._system = template.format(schema_block=_SCHEMA_BLOCK)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, question: str) -> str:
        """
        Generate, validate, and trim a hypothetical passage for *question*.

        Returns the cleaned+trimmed passage ready for embedding.
        Logs warnings for any hallucinated column names but does not raise.
        """
        passage, _ = self.expand_with_report(question)
        return passage

    def expand_with_report(
        self, question: str
    ) -> Tuple[str, ValidationReport]:
        """
        Like ``expand`` but also returns the full ``ValidationReport``.

        Use this from the pipeline/orchestrator when you want to surface
        hallucination warnings in the Streamlit trace panel.

        Returns
        -------
        (cleaned_trimmed_passage, ValidationReport)
        """
        prompt = f"Question: {question}"
        raw: str = self._client.complete(self._system + "\n\n" + prompt)

        # 1. Validate against schema manifest
        report = self._validator.validate(raw)

        # 2. Choose passage: cleaned (strict) or original (lenient)
        base_passage = report.cleaned_passage if self._strict else raw

        # 3. Trim to max_sentences
        trimmed = trim_hyde(base_passage, max_sentences=self._max_sentences)

        # Update the report's cleaned_passage to reflect trimming
        report.cleaned_passage = trimmed

        if not report.is_clean:
            logger.warning(
                "HyDE passage for question %r had %d unknown column token(s): %s",
                question, len(report.unknown_tokens), report.unknown_tokens,
            )

        return trimmed, report

    def expand_or_raise(self, question: str) -> str:
        """
        Like ``expand`` but raises ``SchemaGroundingError`` when the question
        cannot be mapped to any column in the schema.

        Use this in the pipeline instead of ``expand`` to short-circuit
        execution before KG retrieval, SQL generation, and DB execution —
        preventing hallucinated queries and misleading results.

        Two-layer check:

        Layer 1 — Token presence check (existing):
          * The cleaned passage contains **zero** known schema column tokens, AND
          * The original question itself also contains no known column or table name.

        Layer 2 — LLM concept coverage check (new):
          * Even if some schema tokens survive (e.g. ``CLM_STAT_CD`` from "denied
            claims"), the question may contain additional concepts (e.g. "customer
            complaints") that have NO column representation at all.  The LLM is
            asked to enumerate every distinct queryable concept in the question and
            confirm whether each one maps to a real column.  If any concept is
            unmappable, ``SchemaGroundingError`` is raised immediately — the pipeline
            never reaches SQL generation.

        Raises
        ------
        SchemaGroundingError
            When any key concept in the question has no database column mapping.

        Returns
        -------
        str
            The cleaned, trimmed hypothetical passage (same as ``expand``).
        """
        # ── Layer 0: deterministic ontology pre-check (no LLM call) ─────────
        # Scans the question for concepts that are explicitly non-insurance
        # OR are valid insurance terms but have NO column in any of the 4 tables.
        # This is O(len(question)) and scales to any schema size.
        non_db_concepts = _ontology_precheck(question)
        if non_db_concepts is not None:
            # non_db_concepts == [] means fully off-topic (no insurance terms at all)
            # non_db_concepts == [str, ...] means insurance concept(s) with no column
            raise SchemaGroundingError(
                question=question,
                unknown_tokens=non_db_concepts,
                suggestion=_suggest_closest(non_db_concepts[0]) if non_db_concepts else None,
            )

        passage, report = self.expand_with_report(question)

        # ── Layer 1: token presence check (unchanged) ─────────────────────────
        if not report.has_schema_coverage:
            question_upper = question.upper()
            direct_column_hit = any(col in question_upper for col in _ALL_COLUMNS)
            if not direct_column_hit:
                suggestion: Optional[str] = None
                if report.unknown_tokens:
                    suggestion = _suggest_closest(report.unknown_tokens[0])
                raise SchemaGroundingError(
                    question=question,
                    unknown_tokens=report.unknown_tokens,
                    suggestion=suggestion,
                )

        # ── Layer 2: LLM concept coverage check ───────────────────────────────
        # Even when some schema tokens survive, verify that EVERY core concept
        # in the question is representable by a real column.  This catches
        # cases like "show customer complaints related to denied claims" where
        # "denied claims" grounds to CLM_STAT_CD='D' but "customer complaints"
        # has no column at all.
        unmapped = self._check_concept_coverage(question)
        if unmapped:
            suggestion_str: Optional[str] = _suggest_closest(unmapped[0]) if unmapped else None
            raise SchemaGroundingError(
                question=question,
                unknown_tokens=unmapped,
                suggestion=suggestion_str,
            )

        return passage

    def _check_concept_coverage(self, question: str) -> List[str]:
        """
        Ask the LLM to identify every distinct queryable concept in *question*
        and verify whether each one maps to a real column in the schema.

        Returns a list of concept strings that have NO column mapping.
        Returns an empty list when every concept is covered.

        This is a targeted, low-latency call: it sends a compact prompt that
        asks only for a JSON response — no prose, no explanation.
        """
        _COVERAGE_SYSTEM = (
            "You are a strict P&C insurance database schema validator.\n\n"
            + _SCHEMA_BLOCK
            + "\n\n"
            "Your job: given a user question, identify every distinct *queryable concept* "
            "the question is asking about (e.g. 'customer complaints', 'denied claims', "
            "'fraud score', 'policy state'). Then, for each concept, determine whether "
            "there is at least one column in the schema above that can represent it "
            "WITH SUFFICIENT PRECISION.\n\n"
            "Rules:\n"
            "1. A concept is 'covered' ONLY if a real column stores or encodes that "
            "   information precisely enough to answer the question without guessing:\n"
            "   - 'denied claims' → CLM_STAT_CD='D' is covered (exact match).\n"
            "   - 'open claims' → CLM_STAT_CD='O' is covered (exact match).\n"
            "   - 'LIAB claims' → LOSS_TYPE_CD='LIAB' is covered (exact match).\n"
            "   - 'bodily injury claims' → NOT covered. The schema stores loss type as "
            "     LOSS_TYPE_CD with broad values (AUTO/PROP/LIAB/WC/MARINE). "
            "     'Bodily injury' is a sub-type of LIAB with no dedicated column — "
            "     silently mapping it to LIAB='LIAB' would return wrong results.\n"
            "   - 'property damage claims' → NOT covered for the same reason.\n"
            "   - 'customer complaints' → no column exists → not covered.\n"
            "2. Generic filter words like 'show', 'list', 'related to', 'with' are NOT concepts.\n"
            "3. Return ONLY a JSON object with two keys:\n"
            "   - \"covered\": list of concept strings that ARE precisely in the schema\n"
            "   - \"unmapped\": list of concept strings that have NO column representation "
            "     OR that the schema can only represent at a coarser granularity\n"
            "4. Output only the JSON object — no explanation, no markdown fences."
        )
        user_prompt = f"Question: {question}"
        try:
            raw = self._client.complete(_COVERAGE_SYSTEM + "\n\n" + user_prompt)
            # Strip markdown fences if the model adds them
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = __import__("json").loads(raw[start:end])
            unmapped: List[str] = parsed.get("unmapped", [])
            logger.info(
                "Concept coverage check for %r — covered=%s unmapped=%s",
                question, parsed.get("covered", []), unmapped,
            )
            return [str(u) for u in unmapped] if unmapped else []
        except Exception as e:
            # If the LLM call or parse fails, log and allow the pipeline to
            # continue rather than blocking on a safety check failure.
            logger.warning("Concept coverage check failed (%s) — allowing pipeline to proceed.", e)
            return []

    # ------------------------------------------------------------------
    # Convenience: switch strategy at runtime
    # ------------------------------------------------------------------

    def set_strategy(self, strategy: PromptStrategy) -> None:
        """Hot-swap the prompt strategy (rebuilds the system prompt)."""
        self._strategy = strategy
        template = _SYSTEM_PROMPTS[strategy]
        self._system = template.format(schema_block=_SCHEMA_BLOCK)
        logger.info("HyDE prompt strategy switched to: %s", strategy.value)


# ── Schema introspection helpers (useful for the trace panel / debugging) ─────

def get_table_columns(table: str) -> Dict[str, str]:
    """
    Return the column→description mapping for *table*.

    Raises ``KeyError`` if *table* is not in the schema manifest.
    """
    return SCHEMA_MANIFEST[table.upper()]


def column_exists(column: str, table: Optional[str] = None) -> bool:
    """
    Return True if *column* exists in the schema manifest.

    If *table* is given, checks only that specific table; otherwise checks all.
    """
    if table:
        return column.upper() in SCHEMA_MANIFEST.get(table.upper(), {})
    return column.upper() in _ALL_COLUMNS


def tables_for_column(column: str) -> List[str]:
    """Return the list of tables that contain *column* (may be more than one)."""
    return _COL_TO_TABLES.get(column.upper(), [])


def validate_column_list(columns: List[str]) -> ValidationReport:
    """
    Validate an arbitrary list of column name strings against the schema.

    Useful for the retrieval stage to double-check its output before SQL gen.
    Builds a synthetic passage from the column list and runs the validator.
    """
    synthetic = " ".join(columns)
    validator = SchemaValidator()
    return validator.validate(synthetic)