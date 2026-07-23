"""
app.py — Streamlit UI for NLQ Intelligence System.

Features:
  - Natural language input box
  - Interactive results dataframe (sortable, filterable via st.dataframe)
  - AI Summary toggle
  - Query Trace panel (expandable) — shows HyDE output, retrieved columns, SQL
  - Download as CSV button
  - Query history sidebar

Run:
    streamlit run app.py
"""
import html
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import tempfile, os, textwrap
from pyvis.network import Network

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from pipeline.orchestrator import NLQPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLQ Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a3a6b 0%, #2e6da4 100%);
        padding: 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .main-header img {
        height: 56px;
        width: auto;
        background: white;
        border-radius: 6px;
        padding: 4px 8px;
    }
    .main-header-text h2 {
        margin: 0;
        font-size: 1.45rem;
    }
    .main-header-text p {
        margin: 0;
        opacity: 0.85;
        font-size: 0.92rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-left: 4px solid #2e6da4;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    .sql-box {
        background: #1e1b2e;
        color: #e5d9ff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        border-left: 4px solid #8b5cf6;
    }
    .success-badge { color: #28a745; font-weight: bold; }
    .error-badge   { color: #dc3545; font-weight: bold; }

    /* ── Repair-agent before/after trace ───────────────────────────────── */
    .repair-sql-box {
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        margin-bottom: 0.4rem;
    }
    .repair-sql-before {
        background: #2a1414;
        color: #f5c6c6;
        border-left: 4px solid #dc3545;
    }
    .repair-sql-after {
        background: #14261a;
        color: #c6f5d4;
        border-left: 4px solid #28a745;
    }
    .repair-diff-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        white-space: pre-wrap;
        border-left: 4px solid #f0ad4e;
        margin-bottom: 0.4rem;
    }
    .repair-error-box {
        background: #2a1414;
        color: #ffb3b3;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        border-left: 4px solid #dc3545;
        margin-bottom: 0.4rem;
    }
    .repair-note-pass { color: #28a745; }
    .repair-note-fail { color: #dc3545; }
    /* Hide the native "Press Enter to apply" tooltip on text inputs */
    [data-testid="InputInstructions"] { display: none !important; }

    /* ── Cypher / Knowledge Graph lane ─────────────────────────────────── */
    .cypher-box {
        background: #1e1b2e;
        color: #e5d9ff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        border-left: 4px solid #8b5cf6;
    }
    .lane-pill {
        display: inline-block; border-radius: 999px; padding: 3px 14px;
        font-size: 0.82rem; font-weight: 700; letter-spacing: 0.02em;
    }
    .lane-pill-sql    { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd; }
    .lane-pill-cypher { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd; }

    /* ── Swim-lane diagram ──────────────────────────────────────────────── */
    .swimlane-wrap {
        border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.9rem 1rem 1rem;
        margin: 0.6rem 0 1rem; background: #fbfdff;
    }
    .swimlane-title {
        font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; color: #64748b; margin-bottom: 6px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .lane-row {
        display: flex; align-items: center; gap: 0; margin: 6px 0;
        padding: 8px 10px; border-radius: 8px; transition: all 0.2s ease;
    }
    .lane-row-active   { background: rgba(139,92,246,0.07); }
    .lane-row-active.cypher-active { background: rgba(139,92,246,0.07); }
    .lane-row-inactive { opacity: 0.42; filter: grayscale(35%); }
    .lane-name {
        min-width: 150px; font-size: 0.82rem; font-weight: 700; color: #334155;
        display:flex; align-items:center; gap:6px;
    }
    .lane-steps { display: flex; align-items: center; flex-wrap: wrap; gap: 0; flex: 1; }
    .lane-step {
        font-size: 0.74rem; font-weight: 600; padding: 5px 11px; border-radius: 999px;
        white-space: nowrap; color: #1e293b; background: #eef2f7; border: 1px solid #dbe3ec;
    }
    .lane-row-active .lane-step { background:#8b5cf6; color:#fff; border-color:#7c3aed; }
    .lane-row-active.cypher-active .lane-step { background:#8b5cf6; color:#fff; border-color:#7c3aed; }
    .lane-arrow { color: #94a3b8; margin: 0 6px; font-size: 0.8rem; }
    .lane-row-active .lane-arrow { color: #7c3aed; }
    .lane-row-active.cypher-active .lane-arrow { color: #7c3aed; }
    .lane-reason {
        margin-top: 8px; font-size: 0.82rem; color: #475569;
        background: #f8fafc; border-left: 3px solid #cbd5e1; padding: 6px 10px; border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    try:
        cfg = load_config()
        st.session_state.pipeline = NLQPipeline(cfg)
        st.session_state.config_ok = True
    except Exception as e:
        st.session_state.config_ok = False
        st.session_state.config_error = str(e)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, QueryResult)

if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}  # question -> QueryResult

if "irrelevant_query" not in st.session_state:
    st.session_state.irrelevant_query = False

if "definitional_query" not in st.session_state:
    st.session_state.definitional_query = None

if "definitional_answer" not in st.session_state:
    st.session_state.definitional_answer = None


# ── Logo path ───────────────────────────────────────────
LOGO_PATH = Path(__file__).parent / "assets" / "vm_logo.png"

def _logo_b64() -> str:
    """Return base64-encoded logo for inline HTML embedding."""
    import base64
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""

_logo = _logo_b64()
_logo_tag = (
    f'<img src="data:image/png;base64,{_logo}" alt="Value Momentum">'
    if _logo else ""
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    ai_summary = st.toggle("AI Summary", value=True,
                           help="Generate a plain-English narrative of the result set.")
    show_trace = st.toggle("Show Query Trace", value=False,
                           help="Reveal HyDE expansion, retrieved columns, and generated SQL.")

    st.divider()
    st.markdown("#### Sample Queries — SQL lane")
    st.caption("Filters, aggregations, fixed-depth joins")
    _SAMPLE_QUERIES = [
        "show open claims",
        "show denied claims",
        "show payments for open claims",
        "show claims from Texas",
        "show pending claims with payments",
    ]
    for _sq in _SAMPLE_QUERIES:
        if st.button(_sq, key=f"sq_{_sq}", use_container_width=True):
            st.session_state["pending_query"] = _sq
            st.session_state["run_sample"] = True
            st.rerun()

    st.markdown("#### Sample Queries — Cypher / KG lane")
    st.caption("Relationships, multi-hop, unknown-depth traversal")
    _SAMPLE_CYPHER_QUERIES = [
        "which claimants share the same policy as another claimant",
        "find claims connected through the same adjuster",
        "show the chain of payments across a claimant's claims",
        "which policies are linked through the same agent",
    ]
    for _cq in _SAMPLE_CYPHER_QUERIES:
        if st.button(_cq, key=f"cq_{_cq}", use_container_width=True):
            st.session_state["pending_query"] = _cq
            st.session_state["run_sample"] = True
            st.rerun()

    st.divider()
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []

    if st.button("Clear Cache", use_container_width=True):
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_question", None)
        st.session_state.pop("schema_grounding_error", None)
        st.session_state.pop("schema_grounding_question", None)
        st.session_state.query_cache = {}
        st.session_state.irrelevant_query = False
        st.session_state.definitional_query = None
        st.session_state.definitional_answer = None
        st.rerun()

    st.markdown("---")
    st.caption("NLQ Intelligence  ·  Neo4j + Azure OpenAI")
    if st.session_state.get("config_ok"):
        st.caption("Connected")
    else:
        st.caption("Config error")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    {_logo_tag}
    <div class="main-header-text">
        <h2>Graph-Driven NLQ Intelligence</h2>
        <p>Ask questions about Claims, Policies, Payments, and Claimants...</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Config error gate ─────────────────────────────────────────────────────────
if not st.session_state.get("config_ok", False):
    st.error(f"Configuration error: {st.session_state.get('config_error', 'Unknown')}")
    st.info("Check your `.env` file and ensure all required variables are set.")
    st.stop()

# ── Query input ───────────────────────────────────────────────────────────────
query_val = st.session_state.pop("pending_query", "")
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Ask a question about your P&C data:",
        value=query_val,
        placeholder="e.g. Show me all open claims in Texas with reserve over $100,000",
        key="question_input",
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("Run", type="primary", use_container_width=True)

# ── Query classification ──────────────────────────────────────────────────────
# Keywords that signal the question is about P&C insurance data.
_PC_KEYWORDS = {
    "claim", "claims", "claimant", "claimants", "policy", "policies",
    "payment", "payments", "adjuster", "adjusters", "reserve", "incurred",
    "premium", "deductible", "coverage", "loss", "fraud", "liability",
    "underwriter", "underwriting", "insured", "insurer", "endorsement",
    "subrogation", "indemnity", "peril", "exposure", "lob", "line of business",
    "open", "closed", "denied", "pending", "approved", "status", "expir",
    "insurance", "p&c", "property and casualty", "reinsurance",
}

# Phrase patterns that signal a conceptual/definitional question rather than
# a data query.  Checked *before* the keyword scan so "what is a policy?"
# never reaches the pipeline.
_DEFINITIONAL_PATTERNS = re.compile(
    r"""
    ^\s*
    (?:
        what\s+(?:is|are|does|do)\b              # "what is a policy", "what are claims"
      | how\s+(?:does|do|is|are|would|should)\b  # "how does subrogation work"
      | (?:can\s+you\s+)?(?:explain|define|describe|tell\s+me\s+about|elaborate\s+on)\b
      | what\s+does\s+.+\s+mean\b                # "what does premium mean"
      | (?:give\s+me\s+a?\s+)?definition\s+of\b  # "give me a definition of"
      | (?:i\s+(?:want|need|would\s+like)\s+to\s+(?:know|understand|learn)\s+(?:about|what))\b
      | (?:help\s+me\s+understand)\b
      | (?:what\s+(?:do\s+you\s+mean\s+by|does\s+.+\s+stand\s+for))\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Comprehensive set of insurance concepts that trigger a definitional answer.
# Covers broad topics ("insurance", "p&c") down to granular P&C terms.
_INSURANCE_CONCEPTS = {
    # Top-level / broad
    "insurance", "p&c", "p & c", "property and casualty", "property & casualty",
    "property casualty", "general insurance", "non-life insurance",
    # Policy & contract
    "policy", "policies", "policyholder", "policy holder", "policy term",
    "policy period", "policy limit", "policy number", "named insured",
    "additional insured", "declarations page", "dec page", "schedule of insurance",
    "binder", "certificate of insurance", "coi",
    # Claim & claimant
    "claim", "claims", "claimant", "claimants", "first party", "third party",
    "first notice of loss", "fnol", "proof of loss", "claim settlement",
    "claim status", "open claim", "closed claim", "claim denial",
    # Financial / premium
    "premium", "premiums", "net premium", "gross premium", "earned premium",
    "written premium", "unearned premium", "flat premium",
    "deductible", "deductibles", "self-insured retention", "sir",
    "copay", "co-pay", "coinsurance", "co-insurance", "out-of-pocket",
    # Coverage & limits
    "coverage", "coverages", "limit", "limits", "coverage limit",
    "aggregate limit", "per occurrence limit", "sublimit",
    "occurrence", "occurrence basis", "claims-made", "claims made basis",
    "blanket coverage", "scheduled coverage",
    # Loss & liability
    "loss", "losses", "loss ratio", "combined ratio", "expense ratio",
    "liability", "liabilities", "general liability", "gl",
    "professional liability", "errors and omissions", "e&o",
    "directors and officers", "d&o", "umbrella", "excess liability",
    "product liability", "completed operations",
    # Property
    "property insurance", "commercial property", "homeowners", "dwelling",
    "inland marine", "builder's risk", "equipment breakdown",
    "business interruption", "business income", "extra expense",
    "replacement cost", "actual cash value", "acv", "depreciation",
    # Auto
    "auto insurance", "commercial auto", "personal auto",
    "collision coverage", "comprehensive coverage", "uninsured motorist",
    "underinsured motorist", "pip", "personal injury protection",
    "bodily injury", "property damage",
    # Workers comp & specialty
    "workers compensation", "workers' compensation", "workers comp",
    "employer's liability", "employers liability",
    "marine insurance", "ocean marine", "aviation insurance",
    "cyber insurance", "cyber liability", "management liability",
    # People & roles
    "insured", "insurer", "underwriter", "underwriting", "adjuster",
    "adjudicator", "actuary", "actuarial", "broker", "agent",
    "independent agent", "captive agent", "surplus lines",
    "managing general agent", "mga",
    # Risk & exposure
    "risk", "risk management", "exposure", "hazard", "moral hazard",
    "morale hazard", "adverse selection", "risk transfer",
    "risk retention", "self-insurance", "captive insurance", "captive",
    "risk pooling", "pooling",
    # Reinsurance
    "reinsurance", "reinsurer", "treaty reinsurance", "facultative reinsurance",
    "quota share", "excess of loss", "stop loss", "catastrophe reinsurance",
    "cat bond", "catastrophe bond",
    # Reserves & financials
    "reserve", "reserves", "loss reserve", "ibnr",
    "incurred but not reported", "case reserve", "bulk reserve",
    "loss development", "tail factor", "actuarial reserve",
    # Legal / settlement
    "subrogation", "indemnity", "indemnification", "indemnify",
    "contribution", "salvage", "abandonment", "waiver of subrogation",
    "release", "settlement", "mediation", "arbitration", "appraisal",
    "litigation", "bad faith", "good faith",
    # Policy features
    "endorsement", "rider", "exclusion", "exclusions", "condition",
    "conditions", "warranty", "representation", "concealment",
    "material fact", "utmost good faith", "uberrimae fidei",
    "peril", "named peril", "open peril", "all risk",
    "occurrence form", "claims made form",
    # Fraud & SIU
    "fraud", "insurance fraud", "siu", "special investigations unit",
    "red flag", "fraud indicator", "staged accident",
    # Lines of business / product types
    "line of business", "lob", "commercial lines", "personal lines",
    "specialty lines", "surplus lines",
    "homeowners insurance", "renters insurance", "condo insurance",
    "flood insurance", "earthquake insurance",
    "crop insurance", "title insurance", "surety", "surety bond",
    "fidelity bond", "crime insurance",
    # Misc / industry
    "insolvency", "guaranty fund", "admitted carrier", "non-admitted carrier",
    "a.m. best", "am best", "financial strength rating",
    "naic", "national association of insurance commissioners",
    "state insurance department", "insurance commissioner",
    "rate filing", "rate regulation", "actuarial table", "mortality table",
    "loss development factor", "ldf", "incurred losses",
    "paid losses", "outstanding losses",
}

# ── Comprehensive P&C Insurance Glossary ─────────────────────────────────────
_INSURANCE_GLOSSARY: dict[str, str] = {

    # ── Broad / top-level ────────────────────────────────────────────────────
    "insurance": (
        "**Insurance** is a financial arrangement in which an individual or organisation pays a "
        "premium to an insurer in exchange for protection against specified financial losses. "
        "The insurer pools premiums from many policyholders and pays out claims when covered "
        "events occur. Insurance is built on the principle of risk transfer — shifting the "
        "financial impact of uncertain events from the individual to the insurer."
    ),
    "p&c": (
        "**Property & Casualty (P&C) insurance** is a broad category of insurance that covers "
        "damage to or loss of property, and legal liability for injuries or damages caused to "
        "others. It includes lines such as Homeowners, Commercial Property, Auto, General "
        "Liability, Workers' Compensation, and more. P&C is distinct from Life and Health "
        "insurance and is regulated separately in most jurisdictions."
    ),
    "property and casualty": (
        "**Property & Casualty (P&C) insurance** is a broad category of insurance that covers "
        "damage to or loss of property, and legal liability for injuries or damages caused to "
        "others. It includes lines such as Homeowners, Commercial Property, Auto, General "
        "Liability, Workers' Compensation, and more. P&C is distinct from Life and Health "
        "insurance and is regulated separately in most jurisdictions."
    ),
    "property casualty": (
        "**Property & Casualty (P&C) insurance** covers damage to property and liability for "
        "injuries or damages to others. Major lines include Auto, Homeowners, Commercial "
        "Property, General Liability, and Workers' Compensation."
    ),
    "general insurance": (
        "**General insurance** (the term used outside North America for P&C insurance) covers "
        "non-life risks such as property damage, auto accidents, liability, travel, and marine. "
        "Policies are typically short-term (annual) and renewable."
    ),

    # ── Policy & contract ───────────────────────────────────────────────────
    "policy": (
        "An insurance **policy** is a legally binding contract between an insurer and a "
        "policyholder that defines the terms, conditions, coverage limits, exclusions, and "
        "premium obligations. It specifies which losses are covered and the insurer's "
        "obligations upon a covered event."
    ),
    "policies": (
        "An insurance **policy** is a legally binding contract between an insurer and a "
        "policyholder defining coverage terms, limits, exclusions, and premium obligations."
    ),
    "policyholder": (
        "A **policyholder** is the person or organisation that owns an insurance policy and is "
        "responsible for paying the premiums. They may also be the insured party, though the "
        "two roles can be separate (e.g. a parent taking out a policy on a child)."
    ),
    "declarations page": (
        "The **declarations page** (or 'dec page') is the summary page of an insurance policy "
        "that lists key details: policyholder name, insured property or entity, policy number, "
        "effective and expiration dates, coverage types, limits, deductibles, and premium. "
        "It is typically the first page of the policy document."
    ),
    "binder": (
        "A **binder** is a temporary insurance agreement that provides immediate coverage while "
        "the formal policy is being issued. It confirms that coverage is in effect and outlines "
        "the basic terms until the full policy document is delivered."
    ),
    "certificate of insurance": (
        "A **Certificate of Insurance (COI)** is a document issued by an insurer or broker that "
        "summarises the key coverage details of a policy. It serves as proof of insurance for "
        "third parties (e.g., landlords, clients) without disclosing the full policy terms."
    ),
    "named insured": (
        "The **named insured** is the person or entity specifically identified in the policy "
        "declarations as the primary covered party. They have the broadest rights under the "
        "policy, including the ability to make changes or cancel it."
    ),
    "additional insured": (
        "An **additional insured** is a person or organisation added to a policy — beyond the "
        "named insured — who receives coverage under that policy, typically for liability "
        "purposes. Common in contracts where one party requires proof of coverage."
    ),

    # ── Claim & process ─────────────────────────────────────────────────────
    "claim": (
        "A **claim** is a formal request made by a policyholder or claimant to their insurance "
        "company seeking compensation for a covered loss or event. Once filed, it is reviewed "
        "by an adjuster who investigates the loss, determines coverage, and authorises payment."
    ),
    "claims": (
        "A **claim** is a formal request to an insurer for compensation following a covered "
        "loss. The claims process involves filing, investigation, coverage determination, "
        "and settlement or denial."
    ),
    "claimant": (
        "A **claimant** is the individual or entity that files a claim against an insurance "
        "policy, seeking financial compensation for a loss or damage. The claimant may be the "
        "policyholder (first-party claim) or an injured third party (third-party claim)."
    ),
    "first notice of loss": (
        "**First Notice of Loss (FNOL)** is the initial report made to an insurer after an "
        "insured event occurs. It triggers the claims process and typically includes the date, "
        "location, and nature of the loss, the parties involved, and any known damages."
    ),
    "fnol": (
        "**FNOL (First Notice of Loss)** is the initial report filed with an insurer when an "
        "insured event occurs, triggering the claims handling process."
    ),
    "claim status": (
        "**Claim status** indicates the current stage of a claim in the handling process. "
        "Common statuses include: Open (active, under investigation), Closed (resolved), "
        "Denied (coverage not applicable), Pending (awaiting information), and Approved "
        "(payment authorised)."
    ),

    # ── Financial / premium ─────────────────────────────────────────────────
    "premium": (
        "A **premium** is the amount paid by a policyholder — typically monthly or annually — "
        "to keep an insurance policy active. It is calculated based on risk factors such as "
        "coverage type, claims history, location, asset value, and industry. Earned premium "
        "is the portion 'used up' during the coverage period; unearned premium is the "
        "remaining portion if the policy is cancelled early."
    ),
    "earned premium": (
        "**Earned premium** is the portion of a written premium that corresponds to the "
        "coverage period already elapsed. For a 12-month policy halfway through its term, "
        "50% of the premium is earned."
    ),
    "unearned premium": (
        "**Unearned premium** is the portion of a written premium that applies to the "
        "remaining unexpired coverage period. It represents the insurer's liability if the "
        "policy were cancelled today."
    ),
    "deductible": (
        "A **deductible** is the out-of-pocket amount the policyholder must pay on a covered "
        "claim before the insurer pays the remainder. A higher deductible typically lowers "
        "the premium. For example, with a $2,000 deductible on a $10,000 loss, the insurer "
        "pays $8,000."
    ),
    "self-insured retention": (
        "A **Self-Insured Retention (SIR)** is similar to a deductible — the insured pays "
        "losses up to the SIR amount before the insurer's coverage kicks in. Unlike a "
        "deductible, the insured typically handles defence costs within the SIR as well."
    ),
    "coinsurance": (
        "**Coinsurance** has two meanings: (1) In property insurance, a clause requiring the "
        "insured to carry coverage equal to a specified percentage (e.g., 80%) of the "
        "property's value — failure to do so results in a penalty at claim time. "
        "(2) In health insurance, the percentage of costs the insured shares with the "
        "insurer after meeting the deductible."
    ),

    # ── Coverage & limits ───────────────────────────────────────────────────
    "coverage": (
        "**Coverage** refers to the scope of protection an insurance policy provides — the "
        "specific risks, losses, or events for which the insurer agrees to pay. Coverage "
        "limits define the maximum the insurer will pay per occurrence or in aggregate."
    ),
    "aggregate limit": (
        "An **aggregate limit** is the maximum total amount an insurer will pay for all "
        "covered losses during a policy period, regardless of how many individual claims "
        "are made. Once reached, no further claims are covered until renewal."
    ),
    "occurrence": (
        "An **occurrence** in insurance is a single event or continuous exposure to conditions "
        "that results in bodily injury or property damage. An occurrence-based policy covers "
        "losses from events that happen during the policy period, regardless of when the "
        "claim is filed."
    ),
    "claims-made": (
        "A **claims-made policy** provides coverage for claims that are both made and reported "
        "during the policy period, regardless of when the underlying event occurred "
        "(subject to a retroactive date). Common for professional liability and D&O policies."
    ),
    "replacement cost": (
        "**Replacement cost** is the amount needed to replace damaged property with a new "
        "item of like kind and quality at current prices, without any deduction for "
        "depreciation. It is generally higher than Actual Cash Value."
    ),
    "actual cash value": (
        "**Actual Cash Value (ACV)** is the replacement cost of property minus depreciation "
        "at the time of loss. It reflects the item's fair market value at the time of the "
        "claim, not what it would cost to replace it new."
    ),
    "acv": (
        "**ACV (Actual Cash Value)** is replacement cost minus depreciation — the fair market "
        "value of property at the time of loss."
    ),

    # ── Loss & liability ────────────────────────────────────────────────────
    "loss": (
        "In insurance, a **loss** refers to the financial harm suffered by the insured as a "
        "result of a covered peril or event. It can refer to the damage itself or the claim "
        "payment made by the insurer."
    ),
    "loss ratio": (
        "The **loss ratio** is a key insurer profitability metric: incurred losses divided by "
        "earned premiums. A loss ratio below 100% means the insurer collected more in premiums "
        "than it paid in losses. Combined with the expense ratio, it forms the combined ratio."
    ),
    "combined ratio": (
        "The **combined ratio** is the sum of the loss ratio and expense ratio. A combined "
        "ratio below 100% indicates underwriting profitability; above 100% means the insurer "
        "is paying out more in losses and expenses than it earns in premiums."
    ),
    "liability": (
        "**Liability** in insurance is a party's legal responsibility for damages or injuries "
        "caused to others. Liability insurance covers the insured against claims from third "
        "parties who suffer bodily injury or property damage due to the insured's actions "
        "or negligence."
    ),
    "general liability": (
        "**General Liability (GL)** insurance protects businesses against claims of bodily "
        "injury, property damage, and personal/advertising injury arising from their "
        "operations, products, or premises. It is one of the most common commercial P&C lines."
    ),
    "umbrella": (
        "An **umbrella policy** provides additional liability coverage above the limits of "
        "underlying policies (e.g., auto, general liability). It kicks in when the underlying "
        "limit is exhausted and may also cover some gaps not in the underlying policies."
    ),
    "professional liability": (
        "**Professional Liability insurance** (also called E&O — Errors & Omissions) protects "
        "professionals against claims of negligence, errors, or inadequate work. It is common "
        "for lawyers, doctors, architects, consultants, and technology firms."
    ),

    # ── Property lines ──────────────────────────────────────────────────────
    "homeowners": (
        "**Homeowners insurance** is a package policy for owner-occupied residences covering "
        "the dwelling structure, personal property, liability, and additional living expenses "
        "if the home becomes uninhabitable due to a covered loss."
    ),
    "business interruption": (
        "**Business Interruption (BI)** insurance covers lost income and operating expenses "
        "when a business is forced to close or reduce operations due to a covered property "
        "loss (e.g., fire or flood). It helps businesses survive financially during restoration."
    ),

    # ── Auto lines ──────────────────────────────────────────────────────────
    "auto insurance": (
        "**Auto insurance** provides financial protection against physical damage and bodily "
        "injury arising from traffic collisions, as well as liability from accidents. "
        "Personal auto covers private passenger vehicles; commercial auto covers business "
        "fleets and vehicles used for business purposes."
    ),
    "collision coverage": (
        "**Collision coverage** pays for damage to the insured's vehicle resulting from a "
        "collision with another vehicle or object, regardless of fault. It is subject to a "
        "deductible."
    ),
    "comprehensive coverage": (
        "**Comprehensive coverage** pays for vehicle damage from non-collision causes such as "
        "theft, vandalism, fire, hail, flood, or hitting an animal. It is subject to a "
        "deductible and is separate from collision coverage."
    ),
    "uninsured motorist": (
        "**Uninsured Motorist (UM) coverage** protects the insured if they are in an accident "
        "caused by a driver who has no auto insurance. Underinsured Motorist (UIM) coverage "
        "applies when the at-fault driver's limits are insufficient to cover the losses."
    ),

    # ── Workers comp ────────────────────────────────────────────────────────
    "workers compensation": (
        "**Workers' Compensation insurance** provides wage replacement and medical benefits "
        "to employees injured during the course of employment. In exchange, employees "
        "generally waive the right to sue the employer for negligence. It is mandatory in "
        "most US states."
    ),
    "workers comp": (
        "**Workers' Comp** (Workers' Compensation) covers medical expenses and lost wages for "
        "employees injured on the job, and protects employers from related lawsuits."
    ),

    # ── People & roles ──────────────────────────────────────────────────────
    "insured": (
        "The **insured** is the person, entity, or property covered under an insurance policy. "
        "They are entitled to receive claim payments when a covered loss occurs. The insured "
        "and policyholder may be the same person or different parties."
    ),
    "insurer": (
        "The **insurer** (also called the insurance company or carrier) is the organisation "
        "that underwrites the policy, collects premiums, and pays covered claims. Insurers "
        "are regulated by state insurance departments in the US."
    ),
    "underwriting": (
        "**Underwriting** is the process of evaluating risk, deciding whether to offer "
        "coverage, and determining the appropriate premium. Underwriters analyse factors like "
        "loss history, property characteristics, industry, and location to price risk "
        "accurately and protect the insurer's profitability."
    ),
    "adjuster": (
        "A **claims adjuster** investigates insurance claims to determine the extent of the "
        "insurer's liability and negotiates settlements. Staff adjusters are employed by the "
        "insurer; independent adjusters are contracted; public adjusters are hired by the "
        "policyholder to advocate on their behalf."
    ),
    "actuary": (
        "An **actuary** is a professional who uses mathematics, statistics, and financial "
        "theory to assess and price insurance risk. Actuaries calculate premiums, set "
        "reserves, analyse loss trends, and help ensure the insurer remains financially sound."
    ),
    "broker": (
        "An insurance **broker** acts as an intermediary between the client (insured) and "
        "insurance companies. Unlike agents who represent insurers, brokers represent the "
        "client and can shop multiple carriers to find suitable coverage."
    ),
    "managing general agent": (
        "A **Managing General Agent (MGA)** is an insurance intermediary granted underwriting "
        "authority by an insurer. MGAs can bind coverage, issue policies, and sometimes "
        "handle claims on behalf of the carrier — acting almost as an outsourced underwriting "
        "department."
    ),
    "mga": (
        "An **MGA (Managing General Agent)** is an intermediary with delegated underwriting "
        "authority from an insurer, able to bind policies and manage a book of business."
    ),

    # ── Risk concepts ───────────────────────────────────────────────────────
    "risk": (
        "In insurance, **risk** is the possibility of a loss or adverse event occurring. "
        "Insurers assess the likelihood and severity of potential losses to price policies "
        "appropriately. Risk can be pure (loss only, no gain possible) or speculative "
        "(chance of gain or loss)."
    ),
    "hazard": (
        "A **hazard** is a condition that increases the likelihood or severity of a loss. "
        "Physical hazards are tangible conditions (e.g., faulty wiring). Moral hazard arises "
        "when having insurance changes behavior (e.g., being less careful). Morale hazard "
        "is carelessness due to having coverage."
    ),
    "adverse selection": (
        "**Adverse selection** occurs when higher-risk individuals are more likely to seek "
        "insurance than lower-risk individuals, causing the insured pool to be riskier than "
        "average. Insurers counter this through underwriting, medical exams, and risk-based "
        "pricing."
    ),
    "captive insurance": (
        "A **captive** is an insurance company created and owned by a business (or group of "
        "businesses) to insure the risks of its parent. It provides more control over "
        "coverage, cost, and risk management than the commercial market."
    ),

    # ── Reinsurance ─────────────────────────────────────────────────────────
    "reinsurance": (
        "**Reinsurance** is insurance purchased by an insurance company (the ceding company) "
        "from another insurer (the reinsurer) to manage risk exposure and protect against "
        "large or catastrophic losses. Treaty reinsurance covers a portfolio of policies; "
        "facultative reinsurance covers individual risks."
    ),
    "quota share": (
        "A **quota share** is a type of proportional reinsurance where the ceding insurer and "
        "reinsurer share premiums and losses in a fixed percentage. For example, a 40% quota "
        "share means the reinsurer takes 40% of premiums and pays 40% of all losses."
    ),
    "excess of loss": (
        "**Excess of Loss (XOL)** reinsurance is a non-proportional arrangement where the "
        "reinsurer pays losses above a specified retention (attachment point) up to a stated "
        "limit. It protects the insurer from large individual losses or catastrophes."
    ),

    # ── Reserves & financials ────────────────────────────────────────────────
    "reserve": (
        "A **reserve** (or loss reserve) is the amount an insurer sets aside to pay future "
        "claim obligations — losses that have been reported but not yet settled (case "
        "reserves), or incurred but not yet reported (IBNR reserves). Accurate reserving is "
        "critical to financial solvency and regulatory compliance."
    ),
    "ibnr": (
        "**IBNR (Incurred But Not Reported)** is a reserve for losses that have already "
        "occurred but have not yet been reported to the insurer. Actuaries estimate IBNR "
        "using historical loss development patterns."
    ),
    "incurred losses": (
        "**Incurred losses** are the total losses an insurer has sustained during a period, "
        "including paid losses plus the change in outstanding loss reserves. It is used in "
        "the loss ratio calculation: Incurred Losses ÷ Earned Premium."
    ),

    # ── Legal / settlement ───────────────────────────────────────────────────
    "subrogation": (
        "**Subrogation** is the legal right of an insurer — after paying a claim — to step "
        "into the insured's shoes and pursue recovery from a liable third party. It prevents "
        "the insured from collecting twice for the same loss and helps insurers recoup "
        "paid claim amounts."
    ),
    "indemnity": (
        "**Indemnity** is the core insurance principle of restoring the insured to the same "
        "financial position they were in before the loss — no better, no worse. It prevents "
        "insurance from becoming a source of profit for the insured."
    ),
    "salvage": (
        "**Salvage** refers to the residual value of insured property after a loss. When an "
        "insurer pays a total-loss claim, it may take ownership of the damaged property and "
        "recover some cost through salvage sale."
    ),
    "bad faith": (
        "**Bad faith** in insurance refers to an insurer's unreasonable refusal to pay a "
        "valid claim, unjustified delays, or failure to properly investigate. Policyholders "
        "can sue insurers for bad faith, potentially recovering damages beyond the policy "
        "limits."
    ),

    # ── Policy features ──────────────────────────────────────────────────────
    "endorsement": (
        "An **endorsement** (or rider) is a written amendment to an insurance policy that "
        "modifies its original terms — adding, removing, or changing coverage. Endorsements "
        "customise standard policies for specific insured needs."
    ),
    "exclusion": (
        "An **exclusion** is a policy provision that eliminates coverage for specific risks, "
        "losses, or situations. Common exclusions include flood, earthquake, intentional acts, "
        "and wear-and-tear. Understanding exclusions is essential to knowing the real scope "
        "of coverage."
    ),
    "peril": (
        "A **peril** is a specific cause of loss — such as fire, theft, windstorm, or flood. "
        "Named-peril policies cover only the perils explicitly listed; open-peril (all-risk) "
        "policies cover all causes of loss except those explicitly excluded."
    ),

    # ── Fraud ────────────────────────────────────────────────────────────────
    "fraud": (
        "**Insurance fraud** is an intentional act of deception to obtain an illegitimate "
        "financial benefit from an insurer. It includes hard fraud (staging accidents, arson) "
        "and soft fraud (inflating legitimate claims). Insurers use Special Investigations "
        "Units (SIUs) to detect and combat fraud."
    ),
    "siu": (
        "**SIU (Special Investigations Unit)** is a dedicated team within an insurer that "
        "investigates suspected fraudulent claims, complex cases, and referred suspicious "
        "activity. SIUs work with law enforcement and may pursue civil or criminal action."
    ),

    # ── Lines of business ────────────────────────────────────────────────────
    "line of business": (
        "**Line of Business (LOB)** refers to a category of insurance products, such as "
        "Commercial Auto, Homeowners, Workers' Compensation, General Liability, or "
        "Professional Liability. Each LOB has distinct underwriting rules, risk profiles, "
        "pricing factors, and regulatory requirements."
    ),
    "lob": (
        "**LOB (Line of Business)** is a product category within insurance — e.g., Commercial "
        "Auto, Homeowners, Workers' Comp, or General Liability."
    ),
    "commercial lines": (
        "**Commercial lines** refers to insurance products designed for businesses and "
        "organisations, as opposed to personal lines which cover individuals. Examples "
        "include Commercial Property, General Liability, Commercial Auto, and Workers' Comp."
    ),
    "personal lines": (
        "**Personal lines** refers to insurance products sold to individuals and families, "
        "such as Homeowners, Personal Auto, Renters, and Umbrella policies."
    ),
    "surplus lines": (
        "**Surplus lines insurance** covers risks that standard (admitted) insurers are "
        "unwilling or unable to write. Surplus lines carriers are non-admitted but licensed "
        "by state surplus lines offices, and offer more flexible policy terms."
    ),

    # ── Regulatory ───────────────────────────────────────────────────────────
    "naic": (
        "The **NAIC (National Association of Insurance Commissioners)** is a US regulatory "
        "support organisation comprising the chief insurance regulators from all 50 states. "
        "It develops model laws, data standards, and regulatory tools, though actual "
        "insurance regulation occurs at the state level."
    ),
    "admitted carrier": (
        "An **admitted carrier** is an insurance company that has received a state license "
        "(Certificate of Authority) to sell insurance in that state, and is subject to state "
        "rate and form regulation. Non-admitted (surplus lines) carriers are not licensed "
        "but are permitted to write certain specialty risks."
    ),
}

def _is_pc_relevant(text: str) -> bool:
    """Return True if *text* contains at least one P&C domain keyword."""
    lowered = text.lower()
    return any(kw in lowered for kw in _PC_KEYWORDS)


def _is_definitional(text: str) -> bool:
    """
    Return True when the query is a conceptual/definitional question about an
    insurance term rather than a request to query the database.

    A query is considered definitional when it:
      1. Matches a "what is / explain / define / how does" phrasing pattern, AND
      2. References a known insurance concept.
    """
    if not _DEFINITIONAL_PATTERNS.match(text):
        return False
    lowered = text.lower()
    return any(concept in lowered for concept in _INSURANCE_CONCEPTS)


# ── Concepts that exist in insurance domain but have NO column in the 4-table DB ─
# When a user asks for data on one of these concepts, the pipeline would hallucinate
# columns that don't exist.  We catch these early and return a definitional answer.
_DB_UNMAPPED_CONCEPTS: frozenset[str] = frozenset({
    # Legal / settlement concepts
    "subrogation", "waiver of subrogation", "salvage", "abandonment", "contribution",
    "indemnification", "indemnify", "release", "mediation", "arbitration", "appraisal",
    "bad faith", "good faith",
    # Reinsurance
    "reinsurance", "reinsurer", "treaty reinsurance", "facultative reinsurance",
    "quota share", "excess of loss", "stop loss", "catastrophe reinsurance",
    "cat bond", "catastrophe bond",
    # Policy features not stored in DB
    "endorsement", "rider", "exclusion", "exclusions", "condition", "conditions",
    "warranty", "representation", "concealment", "material fact", "utmost good faith",
    "uberrimae fidei", "named peril", "open peril", "all risk",
    "occurrence form", "claims made form",
    # Coverage concepts (no coverage-details table)
    "coverage limit", "aggregate limit", "per occurrence limit", "sublimit",
    "blanket coverage", "scheduled coverage",
    # Premium calculations not in DB
    "earned premium", "unearned premium", "written premium", "net premium",
    "gross premium", "flat premium",
    # Risk/actuarial concepts
    "loss ratio", "combined ratio", "expense ratio", "loss development",
    "tail factor", "actuarial reserve", "ibnr", "incurred but not reported",
    "loss development factor", "ldf", "bulk reserve",
    "moral hazard", "morale hazard", "adverse selection", "risk pooling",
    "risk transfer", "risk retention", "self-insurance", "captive insurance",
    # Coverage types not in DB
    "collision coverage", "comprehensive coverage", "uninsured motorist",
    "underinsured motorist", "pip", "personal injury protection",
    "business interruption", "business income", "extra expense",
    "replacement cost", "actual cash value", "acv",
    "builder's risk", "equipment breakdown", "inland marine",
    # Specialty / regulatory
    "surety", "surety bond", "fidelity bond", "crime insurance",
    "title insurance", "crop insurance", "flood insurance", "earthquake insurance",
    "aviation insurance", "ocean marine",
    "insolvency", "guaranty fund", "admitted carrier", "non-admitted carrier",
    "rate filing", "rate regulation",
    "naic", "national association of insurance commissioners",
    "a.m. best", "am best", "financial strength rating",
    # Binders / certificates
    "binder", "certificate of insurance", "coi", "dec page", "declarations page",
    "schedule of insurance",
    # Other financial
    "copay", "co-pay", "coinsurance", "co-insurance", "out-of-pocket",
    "self-insured retention", "sir",
})


def _is_concept_only_query(text: str) -> bool:
    """
    Return True when the question references an insurance concept that has NO
    corresponding column in the database schema.

    This catches imperative data queries like "show me subrogation claims" or
    "list all reinsurance records" where the concept sounds queryable but the
    DB has no column for it — the pipeline would hallucinate if we let it run.

    Only fires when the question contains NO known DB column names or
    schema-mapped terms (like 'open claims', 'fraud score', 'premium amount')
    that would anchor it to real data.
    """
    from pipeline.hyde_expander import _ALL_COLUMNS, SCHEMA_MANIFEST  

    lowered = text.lower()
    question_upper = text.upper()

    # If the question directly references a real column name, it's a DB query
    if any(col in question_upper for col in _ALL_COLUMNS):
        return False

    # Check whether the question mentions an unmapped concept
    hit_unmapped = any(concept in lowered for concept in _DB_UNMAPPED_CONCEPTS)
    if not hit_unmapped:
        return False

    # Final guard: also check that no DB-mapped keyword provides an anchor.
    # DB-mapped keywords are terms that map directly to real column values/names.
    _DB_MAPPED_KEYWORDS = {
        "open claim", "open claims", "closed claim", "closed claims",
        "denied claim", "denied claims", "pending claim", "pending claims",
        "fraud risk", "fraud score", "attorney", "litigation",
        "reserve", "incurred", "adjuster", "loss date", "report date",
        "close date", "loss type", "claim status",
        "premium", "deductible", "policy number", "policy status",
        "active policy", "cancelled policy", "expired policy",
        "line of business", "personal auto", "homeowners", "commercial", "workers comp",
        "payment status", "voided", "issued", "cleared",
        "indemnity payment", "medical payment", "expense payment",
        "payee", "check number",
        "claimant name", "date of birth", "gender", "phone", "address",
        "claim count", "fraud risk score",
    }
    if any(kw in lowered for kw in _DB_MAPPED_KEYWORDS):
        return False

    return True


def _get_definition(text: str) -> str | None:
    """
    Return the glossary definition for the primary insurance concept found in
    *text*, or None if no matching entry exists.
    Checks multi-word concepts first (e.g. 'line of business') before single words.
    """
    lowered = text.lower()
    # Multi-word concepts first
    for concept in sorted(_INSURANCE_GLOSSARY, key=lambda k: -len(k)):
        if concept in lowered:
            return _INSURANCE_GLOSSARY[concept]
    return None


# ── Pipeline execution ────────────────────────────────────────────────────────
# Trigger for Run button click OR a sidebar sample-query click.
_run_sample = st.session_state.pop("run_sample", False)
if (run_clicked or _run_sample) and question.strip():
    q = question.strip()
    if _is_definitional(q):
        # Conceptual/definitional question — show glossary answer, skip pipeline.
        st.session_state.irrelevant_query = False
        st.session_state.definitional_query = q
        st.session_state.definitional_answer = _get_definition(q)
        st.session_state.pop("schema_grounding_error", None)
        st.session_state.pop("schema_grounding_question", None)
    elif _is_concept_only_query(q):
        # Insurance concept that has no DB column mapping (e.g. "subrogation",
        # "reinsurance").  Return a glossary/definitional answer instead of
        # sending to the pipeline where it would hallucinate columns.
        st.session_state.irrelevant_query = False
        st.session_state.definitional_query = q
        definition = _get_definition(q)
        if definition:
            st.session_state.definitional_answer = definition
        else:
            # Concept recognised as unmapped but no glossary entry — give a
            # clear "not in database" message so the user understands why.
            st.session_state.definitional_answer = (
                f"The concept referenced in your question (**{q}**) is a valid "
                "P&C insurance term, but it does not correspond to any column in "
                "the database.\n\n"
                "The database contains these four tables:\n"
                "- **CLAIMS** — claim lifecycle, loss amounts, reserves, adjuster, litigation flag\n"
                "- **POLICY** — policy terms, premium, deductible, line of business, state\n"
                "- **PAYMENT** — payment transactions, amounts, status, type\n"
                "- **CLAIMANT** — claimant demographics, attorney flag, fraud risk score\n\n"
                "Try a question grounded in these tables, for example: "
                "*'Show open claims in Texas'*, *'List voided payments this year'*, "
                "or *'Find claimants with fraud risk above 80'*."
            )
        st.session_state.pop("schema_grounding_error", None)
        st.session_state.pop("schema_grounding_question", None)
    elif not _is_pc_relevant(q):
        st.session_state.irrelevant_query = True
        st.session_state.definitional_query = None
        st.session_state.pop("schema_grounding_error", None)
        st.session_state.pop("schema_grounding_question", None)
    else:
        st.session_state.irrelevant_query = False
        st.session_state.definitional_query = None
        with st.spinner("Running pipeline..."):
            pipeline: NLQPipeline = st.session_state.pipeline
            cache_key = q.lower().strip()
            if cache_key in st.session_state.query_cache:
                result = st.session_state.query_cache[cache_key]
            else:
                result = pipeline.query(q, generate_summary=ai_summary)
                if not result.schema_grounding_error:
                    st.session_state.query_cache[cache_key] = result
            if result.schema_grounding_error:
                # Store separately so the results area stays clean
                st.session_state["schema_grounding_error"] = result.error
                st.session_state["schema_grounding_question"] = q
            else:
                st.session_state.pop("schema_grounding_error", None)
                st.session_state.pop("schema_grounding_question", None)
                st.session_state.history.insert(0, (q, result))

# ── Definitional query answer ─────────────────────────────────────────────────
if st.session_state.get("definitional_query"):
    q_display = st.session_state.definitional_query
    answer = st.session_state.get("definitional_answer")
    st.markdown(f"### Definition: *{q_display}*")
    if answer:
        st.info(answer)
    else:
        # Fallback: concept recognised as definitional but no glossary entry yet.
        st.info(
            "This appears to be a conceptual insurance question. "
            "I have domain knowledge about P&C insurance terms, but no specific glossary entry "
            f"for your query: **{q_display}**\n\n"
            "To query your data, try something like: *'Show me all open claims'* or "
            "*'List policies expiring this month'*."
        )
    st.caption(
        "💡 To query your database, ask something specific like: "
        "*'Show open claims in Texas'*, *'List policies with premium over $5,000'*, "
        "or *'Find claims with fraud risk score above 80'*."
    )

# ── Schema grounding error — concept not in database ─────────────────────────
elif st.session_state.get("schema_grounding_error"):
    sge_question = st.session_state.get("schema_grounding_question", "your question")
    sge_detail   = st.session_state["schema_grounding_error"]

    # Extract the missing concept(s) from the error message.
    # Expected pattern: "... no corresponding column ... : 'concept one', 'concept two'"
    _missing_match = re.search(r":\s*(.+)$", sge_detail or "", re.IGNORECASE)
    _missing_raw   = _missing_match.group(1).strip() if _missing_match else ""
    # Split on commas and clean up surrounding quotes / whitespace
    _missing_concepts = [
        c.strip().strip("'\"").strip()
        for c in _missing_raw.split(",")
        if c.strip().strip("'\"").strip()
    ] if _missing_raw else []

    # Build pill-style HTML tags for each missing concept
    _pill_html = "".join(
        f"<span style='"
        f"display:inline-block; background:#fee2e2; color:#991b1b; "
        f"border:1px solid #fca5a5; border-radius:999px; "
        f"padding:2px 12px; margin:2px 4px; font-size:0.88rem; font-weight:600;'>"
        f"{c}</span>"
        for c in _missing_concepts
    ) if _missing_concepts else ""

    st.markdown(
        f"""
        <div style="background:#fff5f5; border:1px solid #fca5a5; border-radius:10px;
                    padding:1.1rem 1.4rem; margin-bottom:0.8rem;">
            <div style="font-size:1.05rem; font-weight:600; color:#b91c1c; margin-bottom:0.4rem;">
                ⚠️ We couldn't find data for: <em>{sge_question}</em>
            </div>
            <div style="color:#374151; font-size:0.95rem; margin-bottom:0.6rem;">
                The following {'concept' if len(_missing_concepts) == 1 else 'concepts'} you mentioned
                {'is' if len(_missing_concepts) == 1 else 'are'} not available in our database:
            </div>
            <div style="margin-bottom:0.5rem;">{_pill_html}</div>
            <div style="color:#6b7280; font-size:0.88rem;">
                Our database tracks <strong>Claims</strong>, <strong>Policies</strong>,
                <strong>Payments</strong>, and <strong>Claimants</strong>.
                Please rephrase your question. You can also refer to the sample queries provided.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "💡 Try asking about data that exists — for example: "
        "*'Show open claims with high fraud risk'*, "
        "or *'Find claimants in Texas with attorney representation'*."
    )

# ── Irrelevant query message ──────────────────────────────────────────────────
elif st.session_state.get("irrelevant_query"):
    st.warning(
"Hello! I am a specialized Insurance assistant designed to help you explore and analyze insurance-related data. "
"I can assist with information related to Claims, Policies, Payments, Claimants, and other connected insurance insights. "
"At the moment, I am not designed to answer general-purpose questions outside this domain. "
"For best results, please ask specific insurance-related questions so I can accurately query the database and retrieve meaningful insights. "
"Please try asking an insurance-related query, and I will be happy to help!"

    )

# ── Helper: extract joins actually present in the generated SQL ───────────────
def _extract_sql_joins(sql: str, all_join_conditions: list[str]) -> list[str]:
    """
    Return only those join conditions from *all_join_conditions* that are
    anchored to tables actually present in FROM / JOIN clauses of *sql*.

    Strategy
    --------
    1. Parse every table name that appears after FROM or JOIN keywords in the
       generated SQL (handles aliases, subqueries, multi-line formatting).
    2. For each candidate join condition, extract both table-name tokens
       (e.g. the "CLAIM" in "CLAIM.CLAIM_ID").
    3. Keep the condition only when *all* of its table tokens are in the
       FROM/JOIN table set — this eliminates irrelevant schema FK rows that
       just happen to share a table name present elsewhere in the query.
    """
    if not sql or not all_join_conditions:
        return []

    sql_upper = sql.upper()

    # Step 1: collect tables referenced in FROM / JOIN clauses only.
    # Pattern matches:  FROM  <NAME>  or  JOIN  <NAME>
    # and captures the base table name (before any alias or ON keyword).
    from_join_tables: set[str] = {
        m.group(1)
        for m in re.finditer(
            # Matches: FROM/JOIN, optional whitespace, optional '(', then table name.
            # Captures the base table name before any alias, ON, or WHERE keyword.
            r'\b(?:FROM|JOIN)\s+\(?\s*([A-Z_][A-Z0-9_]*)',
            sql_upper,
        )
    }

    kept = []
    for jc in all_join_conditions:
        # Step 2: extract table tokens from the join condition string.
        tables_in_cond = {
            m.group(1)
            for m in re.finditer(r'\b([A-Z_]+)\.[A-Z_]+\b', jc.upper())
        }
        # Step 3: require every table in the condition to be in FROM/JOIN set.
        if tables_in_cond and tables_in_cond.issubset(from_join_tables):
            kept.append(jc)
    return kept


# ── Helper: rerank retrieved columns with business-priority weighting ─────────
# Actual schema status / code columns that must surface above generic metadata.
# Sourced directly from the P&C DDL: CLM_STAT_CD, PMT_STAT_CD, POL_STAT_CD
# replace the fictional CLAIM_STATUS / PAYMENT_STATUS / POLICY_STATUS names.
_PRIORITY_COLS = {
    # Core identifiers
    "CLAIM_ID", "CLAIMANT_ID", "POLICY_ID", "ADJUSTER_ID", "PAYMENT_ID",
    # Actual schema status code columns
    "CLM_STAT_CD", "PMT_STAT_CD", "POL_STAT_CD",
    # Financial / loss columns
    "RESERVE_AMT", "PAYMENT_AMT", "INCURRED_AMT", "FRAUD_SCORE",
    # Categorical descriptors
    "LINE_OF_BUSINESS", "LOB_CD",
}

# ── US state names and abbreviations → trigger geographic column boost ────────
# Any query containing a state name or 2-letter abbreviation is treated as a
# geographic filter query and gets the same boost as "state" / "city" etc.
_US_STATES: frozenset[str] = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
    # 2-letter abbreviations (lower-cased for matching)
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id",
    "il","in","ia","ks","ky","la","me","md","ma","mi","mn","ms",
    "mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok",
    "or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv",
    "wi","wy",
})

_GEO_BOOST: dict[str, float] = {"STATE_CD": 0.25, "CITY_NM": 0.10, "ZIP_CD": 0.10, "COUNTY_CD": 0.10}


def _has_geo_intent(question: str) -> bool:
    """Return True if the question references a US state name or abbreviation."""
    q = question.lower()
    # Check multi-word state names first, then single tokens
    for state in _US_STATES:
        if " " in state:
            if state in q:
                return True
        else:
            # Word-boundary match for abbreviations (e.g. "TX" not inside "TEXT")
            if re.search(rf'\b{re.escape(state)}\b', q):
                return True
    return False


# Intent-specific boost maps: query-keyword → {col_name: extra_bonus}
# Applied on top of the base +0.15 priority bump so that columns that are
# *most* relevant to the detected query intent rank highest within the tier.
_INTENT_BOOST: list[tuple[str, dict[str, float]]] = [
    # Geographic / location queries (explicit keyword-based only).
    # NOTE: generic prepositions "in" / "from" are intentionally omitted —
    # they appear in almost every query ("claims in open status", "payments from
    # last month") and cause STATE_CD / geo columns to activate on unrelated
    # questions.  Geographic relevance is decided by _has_geo_intent() which
    # checks for actual US state names / abbreviations and explicit geo keywords.
    ("state",    {"STATE_CD": 0.20, "ZIP_CD": 0.15, "COUNTY_CD": 0.15}),
    ("zip",      {"ZIP_CD": 0.20, "STATE_CD": 0.15}),
    ("city",     {"CITY_NM": 0.20, "STATE_CD": 0.10}),
    ("county",   {"COUNTY_CD": 0.20, "STATE_CD": 0.10}),
    ("location", {"STATE_CD": 0.15, "CITY_NM": 0.15, "ZIP_CD": 0.10}),
    ("region",   {"STATE_CD": 0.15, "COUNTY_CD": 0.10}),
    # Claim status queries
    ("open",    {"CLM_STAT_CD": 0.20}),
    ("closed",  {"CLM_STAT_CD": 0.20}),
    ("denied",  {"CLM_STAT_CD": 0.20}),
    ("pending", {"CLM_STAT_CD": 0.20, "PMT_STAT_CD": 0.20}),
    ("status",  {"CLM_STAT_CD": 0.15, "PMT_STAT_CD": 0.15, "POL_STAT_CD": 0.15}),
    # Payment queries
    ("payment", {"PMT_STAT_CD": 0.20, "PAYMENT_AMT": 0.15, "PMT_AMT_GROSS": 0.10, "PMT_AMT_NET": 0.10}),
    ("paid",    {"PMT_STAT_CD": 0.20, "PAYMENT_AMT": 0.15}),
    # Policy queries
    ("policy",  {"POL_STAT_CD": 0.15, "POLICY_ID": 0.10}),
    ("expir",   {"POL_STAT_CD": 0.20, "EFF_DT": 0.10, "EXP_DT": 0.20}),
    # Financial threshold queries
    ("reserve", {"RESERVE_AMT": 0.20, "INCURRED_AMT": 0.10}),
    ("incurred",{"INCURRED_AMT": 0.20, "RESERVE_AMT": 0.10}),
    ("fraud",   {"FRAUD_RISK_SCRE": 0.25, "FRAUD_SCORE": 0.25}),
    ("attorney",{"ATTY_REP_FLG": 0.25}),
    ("litigat", {"LITIGATION_FLG": 0.25}),
    ("adjuster",{"ADJUSTER_ID": 0.20}),
]


def _intent_extra(question: str, col_name: str) -> float:
    """Return the largest applicable intent boost for *col_name* given *question*."""
    q = question.lower()
    col_upper = col_name.upper()
    best = 0.0

    # Keyword-driven boosts from _INTENT_BOOST table
    for keyword, boosts in _INTENT_BOOST:
        if keyword in q:
            best = max(best, boosts.get(col_upper, 0.0))

    # US state name/abbreviation → geographic column boost
    # e.g. "show claims from Texas" → STATE_CD gets +0.25 even without "state" keyword
    if _has_geo_intent(question):
        best = max(best, _GEO_BOOST.get(col_upper, 0.0))

    return best


def _rerank_columns(columns: list[dict], question: str = "") -> list[dict]:
    """
    Re-sort retrieved columns so business-critical identifiers / amounts rank
    above generic metadata (e.g. LOSS_TYPE_CD, REPORT_DT), with an additional
    query-intent boost for geo / status / payment queries.

    Scoring
    -------
    effective_score = base_score
                    + 0.15  (if col in _PRIORITY_COLS)
                    + intent_extra  (query-keyword × column match, 0–0.20)

    Geo columns (STATE_CD, CITY_NM, ZIP_CD, COUNTY_CD) are only boosted —
    and kept in the ranked output — when the question explicitly implies
    geography (US state names/abbreviations or explicit geo keywords).
    Without geo intent, those columns are preserved at their raw Neo4j score
    without any priority bump, so they naturally rank below query-relevant
    columns and are filtered out at the display-threshold stage.

    Duplicates (same column_name) are deduplicated — the highest-scoring
    occurrence is kept so the trace panel never shows redundant rows.
    """
    if not columns:
        return columns

    _GEO_ONLY_COLS = {"STATE_CD", "CITY_NM", "ZIP_CD", "COUNTY_CD"}
    query_has_geo  = _has_geo_intent(question)

    seen: set[str] = set()
    boosted: list[dict] = []
    for col in columns:
        # RetrievedColumn dataclass uses "name" and "table", not "column_name"
        col_name  = str(col.get("name",  col.get("column_name", ""))).upper()
        col_table = str(col.get("table", "")).upper()
        # Deduplicate on TABLE.COLUMN so CLAIMS.CLAIM_ID and PAYMENT.CLAIM_ID
        # are both kept — only exact duplicates are dropped.
        dedup_key = f"{col_table}.{col_name}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        priority_bonus = 0.15 if col_name in _PRIORITY_COLS else 0.0
        # Geo columns get zero bonus unless the query explicitly involves geography
        if col_name in _GEO_ONLY_COLS and not query_has_geo:
            intent_bonus = 0.0
        else:
            intent_bonus = _intent_extra(question, col_name)
        effective = col.get("score", 0) + priority_bonus + intent_bonus
        boosted.append({**col, "_effective_score": effective})

    boosted.sort(key=lambda c: c["_effective_score"], reverse=True)
    for c in boosted:
        c.pop("_effective_score", None)
    return boosted



# ── Neo4j KG Traversal Graph — vis.js full-canvas renderer ───────────────────
# Colour palette: one hue per domain, matched to the legend.
_TABLE_COLORS: dict[str, str] = {
    "CLAIMANT": "#9b59b6",   # Purple  
    "CLAIM":    "#e05252",   # Red
    "POLICY":   "#4a90d9",   # Blue
    "PAYMENT":  "#2ecc8e",   # Green
    "ADJUSTER": "#e67e22",   # Orange
}

def _table_color(table_name: str) -> str:
    t = table_name.upper()
    # Check exact match or table prefix first to avoid "CLAIM" matching "CLAIMANT".
    for prefix, color in _TABLE_COLORS.items():
        if t == prefix or t.startswith(prefix + "_"):
            return color
    for prefix, color in _TABLE_COLORS.items():
        if prefix in t:
            return color
    return "#78909c"


# ── SQL parsing helpers ────────────────────────────────────────────────────────
_SQL_KW = frozenset({
    "SELECT","WHERE","ON","SET","VALUES","INTO","WITH","LATERAL","UNNEST",
    "DUAL","TABLE","AS","JOIN","LEFT","RIGHT","INNER","OUTER","CROSS","FULL",
    "NATURAL","STRAIGHT_JOIN","AND","OR","NOT","IN","IS","NULL","BETWEEN",
    "LIKE","CASE","WHEN","THEN","ELSE","END","EXISTS","DISTINCT","ALL",
    "UNION","INTERSECT","EXCEPT","LIMIT","OFFSET","ORDER","GROUP","BY",
    "HAVING","ASC","DESC",
})

def _parse_sql_tables_and_aliases(sql: str) -> tuple[set[str], dict[str, str]]:
    """Return (real_tables, alias_map) from FROM/JOIN clauses."""
    sql_u = sql.upper()
    real_tables: set[str] = set()
    alias_map:   dict[str, str] = {}
    from_join_re = re.compile(
        r'\b(?:FROM|JOIN)\s+\(?\s*([A-Z_][A-Z0-9_$#]*)', re.IGNORECASE)
    alias_re = re.compile(
        r'[ \t]+(?:AS[ \t]+)?([A-Z_][A-Z0-9_$#]*)(?![ \t]*\.)', re.IGNORECASE)
    for m in from_join_re.finditer(sql_u):
        token = m.group(1).upper()
        if token in _SQL_KW:
            continue
        real_tables.add(token)
        am = alias_re.match(sql_u, m.end())
        if am:
            alias = am.group(1).upper()
            if alias not in _SQL_KW and alias != token:
                alias_map[alias] = token
    return real_tables, alias_map


def _resolve_table(token: str, alias_map: dict[str, str]) -> str:
    return alias_map.get(token.upper(), token.upper())


def _extract_sql_columns(sql: str, real_tables: set[str],
                         alias_map: dict[str, str]) -> set[tuple[str, str]]:
    """Return every (TABLE, COLUMN) pair qualified in the SQL body."""
    refs: set[tuple[str, str]] = set()
    for m in re.finditer(r'\b([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)\b',
                         sql.upper()):
        resolved = _resolve_table(m.group(1), alias_map)
        if resolved in real_tables:
            refs.add((resolved, m.group(2)))
    return refs


def _extract_fk_edges(sql: str, join_conditions: list[str],
                      real_tables: set[str],
                      alias_map: dict[str, str]) -> list[tuple[str, str, str]]:
    """
    Return deduplicated (src_table, dst_table, exact_predicate) triples.
    Priority: join_conditions supplied by Neo4j, then ON-clause fallback.
    The predicate string is the exact equality expression from the SQL/KG,
    e.g. 'CLAIMS.CLAIMANT_ID = CLAIMANT.CLAIMANT_ID'.
    """
    edges:      list[tuple[str, str, str]] = []
    seen_pairs: set[frozenset]             = set()
    jc_re = re.compile(
        r'\b([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)'
        r'\s*=\s*'
        r'([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)\b'
    )

    def _add(raw_src: str, raw_dst: str, predicate: str) -> None:
        s = _resolve_table(raw_src, alias_map)
        d = _resolve_table(raw_dst, alias_map)
        if s not in real_tables or d not in real_tables or s == d:
            return
        k = frozenset({s, d})
        if k in seen_pairs:
            return
        seen_pairs.add(k)
        edges.append((s, d, predicate))

    # Source 1 — Neo4j-supplied join_conditions (authoritative)
    for jc in (join_conditions or []):
        m = jc_re.search(jc.upper())
        if m:
            # Reconstruct predicate preserving original casing from jc
            pred = f"{m.group(1)}.{m.group(2)} = {m.group(3)}.{m.group(4)}"
            _add(m.group(1), m.group(3), pred)

    # Source 2 — ON clauses parsed directly from generated SQL
    for m in re.finditer(
        r'\bON\s+(.+?)(?=\s+(?:LEFT|RIGHT|INNER|CROSS|FULL|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT)|$)',
        sql.upper(), re.DOTALL
    ):
        jm = jc_re.search(m.group(1).strip())
        if jm:
            pred = f"{jm.group(1)}.{jm.group(2)} = {jm.group(3)}.{jm.group(4)}"
            _add(jm.group(1), jm.group(3), pred)

    return edges


# ── Main renderer — vis.js traversal graph (table + column nodes + FK path) ───
def _render_traversal_graph(
    sql: str,
    retrieved_columns: list[dict],
    join_conditions: list[str],
    question: str = "",
    height: int = 560,
    reranked_columns: list[dict] | None = None,
) -> None:
    """
    Render a Neo4j-style force-directed knowledge graph on a light canvas.

    Nodes
    -----
    - Large circle per TABLE, coloured by domain
    - Small circle per COLUMN that appears in the generated SQL, attached to
      its parent table via a thin HAS edge

    Edges
    -----
    - :REFERENCES  — thick directed arc between TABLE nodes (the KG traversal path)
    - :JOIN_ON     — distinct directed arc between the two specific COLUMN nodes
                     used as the join key in the SQL (e.g. CLAIMS.CLAIM_ID →
                     PAYMENT.CLAIM_ID), so the join column linkage is explicit
    - HAS          — thin line from TABLE → each of its COLUMN nodes

    Theme: light (#f8fafc canvas, white tooltips, dark text)
    """
    import streamlit.components.v1 as components
    import json

    if not sql or not sql.strip():
        st.caption("No SQL available — graph cannot be rendered.")
        return

    # ── 1. Parse SQL ──────────────────────────────────────────────────────────
    real_tables, alias_map = _parse_sql_tables_and_aliases(sql)
    if not real_tables:
        st.caption("No table traversal detected in the generated SQL.")
        return

    sql_col_refs = _extract_sql_columns(sql, real_tables, alias_map)
    fk_edges     = _extract_fk_edges(sql, join_conditions, real_tables, alias_map)

    # ── 2. Color palette — soft pastels on light background ───────────────────
    _DOMAIN_COLORS: dict[str, str] = {
        "CLAIMANT": "#a78bfa",   # soft lavender
        "CLAIM":    "#f87171",   # soft rose-red
        "POLICY":   "#60a5fa",   # soft sky blue
        "PAYMENT":  "#34d399",   # soft mint green
        "ADJUSTER": "#fbbf24",   # soft amber
    }
    _DOMAIN_BORDER: dict[str, str] = {
        "CLAIMANT": "#8b5cf6",
        "CLAIM":    "#ef4444",
        "POLICY":   "#3b82f6",
        "PAYMENT":  "#10b981",
        "ADJUSTER": "#f59e0b",
    }

    def _tbl_color(t: str) -> str:
        u = t.upper()
        for k, v in _DOMAIN_COLORS.items():
            if u == k or u.startswith(k + "_"): return v
        for k, v in _DOMAIN_COLORS.items():
            if k in u: return v
        return "#475569"

    def _tbl_border(t: str) -> str:
        u = t.upper()
        for k, v in _DOMAIN_BORDER.items():
            if u == k or u.startswith(k + "_"): return v
        for k, v in _DOMAIN_BORDER.items():
            if k in u: return v
        return "#1e293b"

    # ── 3. Build node + edge lists ────────────────────────────────────────────
    vis_nodes: list[dict] = []
    vis_edges: list[dict] = []
    node_id   = 0
    table_node_id:  dict[str, int]         = {}   # table_name → node id
    col_node_id_map: dict[tuple[str,str], int] = {}  # (TABLE, COL) → node id

    dst_tables = {dst for _, dst, _ in fk_edges}
    seen_order: set[str] = set()
    ordered_tables = (
        [t for t in sorted(real_tables) if t not in dst_tables] +
        [t for t in sorted(real_tables) if t in dst_tables]
    ) or sorted(real_tables)
    ordered_tables = [t for t in ordered_tables
                      if not (t in seen_order or seen_order.add(t))]  # type: ignore[func-returns-value]

    traversal_sequence: list[int] = []

    # ── 3a. TABLE nodes + their COLUMN nodes + HAS edges ─────────────────────
    for tbl in ordered_tables:
        color  = _tbl_color(tbl)
        border = _tbl_border(tbl)
        tbl_sql_cols = sorted(c for (t, c) in sql_col_refs if t == tbl)

        col_rows_html = "".join(
            f"<div style='font-family:monospace;font-size:10px;color:#374151;"
            f"padding:1px 0;white-space:nowrap'>"
            f"<span style='color:{color};font-weight:700'>·</span> {c}"
            f"</div>"
            for c in tbl_sql_cols
        ) or "<div style='color:#9ca3af;font-size:10px'>No explicit cols (SELECT *)</div>"

        # Table node tooltip (light card)
        tbl_tooltip = (
            f"<div style='font-family:Inter,system-ui,sans-serif;"
            f"background:#ffffff;border:2px solid {color};"
            f"border-radius:12px;padding:12px 16px;min-width:220px;"
            f"box-shadow:0 8px 24px rgba(0,0,0,0.12)'>"
            f"<div style='font-size:13px;font-weight:700;color:{color};"
            f"margin-bottom:8px;border-bottom:1px solid {color}33;padding-bottom:6px;"
            f"display:flex;align-items:center;gap:8px'>"
            f"<span style='display:inline-block;width:14px;height:14px;"
            f"border-radius:50%;background:{color}'></span>"
            f":{tbl}</div>"
            f"<div style='color:#6b7280;font-size:10px;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px'>"
            f"Properties in query ({len(tbl_sql_cols)})</div>"
            f"{col_rows_html}</div>"
        )

        tbl_nid = node_id
        table_node_id[tbl] = tbl_nid
        traversal_sequence.append(tbl_nid)

        vis_nodes.append({
            "id":    tbl_nid,
            "label": tbl,
            "group": "table",
            "shape": "dot",
            "size":  38,
            "color": {
                "background": color,
                "border":     border,
                "highlight":  {"background": color,  "border": "#1e293b"},
                "hover":      {"background": color,  "border": "#1e293b"},
            },
            "font": {
                "size": 13, "bold": True, "color": "#ffffff",
                "face": "Inter,system-ui,sans-serif",
                "strokeWidth": 3, "strokeColor": border,
            },
            "shadow": {"enabled": True, "color": color + "33", "size": 10, "x": 0, "y": 2},
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "title": tbl_tooltip,
        })
        node_id += 1

        # Column nodes
        for col in tbl_sql_cols:
            col_tooltip = (
                f"<div style='font-family:Inter;background:#ffffff;"
                f"border:2px solid {color};border-radius:10px;"
                f"padding:10px 14px;min-width:180px;"
                f"box-shadow:0 4px 16px rgba(0,0,0,0.10);font-size:11px'>"
                f"<div style='color:{color};font-weight:700;font-size:11px;"
                f"margin-bottom:4px'>:{tbl}</div>"
                f"<div style='color:#111827;font-family:monospace;font-size:13px;"
                f"font-weight:700'>{col}</div>"
                f"</div>"
            )
            col_label = col if len(col) <= 12 else col[:11] + "…"
            col_nid = node_id
            col_node_id_map[(tbl, col)] = col_nid
            vis_nodes.append({
                "id":    col_nid,
                "label": col_label,
                "group": "column",
                "shape": "dot",
                "size":  13,
                "color": {
                    "background": "#fef3c7",
                    "border":     color,
                    "highlight":  {"background": "#fef3c7", "border": "#9ca3af"},
                    "hover":      {"background": "#fef3c7", "border": color},
                },
                "font": {
                    "size": 9, "bold": False, "color": "#6b7280",
                    "face": "monospace",
                    "strokeWidth": 2, "strokeColor": "#ffffff",
                },
                "shadow": {"enabled": True, "color": color + "22", "size": 4, "x": 0, "y": 1},
                "borderWidth": 1.2,
                "title": col_tooltip,
            })
            # HAS edge: TABLE → COLUMN
            vis_edges.append({
                "from":  tbl_nid,
                "to":    col_nid,
                "label": "HAS",
                "color": {"color": color + "44", "highlight": color + "88", "hover": color + "88"},
                "width": 0.8,
                "dashes": False,
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.45, "type": "arrow"}},
                "font": {
                    "size": 7, "color": color + "bb", "bold": False,
                    "face": "Inter,system-ui,sans-serif",
                    "strokeWidth": 0, "align": "middle",
                    "background": "rgba(248,250,252,0.85)",
                },
                "smooth": {"type": "dynamic"},
                "physics": True,
                "length": 95,
                "selectable": True,
            })
            node_id += 1

    # ── 3b. :REFERENCES edges — TABLE → TABLE (traversal path) ───────────────
    fk_edge_ids: list[int] = []
    _JOIN_PALETTE = ["#fbbf24", "#67e8f9", "#c4b5fd", "#f9a8d4", "#86efac"]

    for j_idx, (src, dst, pred) in enumerate(fk_edges):
        if src not in table_node_id or dst not in table_node_id:
            continue

        join_key   = pred.split("=")[0].strip().split(".")[-1] if "=" in pred else pred
        edge_color = _JOIN_PALETTE[j_idx % len(_JOIN_PALETTE)]
        lhs, rhs   = (pred.split("=") + [""])[:2]

        ref_tooltip = (
            f"<div style='font-family:Inter;background:#ffffff;"
            f"border:2px solid {edge_color};border-radius:12px;"
            f"padding:12px 16px;min-width:240px;"
            f"box-shadow:0 8px 24px rgba(0,0,0,0.12);font-size:11px'>"
            f"<div style='font-weight:700;color:{edge_color};font-size:13px;"
            f"margin-bottom:10px;display:flex;align-items:center;gap:8px'>"
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{edge_color}'></span>"
            f":REFERENCES</div>"
            f"<div style='background:#f1f5f9;border:1px solid #e2e8f0;"
            f"border-radius:8px;padding:8px 12px;font-family:monospace;"
            f"font-size:11px;color:#111827;white-space:nowrap'>"
            f"{lhs.strip()} "
            f"<span style='color:{edge_color};font-weight:700'>=</span> "
            f"{rhs.strip()}"
            f"</div>"
            f"<div style='color:#6b7280;font-size:10px;margin-top:8px'>"
            f"KG traversal · join key: <b style='color:#111827'>{join_key}</b></div>"
            f"</div>"
        )

        eid = node_id + j_idx
        vis_edges.append({
            "id":    eid,
            "from":  table_node_id[src],
            "to":    table_node_id[dst],
            "label": f":REFERENCES\n{join_key}",
            "title": ref_tooltip,
            "color": {"color": edge_color, "highlight": "#64748b",
                      "hover": "#64748b", "opacity": 0.75},
            "width":  2,
            "dashes": False,
            "arrows": {"to": {"enabled": True, "scaleFactor": 1.1, "type": "arrow"}},
            "font": {
                "size": 10, "color": edge_color + "cc", "bold": False,
                "face": "Inter,system-ui,sans-serif",
                "strokeWidth": 2, "strokeColor": "#f8fafc",
                "align": "middle",
                "background": "rgba(248,250,252,0.90)",
                "multi": True,
            },
            "shadow": {"enabled": True, "color": edge_color + "22", "size": 4, "x": 0, "y": 0},
            "smooth": {"type": "curvedCW", "roundness": 0.25},
            "length": 260,
            "physics": True,
        })
        fk_edge_ids.append(eid)

    # ── 3c. :JOIN_ON edges — COLUMN → COLUMN (the actual join key linkage) ────
    # Parse each FK predicate into (src_table, src_col, dst_table, dst_col) and
    # draw a distinct dashed directed edge between the two column nodes so the
    # user can see exactly which property connects the two table nodes.
    _JOIN_ON_PALETTE = ["#7dd3fc", "#c4b5fd", "#fda4af", "#6ee7b7", "#fed7aa"]
    join_on_edge_ids: list[int] = []
    _pred_re = re.compile(
        r'\b([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)'
        r'\s*=\s*'
        r'([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\b'
    )
    edge_id_counter = node_id + len(fk_edges) + 10   # safe offset

    for j_idx, (src_tbl, dst_tbl, pred) in enumerate(fk_edges):
        m = _pred_re.search(pred.upper())
        if not m:
            continue
        # Both sides of the equality
        sides = [(m.group(1), m.group(2)), (m.group(3), m.group(4))]
        # Resolve alias → real table for each side
        resolved = [(_resolve_table(t, alias_map), c) for t, c in sides]
        # We need both column nodes to exist in the graph
        src_pair = resolved[0] if resolved[0][0] == src_tbl else resolved[1]
        dst_pair = resolved[1] if resolved[1][0] == dst_tbl else resolved[0]

        src_cnid = col_node_id_map.get(src_pair)
        dst_cnid = col_node_id_map.get(dst_pair)

        # If one or both join-key columns weren't in sql_col_refs (e.g. only
        # used in the ON clause, not SELECTed), they won't have a node yet —
        # create a lightweight phantom column node so the join is still visible.
        join_col_color = _JOIN_ON_PALETTE[j_idx % len(_JOIN_ON_PALETTE)]

        for pair, existing_nid in [(src_pair, src_cnid), (dst_pair, dst_cnid)]:
            if existing_nid is None and pair[0] in real_tables:
                phantom_tbl, phantom_col = pair
                p_color  = _tbl_color(phantom_tbl)
                p_border = _tbl_border(phantom_tbl)
                phantom_nid = edge_id_counter
                edge_id_counter += 1
                col_node_id_map[pair] = phantom_nid
                col_label = phantom_col if len(phantom_col) <= 12 else phantom_col[:11] + "…"
                vis_nodes.append({
                    "id":    phantom_nid,
                    "label": col_label,
                    "group": "column",
                    "shape": "dot",
                    "size":  13,
                    "color": {
                        "background": "#fef3c7",
                        "border":     p_color,
                        "highlight":  {"background": "#fef3c7", "border": "#9ca3af"},
                        "hover":      {"background": "#fef3c7", "border": p_color},
                    },
                    "font": {
                        "size": 9, "bold": False, "color": "#6b7280",
                        "face": "monospace",
                        "strokeWidth": 2, "strokeColor": "#ffffff",
                    },
                    "shadow": {"enabled": True, "color": p_color + "22", "size": 4, "x": 0, "y": 1},
                    "borderWidth": 1.5,
                    "title": (
                        f"<div style='font-family:Inter;background:#ffffff;"
                        f"border:2px solid {p_color};border-radius:10px;"
                        f"padding:10px 14px;min-width:180px;"
                        f"box-shadow:0 4px 16px rgba(0,0,0,0.10);font-size:11px'>"
                        f"<div style='color:{p_color};font-weight:700;font-size:11px;"
                        f"margin-bottom:4px'>:{phantom_tbl}</div>"
                        f"<div style='color:#111827;font-family:monospace;font-size:13px;"
                        f"font-weight:700'>{phantom_col}</div>"
                        f"<div style='color:#6b7280;font-size:10px;margin-top:4px'>"
                        f"Join key column</div></div>"
                    ),
                })
                # HAS edge from parent table to phantom column
                if phantom_tbl in table_node_id:
                    vis_edges.append({
                        "from":  table_node_id[phantom_tbl],
                        "to":    phantom_nid,
                        "label": "HAS",
                        "color": {"color": p_color + "44", "highlight": p_color + "88", "hover": p_color + "88"},
                        "width": 0.8,
                        "dashes": False,
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.45, "type": "arrow"}},
                        "font": {
                            "size": 7, "color": p_color + "bb", "bold": False,
                            "face": "Inter,system-ui,sans-serif",
                            "strokeWidth": 0, "align": "middle",
                            "background": "rgba(248,250,252,0.85)",
                        },
                        "smooth": {"type": "dynamic"},
                        "physics": True,
                        "length": 95,
                        "selectable": True,
                    })

        # Re-fetch after possible phantom creation
        src_cnid = col_node_id_map.get(src_pair)
        dst_cnid = col_node_id_map.get(dst_pair)
        if src_cnid is None or dst_cnid is None:
            continue

        lhs_str = f"{src_pair[0]}.{src_pair[1]}"
        rhs_str = f"{dst_pair[0]}.{dst_pair[1]}"
        join_on_tooltip = (
            f"<div style='font-family:Inter;background:#ffffff;"
            f"border:2px solid {join_col_color};border-radius:12px;"
            f"padding:12px 16px;min-width:240px;"
            f"box-shadow:0 8px 24px rgba(0,0,0,0.12);font-size:11px'>"
            f"<div style='font-weight:700;color:{join_col_color};font-size:13px;"
            f"margin-bottom:10px;display:flex;align-items:center;gap:8px'>"
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{join_col_color}'></span>"
            f":JOIN_ON</div>"
            f"<div style='background:#f1f5f9;border:1px solid #e2e8f0;"
            f"border-radius:8px;padding:8px 12px;font-family:monospace;"
            f"font-size:11px;color:#111827;white-space:nowrap'>"
            f"{lhs_str} "
            f"<span style='color:{join_col_color};font-weight:700'>=</span> "
            f"{rhs_str}"
            f"</div>"
            f"<div style='color:#6b7280;font-size:10px;margin-top:8px'>"
            f"SQL join condition — column-level linkage</div>"
            f"</div>"
        )

        join_on_eid = edge_id_counter
        edge_id_counter += 1
        join_on_edge_ids.append(join_on_eid)
        vis_edges.append({
            "id":    join_on_eid,
            "from":  src_cnid,
            "to":    dst_cnid,
            "label": ":JOIN_ON",
            "title": join_on_tooltip,
            "color": {"color": join_col_color, "highlight": "#64748b",
                      "hover": "#64748b", "opacity": 0.80},
            "width":  1.5,
            "dashes": [5, 4],
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8, "type": "arrow"}},
            "font": {
                "size": 9, "color": join_col_color + "cc", "bold": False,
                "face": "Inter,system-ui,sans-serif",
                "strokeWidth": 2, "strokeColor": "#f8fafc",
                "align": "middle",
                "background": "rgba(248,250,252,0.90)",
            },
            "shadow": {"enabled": True, "color": join_col_color + "22",
                       "size": 4, "x": 0, "y": 0},
            "smooth": {"type": "curvedCCW", "roundness": 0.3},
            "length": 160,
            "physics": True,
        })

    nodes_json         = json.dumps(vis_nodes)
    edges_json         = json.dumps(vis_edges)
    traversal_json     = json.dumps(traversal_sequence)
    fk_edge_ids_json   = json.dumps(fk_edge_ids)
    join_on_ids_json   = json.dumps(join_on_edge_ids)

    # ── 4. Footer stats & legend ──────────────────────────────────────────────
    n_tables   = len(real_tables)
    n_refs     = len(fk_edges)
    n_cols     = len(sql_col_refs)
    n_join_ons = len(join_on_edge_ids)

    legend_html = ""
    for tbl in ordered_tables:
        c  = _tbl_color(tbl)
        bd = _tbl_border(tbl)
        n_c = sum(1 for (t, _) in sql_col_refs if t == tbl)
        legend_html += (
            f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:14px'>"
            f"<span style='width:12px;height:12px;border-radius:50%;"
            f"background:{c};border:2px solid {bd};"
            f"display:inline-block;flex-shrink:0'></span>"
            f"<span style='color:#1e293b;font-weight:600;font-size:11px'>:{tbl}</span>"
            f"<span style='color:#6b7280;font-size:10px'>({n_c} cols)</span></span>"
        )

    # ── 5. Full HTML — light canvas, force-directed physics ───────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet"/>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ width:100%; height:100%; background:#f8fafc; overflow:hidden; }}
  #graph {{ width:100%; height:{height}px; background:#f8fafc; }}
  div.vis-tooltip {{
    background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 !important;
  }}
  #footer {{
    position:absolute; bottom:0; left:0; right:0; z-index:10;
    background:rgba(248,250,252,0.97); border-top:1px solid #e2e8f0;
    padding:6px 14px; display:flex; align-items:center;
    justify-content:space-between; flex-wrap:wrap; gap:4px;
    font-family:Inter,system-ui,sans-serif; font-size:10.5px;
    box-shadow:0 -2px 8px rgba(0,0,0,0.06);
  }}
  #hint {{
    position:absolute; top:10px; left:50%; transform:translateX(-50%); z-index:10;
    background:rgba(255,255,255,0.92); border:1px solid #e2e8f0; color:#475569;
    border-radius:20px; padding:4px 16px;
    font-size:10px; font-family:Inter,system-ui,sans-serif;
    pointer-events:none; white-space:nowrap;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
  }}
  #badge {{
    display:none; position:absolute; top:10px; right:12px; z-index:10;
    background:#f0fdf4; border:1px solid #86efac; color:#166534;
    border-radius:20px; padding:4px 12px;
    font-size:9.5px; font-family:Inter,system-ui,sans-serif;
  }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="hint">Hover nodes &amp; edges for details · Drag to rearrange · Scroll to zoom</div>
<div id="badge">● Graph stabilised</div>

<div id="footer">
  <span style="color:#475569;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <span><b style="color:#1e293b">{n_tables}</b> node label(s)</span>
    <span style="color:#cbd5e1">·</span>
    <span><b style="color:#1e293b">{n_refs}</b> :REFERENCES</span>
    <span style="color:#cbd5e1">·</span>
    <span><b style="color:#1e293b">{n_join_ons}</b> :JOIN_ON</span>
    <span style="color:#cbd5e1">·</span>
    <span><b style="color:#1e293b">{n_cols}</b> property ref(s)</span>
    &nbsp;
    <span style="display:inline-flex;align-items:center;gap:4px">
      <span style="width:22px;height:2px;background:#fbbf24;display:inline-block;border-radius:2px;opacity:0.85"></span>
      <span style="color:#64748b">:REFERENCES</span>
    </span>
    <span style="display:inline-flex;align-items:center;gap:4px">
      <span style="width:18px;height:0;border-top:2px dashed #7dd3fc;display:inline-block;opacity:0.85"></span>
      <span style="color:#64748b">:JOIN_ON</span>
    </span>
    <span style="display:inline-flex;align-items:center;gap:4px">
      <span style="width:14px;height:1px;background:#cbd5e1;display:inline-block"></span>
      <span style="color:#94a3b8">HAS</span>
    </span>
  </span>
  <span style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">{legend_html}</span>
</div>

<script>
(function() {{
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var container = document.getElementById('graph');

  var options = {{
    layout: {{
      improvedLayout: true,
      hierarchical: {{ enabled: false }}
    }},
    physics: {{
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{
        gravitationalConstant: -80,
        centralGravity: 0.01,
        springLength: 160,
        springConstant: 0.06,
        damping: 0.5,
        avoidOverlap: 0.9,
      }},
      stabilization: {{
        enabled: true,
        iterations: 300,
        updateInterval: 25,
        fit: true,
      }},
      maxVelocity: 50,
      minVelocity: 0.5,
      timestep: 0.4,
    }},
    interaction: {{
      hover: true,
      tooltipDelay: 60,
      zoomView: true,
      dragNodes: true,
      dragView: true,
      navigationButtons: false,
      keyboard: false,
      multiselect: false,
    }},
    nodes: {{
      borderWidthSelected: 4,
    }},
    edges: {{
      selectionWidth: 4,
      hoverWidth: 2,
      font: {{
        size: 10,
        strokeWidth: 3,
        strokeColor: '#f8fafc',
        align: 'middle',
        face: 'Inter,system-ui,sans-serif',
        multi: true,
      }},
      smooth: {{ type: 'dynamic' }},
    }},
  }};

  var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);

  // ── Traversal pulse: flash table nodes then REFERENCES edges ─────────────
  var traversalSeq  = {traversal_json};
  var fkEdgeIds     = {fk_edge_ids_json};
  var joinOnEdgeIds = {join_on_ids_json};

  function pulseTraversal() {{
    var delay = 0;
    traversalSeq.forEach(function(nid) {{
      setTimeout(function() {{
        var n = nodes.get(nid);
        if (!n) return;
        var origBorder = n.borderWidth || 2;
        var origShadow = n.shadow || {{ enabled: true }};
        nodes.update({{ id: nid, borderWidth: 7,
          shadow: {{ enabled: true, color: '#94a3b844', size: 20, x: 0, y: 0 }} }});
        setTimeout(function() {{
          nodes.update({{ id: nid, borderWidth: origBorder, shadow: origShadow }});
        }}, 380);
      }}, delay);
      delay += 420;
    }});
    // Flash :REFERENCES edges
    setTimeout(function() {{
      fkEdgeIds.forEach(function(eid, i) {{
        setTimeout(function() {{
          var e = edges.get(eid);
          if (!e) return;
          var origW = e.width || 3;
          var origC = e.color;
          edges.update({{ id: eid, width: origW + 5,
            color: {{ color: '#1e293b', opacity: 1.0 }} }});
          setTimeout(function() {{
            edges.update({{ id: eid, width: origW, color: origC }});
          }}, 360);
        }}, i * 420);
      }});
    }}, delay + 100);
    // Flash :JOIN_ON edges after REFERENCES
    setTimeout(function() {{
      joinOnEdgeIds.forEach(function(eid, i) {{
        setTimeout(function() {{
          var e = edges.get(eid);
          if (!e) return;
          var origW = e.width || 2;
          var origC = e.color;
          edges.update({{ id: eid, width: origW + 4,
            color: {{ color: '#1e293b', opacity: 1.0 }} }});
          setTimeout(function() {{
            edges.update({{ id: eid, width: origW, color: origC }});
          }}, 360);
        }}, i * 380);
      }});
    }}, delay + fkEdgeIds.length * 420 + 300);
  }}

  network.once('stabilized', function() {{
    document.getElementById('badge').style.display = 'block';
    document.getElementById('hint').style.display  = 'none';
    network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
    network.setOptions({{ physics: {{ enabled: false }} }});
    setTimeout(pulseTraversal, 800);
  }});
  network.on('stabilizationIterationsDone', function() {{
    document.getElementById('badge').style.display = 'block';
    network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
    network.setOptions({{ physics: {{ enabled: false }} }});
    setTimeout(pulseTraversal, 800);
  }});
}})();
</script>
</body>
</html>"""

    components.html(html, height=height + 6, scrolling=False)


def _format_results_df_for_display(df: pd.DataFrame, max_items: int = 8) -> pd.DataFrame:
    """
    Return a display-only copy of a results DataFrame with list/tuple-valued
    cells (e.g. a `claim_ids` column from a clustering query) rendered as a
    clean, readable string instead of Streamlit's default one-chip-per-item
    layout, which looks cluttered for long lists.

    Short lists are shown fully comma-separated; long lists are truncated
    with a "+N more" suffix. The raw `df` (with real Python lists) is left
    untouched, so CSV export and the AI summary still see the full data.
    """
    def _format_cell(value):
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            if len(items) > max_items:
                shown = ", ".join(str(v) for v in items[:max_items])
                return f"{shown}, +{len(items) - max_items} more"
            return ", ".join(str(v) for v in items)
        return value

    has_list_col = any(
        df[col].apply(lambda v: isinstance(v, (list, tuple, set))).any()
        for col in df.columns
    )
    if not has_list_col:
        return df

    display_df = df.copy()
    for col in display_df.columns:
        if display_df[col].apply(lambda v: isinstance(v, (list, tuple, set))).any():
            display_df[col] = display_df[col].apply(_format_cell)
    return display_df


def _generate_table_summary(df: pd.DataFrame, question: str, pipeline_summary: str = "") -> str:
    """
    Return a plain-English summary grounded in the actual result DataFrame.

    Strategy
    --------
    The summary is always built directly from the DataFrame statistics to ensure
    it is accurate and never hallucinated. The pipeline LLM summary is ignored
    because it may reference values or counts inconsistent with the actual data.
    """
    if df is None or df.empty:
        return ""

    n_rows = len(df)
    n_cols = len(df.columns)

    def _readable(col: str) -> str:
        return (
            col.replace("_", " ")
               .replace("AMT", "Amount")
               .replace("CD", "Code")
               .replace("STAT", "Status")
               .replace("NO", "Number")
               .replace("DT", "Date")
               .title()
               .strip()
        )

    # ── Opening sentence ──────────────────────────────────────────────────────
    col_list = ", ".join(_readable(c) for c in df.columns[:6])
    col_hint = f" covering {col_list}{' and more' if n_cols > 6 else ''}" if n_cols > 0 else ""
    summary_parts = [
        f"The query returned {n_rows:,} record(s) across {n_cols} column(s){col_hint}."
    ]

    # ── Numeric columns ───────────────────────────────────────────────────────
    num_sentences: list[str] = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if s.empty:
            continue
        readable = _readable(col)
        total = s.sum()
        lo, hi = s.min(), s.max()
        avg = s.mean()
        col_upper = col.upper()
        if ("AMT" in col_upper or "AMOUNT" in col_upper
                or "RESERVE" in col_upper or "INCURRED" in col_upper
                or "PAYMENT" in col_upper):
            num_sentences.append(
                f"The {readable} ranges from ${lo:,.2f} to ${hi:,.2f} "
                f"(average ${avg:,.2f}, total ${total:,.2f})."
            )
        elif "SCORE" in col_upper or "RATE" in col_upper or "PCT" in col_upper:
            num_sentences.append(
                f"The {readable} spans {lo:.2f} to {hi:.2f} with an average of {avg:.2f}."
            )
        elif "COUNT" in col_upper or "NUM" in col_upper or "NBR" in col_upper:
            num_sentences.append(
                f"The {readable} ranges from {int(lo):,} to {int(hi):,} (average {avg:,.1f})."
            )

    summary_parts.extend(num_sentences[:4])

    # ── Categorical columns ───────────────────────────────────────────────────
    cat_sentences: list[str] = []
    for col in df.select_dtypes(exclude="number").columns[:5]:
        vc = df[col].dropna().value_counts()
        if vc.empty or len(vc) > 25:
            continue
        readable = _readable(col)
        n_unique = len(vc)
        top_val, top_count = vc.index[0], vc.iloc[0]
        top_pct = top_count / n_rows * 100

        if n_unique == 1:
            cat_sentences.append(
                f"All {n_rows:,} records have {readable} = \"{top_val}\"."
            )
        else:
            top_items = ", ".join(
                f"{v} ({c:,} record{'s' if c != 1 else ''}, {c / n_rows * 100:.0f}%)"
                for v, c in vc.head(4).items()
            )
            cat_sentences.append(
                f"The {readable} field has {n_unique} distinct value(s) — "
                f"the most common is \"{top_val}\" ({top_count:,} record{'s' if top_count != 1 else ''}, "
                f"{top_pct:.0f}%). Breakdown: {top_items}."
            )

    summary_parts.extend(cat_sentences[:3])

    # ── Date columns ─────────────────────────────────────────────────────────
    date_sentences: list[str] = []
    for col in df.columns:
        col_upper = col.upper()
        if "DT" not in col_upper and "DATE" not in col_upper:
            continue
        col_series = df[col].dropna()
        if col_series.empty:
            continue
        try:
            parsed = pd.to_datetime(col_series, errors="coerce").dropna()
            if parsed.empty:
                continue
            readable = _readable(col)
            date_sentences.append(
                f"The {readable} spans from {parsed.min().strftime('%b %d, %Y')} "
                f"to {parsed.max().strftime('%b %d, %Y')}."
            )
        except Exception:
            continue

    summary_parts.extend(date_sentences[:2])

    return " ".join(summary_parts)


# ── Repair-agent trace renderer ────────────────────────────────────────────────
def _render_sql_repair_trace(repair_detail: list, full_trace: str = "") -> None:
    """Render the full self-healing repair trace for the SQL lane: one
    expandable block per repair cycle, each showing exactly what failed,
    why, what the repair agent changed, and whether that fix was accepted —
    plus a clear Before (failing SQL) / After (repaired SQL) comparison.
    """
    if not repair_detail:
        return

    st.markdown(f"**Repair Trace — {len(repair_detail)} repair cycle(s) run:**")

    for ra in repair_detail:
        passed = getattr(ra, "validation_passed", True)
        terminated = getattr(ra, "terminated_retries", False)
        status_icon = "🛑" if terminated else ("✅" if passed else "⚠️")
        status_text = (
            "stopped retry loop" if terminated
            else ("fix accepted, retrying" if passed else "fix rejected by validation")
        )

        with st.expander(
            f"{status_icon} Repair cycle #{ra.attempt_number} — {status_text}",
            expanded=terminated,
        ):
            st.markdown("**What failed:**")
            st.markdown(
                f"<div class='repair-error-box'>{html.escape(ra.error)}</div>",
                unsafe_allow_html=True,
            )

            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("**Before — failing SQL:**")
                st.markdown(
                    f"<div class='repair-sql-box repair-sql-before'>"
                    f"{html.escape(ra.failing_sql)}</div>",
                    unsafe_allow_html=True,
                )
            with col_after:
                st.markdown("**After — repaired SQL:**")
                st.markdown(
                    f"<div class='repair-sql-box repair-sql-after'>"
                    f"{html.escape(ra.repaired_sql or '(empty response)')}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**What changed (diff):**")
            st.markdown(
                f"<div class='repair-diff-box'>"
                f"{html.escape(ra.sql_diff or '(no diff — SQL unchanged)')}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**Validation:**")
            for note in getattr(ra, "validation_notes", []):
                css = "repair-note-pass" if passed else "repair-note-fail"
                st.markdown(f"<span class='{css}'>• {html.escape(note)}</span>",
                            unsafe_allow_html=True)
            if terminated:
                st.error(f"Retry loop stopped: {ra.termination_reason}")

            with st.expander("Exact prompt sent to the repair LLM"):
                st.markdown("*System prompt:*")
                st.code(ra.repair_system_prompt, language="text")
                st.markdown("*User prompt:*")
                st.code(ra.repair_user_prompt, language="text")

    if full_trace:
        with st.expander("Full raw repair trace (text report)"):
            st.code(full_trace, language="text")


# ── Trace panel renderer ──────────────────────────────────────────────────────
def _render_stage_timings(trace) -> None:
    """
    Render the actual, measured wall-clock time each pipeline stage took for
    this query, as a compact horizontal bar breakdown plus a total. Uses the
    same violet accent as the rest of the trace UI so it reads as one theme
    regardless of which lane (SQL or Cypher) produced it.
    """
    timings = getattr(trace, "stage_timings", None)
    if not timings:
        return

    total   = sum(timings.values()) or 1e-9
    max_val = max(timings.values()) or 1e-9

    st.markdown("**Stage-wise Actual Execution Time:**")
    rows_html = ["<div style='display:flex; flex-direction:column; gap:6px; margin-bottom:0.5rem;'>"]
    for stage, seconds in timings.items():
        bar_pct = max(2, round((seconds / max_val) * 100))
        share_pct = (seconds / total) * 100
        rows_html.append(
            "<div style='display:flex; align-items:center; gap:10px;'>"
            f"<div style='min-width:270px; font-size:0.8rem; color:#334155;'>{stage}</div>"
            "<div style='flex:1; background:#f1f5f9; border-radius:6px; height:16px; overflow:hidden;'>"
            f"<div style='width:{bar_pct}%; background:#8b5cf6; height:100%; border-radius:6px;'></div>"
            "</div>"
            f"<div style='min-width:100px; text-align:right; font-size:0.78rem; "
            f"font-weight:600; color:#5b21b6;'>{seconds:.2f}s &nbsp;({share_pct:.0f}%)</div>"
            "</div>"
        )
    rows_html.append("</div>")
    st.markdown("\n".join(rows_html), unsafe_allow_html=True)
    st.caption(f"Total measured: {total:.2f}s across {len(timings)} stage(s).")
    st.divider()


def _render_cypher_trace_panel(last_r, last_q: str) -> None:
    """Query Trace expander for the Cypher / Knowledge Graph lane."""
    with st.expander("Query Trace", expanded=True):
        t = last_r.trace

        st.markdown("**Stage -1 — Lane Routing:**")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.markdown(
                f"<div class='metric-card'><b>Lane</b><br>Cypher / Knowledge Graph</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='metric-card'><b>Method</b><br>{getattr(t, 'route_method', '—')}</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"<div class='metric-card'><b>Reason</b><br>{getattr(t, 'route_reason', '—')}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("**Stage 1 — Graph Schema Grounding:**")
        st.caption(
            "Business Knowledge Graph in Neo4j: "
            "(:Claimant)-[:FILED]->(:Claim)-[:COVERED_BY]->(:Policy), "
            "(:Claim)-[:HAS_PAYMENT]->(:Payment), plus SAME_ADJUSTER_AS / "
            "SAME_AGENT_AS relationship edges for multi-hop questions."
        )

        st.markdown("**Live Traversal Graph (Knowledge Graph):**")
        _render_kg_traversal_graph(t.generated_cypher or "", height=420)

        st.markdown(
            f"**Stage 2+3 — Generated & Executed Cypher** "
            f"(repair attempts: {getattr(t, 'cypher_repair_attempts', 0)}):"
        )
        st.markdown(
            f"<div class='cypher-box'>{t.generated_cypher}</div>",
            unsafe_allow_html=True,
        )

        if getattr(t, "cypher_repair_history", None):
            st.markdown("**Repair History:**")
            for i, r in enumerate(t.cypher_repair_history, 1):
                with st.expander(f"Attempt {i} failed"):
                    st.code(r.get("cypher", ""), language="cypher")
                    st.error(r.get("error", ""))


def _render_trace_panel(last_r, last_q: str) -> None:
    """Render the full Query Trace expander for one QueryResult."""
    if getattr(last_r.trace, "lane", "SQL") == "CYPHER":
        _render_cypher_trace_panel(last_r, last_q)
        return

    with st.expander("Query Trace", expanded=True):
        t = last_r.trace

        # ── Stage 0: Scoped Schema Retrieval ─────────────────────────────────
        st.markdown("**Stage 0 — Scoped Schema Retrieval:**")
        scoped_tables = getattr(t, "scoped_tables", [])
        method        = getattr(t, "scope_retrieval_method", "—")
        table_scores  = getattr(t, "table_scores", {})

        if scoped_tables:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                score_rows = [
                    {"table": tbl, "relevance_score": f"{table_scores.get(tbl, 0.0):.2f}"}
                    for tbl in scoped_tables
                ]
                st.dataframe(
                    pd.DataFrame(score_rows),
                    use_container_width=True,
                    height=min(200, 40 + len(scoped_tables) * 35),
                )
            with col_b:
                st.markdown(
                    f"<div class='metric-card'><b>Method</b><br>{method}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='metric-card'><b>Tables scoped</b><br>{len(scoped_tables)}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Stage 0 data not available (legacy result).")

        st.divider()

        # ── Stage 1: HyDE expansion ───────────────────────────────────────────
        st.markdown("**Stage 1 — HyDE Expanded Query:**")
        st.text_area(
            "", t.expanded_query or "", height=80,
            label_visibility="collapsed", key="hyde_out", disabled=True,
        )

        # ── Stage 2: reranked & deduplicated columns ──────────────────────────
        st.markdown("**Stage 2 — Retrieved Columns:**")
        if t.retrieved_columns:
            reranked = _rerank_columns(t.retrieved_columns, question=last_q)
            DISPLAY_SCORE_THRESHOLD = 0.75
            reranked_filtered = (
                [c for c in reranked if (c.get("score", 0) if isinstance(c, dict) else c.score) >= DISPLAY_SCORE_THRESHOLD]
                or reranked[:10]
            )
            df_cols = pd.DataFrame(reranked_filtered)
            display_cols, rename_map = [], {}
            for preferred, fallback in [
                ("table", None), ("name", "column_name"), ("description", None), ("score", None)
            ]:
                if preferred in df_cols.columns:
                    display_cols.append(preferred)
                elif fallback and fallback in df_cols.columns:
                    display_cols.append(fallback)
                    rename_map[fallback] = preferred
            df_display = df_cols[display_cols].rename(columns=rename_map)
            if "score" in df_display.columns:
                df_display = df_display.copy()
                df_display["score"] = df_display["score"].apply(
                    lambda s: f"{float(s):.1%}" if s is not None and s != "" else s
                )
            st.dataframe(df_display, use_container_width=True, height=250)

        # ── Stage 3: joins in generated SQL ──────────────────────────────────
        st.markdown("**Stage 3 — Join Conditions (present in generated SQL):**")
        sql_joins = _extract_sql_joins(t.generated_sql, t.join_conditions or [])
        if sql_joins:
            for jc in sql_joins:
                st.code(jc, language="sql")
        else:
            st.caption("No joins required.")

        # ── Live Traversal Graph ──────────────────────────────────────────────
        st.markdown("**Live Traversal Graph:**")
        _trace_reranked = _rerank_columns(t.retrieved_columns or [], question=last_q)
        _render_traversal_graph(
            sql=t.generated_sql or "",
            retrieved_columns=t.retrieved_columns or [],
            join_conditions=t.join_conditions or [],
            question=last_q,
            height=500,
            reranked_columns=_trace_reranked,
        )

        # ── Stage 4+5: Generated SQL ──────────────────────────────────────────
        st.markdown(
            f"**Stage 4+5 — Generated SQL** "
            f"(repair attempts: {t.sql_repair_attempts}):"
        )
        st.markdown(
            f"<div class='sql-box'>{html.escape(t.generated_sql or '')}</div>",
            unsafe_allow_html=True,
        )

        if t.sql_repair_attempts:
            st.divider()
            _render_sql_repair_trace(
                getattr(t, "sql_repair_detail", []),
                getattr(t, "sql_repair_full_trace", ""),
            )


SQL_LANE_STEPS = [
    "Schema Scoping", "HyDE Expansion", "KG Column Retrieval",
    "Join Discovery", "SQL Generation", "SQLite Execution",
]
CYPHER_LANE_STEPS = [
    "Relationship Detection", "Graph Schema Grounding",
    "Cypher Generation", "Neo4j Traversal", "Result Binding",
]


def _render_swimlanes(trace) -> None:
    """
    Render the two-swim-lane comparison diagram: Plain SQL vs Cypher /
    Knowledge Graph. The lane actually used for this question is
    highlighted; the other is dimmed, so the routing decision is visible
    at a glance.
    """
    lane = getattr(trace, "lane", "SQL")
    is_sql = (lane == "SQL")

    def _steps_html(steps: list[str]) -> str:
        parts = []
        for i, s in enumerate(steps):
            parts.append(f"<span class='lane-step'>{s}</span>")
            if i < len(steps) - 1:
                parts.append("<span class='lane-arrow'>&#8594;</span>")
        return "".join(parts)

    sql_row_class    = "lane-row lane-row-active" if is_sql else "lane-row lane-row-inactive"
    cypher_row_class = "lane-row lane-row-active cypher-active" if not is_sql else "lane-row lane-row-inactive"

    html = f"""
    <div class="swimlane-wrap">
        <div class="swimlane-title">
            <span>Plain SQL vs Knowledge Graph + Cypher</span>
            <span class="lane-pill {'lane-pill-sql' if is_sql else 'lane-pill-cypher'}">
                ACTIVE: {'SQL' if is_sql else 'CYPHER / KNOWLEDGE GRAPH'}
            </span>
        </div>
        <div class="{sql_row_class}">
            <div class="lane-name">&#128202; Plain SQL</div>
            <div class="lane-steps">{_steps_html(SQL_LANE_STEPS)}</div>
        </div>
        <div class="{cypher_row_class}">
            <div class="lane-name">&#128279; Cypher / KG</div>
            <div class="lane-steps">{_steps_html(CYPHER_LANE_STEPS)}</div>
        </div>
        <div class="lane-reason">
            <b>Routing decision</b> ({getattr(trace, 'route_method', '—')}):
            {getattr(trace, 'route_reason', 'No routing information available.')}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_kg_traversal_graph(cypher: str, height: int = 420) -> None:
    """
    Live traversal graph for the Cypher / Knowledge Graph lane.

    Shows the fixed business-graph schema (Claimant / Claim / Policy /
    Payment + their real relationships as ingested into Neo4j by
    kg/graph_ingest.py) and highlights the node labels and relationship
    types that the *actual generated Cypher* touches, so the visual reflects
    the real query rather than a generic diagram.
    """
    import streamlit.components.v1 as components
    import json as _json

    if not cypher or not cypher.strip():
        st.caption("No Cypher available — graph cannot be rendered.")
        return

    _DOMAIN_COLORS = {
        "Claimant": ("#a78bfa", "#8b5cf6"),
        "Claim":    ("#f87171", "#ef4444"),
        "Policy":   ("#60a5fa", "#3b82f6"),
        "Payment":  ("#34d399", "#10b981"),
    }
    _RELS = [
        ("Claimant", "Claim",   "FILED"),
        ("Claim",    "Policy",  "COVERED_BY"),
        ("Claim",    "Payment", "HAS_PAYMENT"),
        ("Claim",    "Claim",   "SAME_ADJUSTER_AS"),
        ("Policy",   "Policy",  "SAME_AGENT_AS"),
    ]

    used_labels = {lbl for lbl in _DOMAIN_COLORS if re.search(rf"\b{lbl}\b", cypher)}
    used_rels   = {r[2] for r in _RELS if re.search(rf"\b{r[2]}\b", cypher)}
    hop_match   = re.search(r"\*\s*\d*\s*\.\.\s*\d+", cypher) or re.search(r"shortestPath", cypher, re.IGNORECASE)

    vis_nodes, vis_edges = [], []
    node_id_map = {}
    for i, (label, (bg, border)) in enumerate(_DOMAIN_COLORS.items()):
        active = label in used_labels
        node_id_map[label] = i
        vis_nodes.append({
            "id": i, "label": label, "shape": "dot",
            "size": 34 if active else 24,
            "color": {
                "background": bg if active else "#e5e7eb",
                "border": border if active else "#cbd5e1",
            },
            "font": {"size": 13 if active else 11, "bold": active,
                     "color": "#ffffff" if active else "#94a3b8",
                     "face": "Inter,system-ui,sans-serif",
                     "strokeWidth": 3, "strokeColor": border if active else "#cbd5e1"},
            "borderWidth": 3 if active else 1,
            "shadow": {"enabled": active, "color": bg + "55", "size": 12},
        })

    for src, dst, rel in _RELS:
        active = rel in used_rels
        vis_edges.append({
            "from": node_id_map[src], "to": node_id_map[dst],
            "label": f":{rel}",
            "color": {"color": "#7c3aed" if active else "#e2e8f0"},
            "width": 3 if active else 1,
            "dashes": not active,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "font": {"size": 10, "color": "#7c3aed" if active else "#94a3b8",
                     "strokeWidth": 2, "strokeColor": "#f8fafc"},
        })

    hop_badge = ""
    if hop_match:
        hop_badge = (
            "<span style='background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;"
            "border-radius:999px;padding:2px 10px;font-size:10.5px;font-weight:700;'>"
            "Variable-length / shortest-path traversal detected</span>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet"/>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ width:100%; height:100%; background:#f8fafc; overflow:hidden; font-family:Inter,system-ui,sans-serif; }}
  #graph {{ width:100%; height:{height - 34}px; background:#f8fafc; }}
  #footer {{ padding:6px 10px; font-size:11px; color:#475569; display:flex; gap:10px; align-items:center; }}
</style></head>
<body>
<div id="graph"></div>
<div id="footer">
  <span>Highlighted = labels/relationships used by the executed Cypher</span>
  {hop_badge}
</div>
<script>
(function() {{
  var nodes = new vis.DataSet({_json.dumps(vis_nodes)});
  var edges = new vis.DataSet({_json.dumps(vis_edges)});
  var container = document.getElementById('graph');
  var options = {{
    physics: {{ enabled: true, solver: 'forceAtlas2Based',
      forceAtlas2Based: {{ gravitationalConstant: -60, springLength: 140, springConstant: 0.07 }},
      stabilization: {{ enabled: true, iterations: 200 }} }},
    interaction: {{ hover: true, dragNodes: true, dragView: true, zoomView: true }},
    edges: {{ smooth: {{ type: 'dynamic' }} }},
  }};
  var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
  network.once('stabilizationIterationsDone', function() {{
    network.fit({{ animation: {{ duration: 500 }} }});
    network.setOptions({{ physics: {{ enabled: false }} }});
  }});
}})();
</script>
</body></html>"""
    components.html(html, height=height, scrolling=False)


# ── Results area ──────────────────────────────────────────────────────────────
# Only render pipeline results when the last action was a real data query
# (not a definitional answer or irrelevant-query warning).
_show_results = (
    st.session_state.history
    and not st.session_state.get("definitional_query")
    and not st.session_state.get("irrelevant_query")
)

if _show_results:
    last_q, last_r = st.session_state.history[0]

    _lane = getattr(last_r.trace, "lane", "SQL")
    _lane_pill_cls = "lane-pill-sql" if _lane == "SQL" else "lane-pill-cypher"
    _lane_label = "Plain SQL" if _lane == "SQL" else "Cypher / Knowledge Graph"
    st.markdown(
        f"### Results for: *{last_q}* "
        f"<span class='lane-pill {_lane_pill_cls}'>{_lane_label}</span>",
        unsafe_allow_html=True,
    )

    # ── Swim-lane comparison diagram ────────────────────────────────────────
    _render_swimlanes(last_r.trace)

    # ── Status bar ────────────────────────────────────────────────────────────
    cols = st.columns(5)
    with cols[0]:
        status = "Success" if last_r.success else "Failed"
        st.markdown(f"<div class='metric-card'><b>Status</b><br>{status}</div>",
                    unsafe_allow_html=True)
    with cols[1]:
        rows = len(last_r.df) if last_r.df is not None else 0
        st.markdown(f"<div class='metric-card'><b>Rows</b><br>{rows:,}</div>",
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            f"<div class='metric-card'><b>Elapsed</b><br>{last_r.trace.elapsed_seconds}s</div>",
            unsafe_allow_html=True)
    with cols[3]:
        if _lane == "SQL":
            score = f"{last_r.trace.max_retrieval_score:.2%}"
            st.markdown(f"<div class='metric-card'><b>KG Score</b><br>{score}</div>",
                        unsafe_allow_html=True)
        else:
            attempts = getattr(last_r.trace, "cypher_repair_attempts", 0)
            st.markdown(f"<div class='metric-card'><b>Repairs</b><br>{attempts}</div>",
                        unsafe_allow_html=True)
    with cols[4]:
        if _lane == "SQL":
            attempts = getattr(last_r.trace, "sql_repair_attempts", 0)
            st.markdown(f"<div class='metric-card'><b>Repairs</b><br>{attempts}</div>",
                        unsafe_allow_html=True)

    # ── Stage-wise actual execution time (always visible, regardless of
    #    the "Show Query Trace" toggle — this is the headline demo metric) ──
    _render_stage_timings(last_r.trace)

    # ── Generated query (always visible, regardless of trace toggle) ──────────
    if _lane == "SQL" and last_r.trace.generated_sql:
        _sql_repairs = getattr(last_r.trace, "sql_repair_attempts", 0)
        _sql_label = "Generated SQL"
        if _sql_repairs:
            _sql_label += f"  (⚠️ self-healed after {_sql_repairs} repair attempt(s))"
        with st.expander(_sql_label, expanded=False):
            st.markdown(f"<div class='sql-box'>{html.escape(last_r.trace.generated_sql)}</div>",
                        unsafe_allow_html=True)
            if _sql_repairs:
                st.divider()
                _render_sql_repair_trace(
                    getattr(last_r.trace, "sql_repair_detail", []),
                    getattr(last_r.trace, "sql_repair_full_trace", ""),
                )
    elif _lane == "CYPHER" and last_r.trace.generated_cypher:
        with st.expander("Generated Cypher", expanded=False):
            st.markdown(f"<div class='cypher-box'>{last_r.trace.generated_cypher}</div>",
                        unsafe_allow_html=True)

    # ── Error message ─────────────────────────────────────────────────────────
    if not last_r.success:
        st.error(f"Query failed: {last_r.error}")

    # ── Results dataframe ─────────────────────────────────────────────────────
    if last_r.df is not None and not last_r.df.empty:
        st.dataframe(_format_results_df_for_display(last_r.df), use_container_width=True, height=400)

        csv = last_r.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv",
        )

        # ── AI Summary (below table + download button) ────────────────────────
        if ai_summary:
            clean_summary = _generate_table_summary(last_r.df, last_q, pipeline_summary=last_r.summary or "")
            if clean_summary:
                st.markdown(
                    f"<div style='background:#dbeafe; border:1px solid #93c5fd; border-radius:0.5rem; "
                    f"padding:1rem 1.2rem; font-size:1rem; font-family:inherit; color:#1e3a5f; margin-top:0.6rem;'>"
                    f"<b>AI Summary:</b> {clean_summary}</div>",
                    unsafe_allow_html=True,
                )
    elif last_r.success:
        st.warning("Query succeeded but returned no rows.")
        # Still show summary even when no rows returned
        if ai_summary and last_r.summary:
            clean_summary = re.sub(r'\*{1,3}', '', last_r.summary).strip()
            clean_summary = clean_summary.replace('`', '')
            st.markdown(
                f"<div style='background:#dbeafe; border:1px solid #93c5fd; border-radius:0.5rem; "
                f"padding:1rem 1.2rem; font-size:1rem; font-family:inherit; color:#1e3a5f; margin-top:0.6rem;'>"
                f"<b>AI Summary:</b> {clean_summary}</div>",
                unsafe_allow_html=True,
            )

    # ── Query Trace panel ─────────────────────────────────────────────────────
    if show_trace:
        _render_trace_panel(last_r, last_q)

    st.divider()

    # ── History ───────────────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("### Query History")
        for i, (hq, hr) in enumerate(st.session_state.history[1:], 1):
            icon = "+" if hr.success else "x"
            rows = len(hr.df) if hr.df is not None else 0
            lane_tag = "SQL" if getattr(hr.trace, "lane", "SQL") == "SQL" else "CYPHER"
            with st.expander(f"[{icon}] [{lane_tag}] {hq}  —  {rows} rows · {hr.trace.elapsed_seconds}s"):
                if hr.df is not None and not hr.df.empty:
                    st.dataframe(_format_results_df_for_display(hr.df), use_container_width=True, height=200)
                elif not hr.success:
                    st.error(hr.error)

else:
    # Welcome state — shown when no results yet
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #666;">
        <p>Type a question above about your P&amp;C insurance data to get started.</p>
        <p>Every question is automatically routed to one of two swim lanes:</p>
        <p><b>Plain SQL</b> &mdash; entity/attribute questions, filters, aggregations, reporting,
        and fixed-depth joins:<br>
        Schema Scoping &rarr; HyDE Expansion &rarr; KG Column Retrieval &rarr; Join Discovery
        &rarr; SQL Generation &rarr; SQLite Execution</p>
        <p><b>Cypher / Knowledge Graph</b> &mdash; relationship, multi-hop, and unknown-depth
        traversal questions:<br>
        Relationship Detection &rarr; Graph Schema Grounding &rarr; Cypher Generation
        &rarr; Neo4j Traversal &rarr; Result Binding</p>
    </div>
    """, unsafe_allow_html=True)