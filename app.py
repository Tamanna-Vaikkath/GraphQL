"""
app.py — Streamlit Chatbot UI for P&C Insurance NLQ System.

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
    page_title="P&C Insurance NLQ Chatbot",
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
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
    }
    .success-badge { color: #28a745; font-weight: bold; }
    .error-badge   { color: #dc3545; font-weight: bold; }
    /* Hide the native "Press Enter to apply" tooltip on text inputs */
    [data-testid="InputInstructions"] { display: none !important; }
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

if "irrelevant_query" not in st.session_state:
    st.session_state.irrelevant_query = False

if "definitional_query" not in st.session_state:
    st.session_state.definitional_query = None

if "definitional_answer" not in st.session_state:
    st.session_state.definitional_answer = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = None

# ── Logo path (relative to app.py) ───────────────────────────────────────────
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
    st.markdown("#### Sample Queries")
    _SAMPLE_QUERIES = [
        "show open claims",
        "show denied claims",
        "show payments for open claims",
        "show high reserve claims",
        "show claims from Texas",
        "show pending claims with payments",
    ]
    for _sq in _SAMPLE_QUERIES:
        if st.button(_sq, key=f"sq_{_sq}", use_container_width=True):
            st.session_state["pending_query"] = _sq
            st.session_state["run_sample"] = True
            st.rerun()

    st.divider()
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []

    if st.button("Clear Cache", use_container_width=True):
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_question", None)
        st.session_state.irrelevant_query = False
        st.session_state.definitional_query = None
        st.session_state.definitional_answer = None
        st.rerun()

    st.markdown("---")
    if st.button("KG Reasoning Explorer", use_container_width=True,
                 help="Visually explore why each column was retrieved, semantic scores, and join path discovery."):
        st.session_state["active_tab"] = "kg_explorer"
        st.rerun()

    if st.session_state.get("active_tab") == "kg_explorer":
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.pop("active_tab", None)
            st.rerun()

    st.markdown("---")
    st.caption("P&C Insurance NLQ  ·  Neo4j + Azure OpenAI")
    if st.session_state.get("config_ok"):
        st.caption("Connected")
    else:
        st.caption("Config error")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    {_logo_tag}
    <div class="main-header-text">
        <h2>P&amp;C Insurance NLQ Chatbot</h2>
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
    elif not _is_pc_relevant(q):
        st.session_state.irrelevant_query = True
        st.session_state.definitional_query = None
    else:
        st.session_state.irrelevant_query = False
        st.session_state.definitional_query = None
        with st.spinner("Running pipeline..."):
            pipeline: NLQPipeline = st.session_state.pipeline
            result = pipeline.query(q, generate_summary=ai_summary)
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

# ── Irrelevant query message ──────────────────────────────────────────────────
elif st.session_state.get("irrelevant_query"):
    st.warning(
        "Hello! I am a specialized P&C Insurance NLQ assistant designed to help you explore and analyze insurance-related data. "
        "I can assist with information related to Claims, Policies, Payments, Claimants, and other connected insurance insights. "
        "At the moment, I am not designed to answer general-purpose questions outside this domain. "
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
    "CLAIMANT": "#9b59b6",   # Purple  — must come before CLAIM to avoid substring match
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


# ── Main renderer — vis.js full-canvas traversal graph ────────────────────────
def _render_traversal_graph(
    sql: str,
    retrieved_columns: list[dict],
    join_conditions: list[str],
    question: str = "",
    height: int = 420,
    reranked_columns: list[dict] | None = None,
) -> None:
    """
    Render a live, physics-driven vis.js graph aligned with KG retrieval → LLM reranking.

    Table nodes are sized and coloured by their reranking importance:
      - Active/reranked tables get full colour + label + shadow
      - Column nodes inside each table are colour-coded by reranked score tier
      - Inactive tables are dimmed
    Reranked columns (from LLM reranking step) are highlighted separately from
    raw-retrieved columns, making the pipeline alignment user-visible.
    """
    import streamlit.components.v1 as components
    import json

    if not sql or not sql.strip():
        st.caption("No SQL available — graph cannot be rendered.")
        return

    # ── 1. Extract graph data ─────────────────────────────────────────────────
    real_tables, alias_map = _parse_sql_tables_and_aliases(sql)
    if not real_tables:
        st.caption("No table traversal detected in the generated SQL.")
        return

    sql_col_refs = _extract_sql_columns(sql, real_tables, alias_map)
    fk_edges     = _extract_fk_edges(sql, join_conditions, real_tables, alias_map)

    # Build (TABLE, COL) → best KG score from raw retrieved columns
    kg_scores: dict[tuple[str, str], float] = {}
    for col in (retrieved_columns or []):
        tbl = str(col.get("table", "")).upper()
        nm  = str(col.get("name", col.get("column_name", ""))).upper()
        sc  = float(col.get("score", 0))
        key = (tbl, nm)
        if key not in kg_scores or sc > kg_scores[key]:
            kg_scores[key] = sc

    # Build (TABLE, COL) → reranked score — used for highlighting reranking tier
    reranked_scores: dict[tuple[str, str], float] = {}
    reranked_rank:   dict[tuple[str, str], int]   = {}
    if reranked_columns:
        for rank_i, col in enumerate(reranked_columns):
            tbl = str(col.get("table", "")).upper()
            nm  = str(col.get("name", col.get("column_name", ""))).upper()
            sc  = float(col.get("score", 0))
            key = (tbl, nm)
            reranked_scores[key] = sc
            reranked_rank[key]   = rank_i + 1
    else:
        reranked_scores = kg_scores
        reranked_rank   = {k: i+1 for i, k in enumerate(
            sorted(kg_scores, key=lambda x: kg_scores[x], reverse=True)
        )}

    # Build active tables: any table appearing in reranked top-15 OR in SQL
    reranked_tables: set[str] = {t for (t, _) in list(reranked_scores.keys())[:15]}

    # Columns actually in the SQL, with scores (SELECT * fallback included)
    graph_cols: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    if sql_col_refs:
        for (tbl, col) in sorted(sql_col_refs):
            if (tbl, col) not in seen:
                seen.add((tbl, col))
                graph_cols.append((tbl, col, kg_scores.get((tbl, col), 0.0)))
    # Also add reranked columns not already in graph_cols (for full picture)
    for (tbl, col), sc in reranked_scores.items():
        if tbl in real_tables and (tbl, col) not in seen:
            seen.add((tbl, col))
            graph_cols.append((tbl, col, sc))
    # Fallback: KG-scored columns
    if not graph_cols:
        for (tbl, col), sc in kg_scores.items():
            if tbl in real_tables and (tbl, col) not in seen:
                seen.add((tbl, col))
                graph_cols.append((tbl, col, sc))

    # ── 2. Build vis.js data arrays ──────────────────────────────────────────
    # Per-table: column list for tooltip, FK partners for tooltip, node size
    tbl_cols:    dict[str, list[tuple[str, float, float]]] = {t: [] for t in real_tables}
    tbl_fk_desc: dict[str, list[str]]                      = {t: [] for t in real_tables}

    for (tbl, col, sc) in graph_cols:
        if tbl in tbl_cols:
            rsc = reranked_scores.get((tbl, col), 0.0)
            tbl_cols[tbl].append((col, sc, rsc))

    # Sort columns: by reranked score desc, then kg score
    for tbl in tbl_cols:
        tbl_cols[tbl].sort(key=lambda x: (x[2], x[1]), reverse=True)

    for (src, dst, pred) in fk_edges:
        tbl_fk_desc[src].append(pred)
        tbl_fk_desc[dst].append(pred)

    # ── White-background color palette ───────────────────────────────────────
    _WB_COLORS: dict[str, str] = {
        "CLAIMANT": "#8b5cf6",
        "CLAIM":    "#ef4444",
        "POLICY":   "#3b82f6",
        "PAYMENT":  "#10b981",
        "ADJUSTER": "#f97316",
    }

    def _wb_col(t: str) -> str:
        u = t.upper()
        for k, v in _WB_COLORS.items():
            if u == k or u.startswith(k + "_"): return v
        for k, v in _WB_COLORS.items():
            if k in u: return v
        return "#64748b"

    # Build vis nodes — only tables that appear in the generated SQL
    vis_nodes = []
    for tbl in sorted(real_tables):
        color = _wb_col(tbl)
        cols_for_tbl = tbl_cols[tbl]

        # Tooltip: only list the retrieved column names for this table — no scores, no extras
        retrieved_col_names = [col for (col, sc, rsc) in cols_for_tbl if sc > 0 or (tbl, col) in sql_col_refs]
        # Also include any SQL-referenced columns not already in the list
        for (t, c) in sql_col_refs:
            if t == tbl and c not in retrieved_col_names:
                retrieved_col_names.append(c)

        col_rows = ""
        for col in retrieved_col_names[:20]:
            col_rows += (
                f"<div style='font-family:monospace;font-size:11px;"
                f"color:#1e293b;padding:2px 0;white-space:nowrap'>"
                f"· {col}</div>"
            )
        if not col_rows:
            col_rows = "<div style='color:#94a3b8;font-size:10px'>No retrieved columns</div>"

        tooltip = (
            f"<div style='font-family:Inter,system-ui,sans-serif;"
            f"background:#ffffff;border:1px solid #e2e8f0;"
            f"border-radius:8px;padding:10px 12px;min-width:200px;"
            f"box-shadow:0 4px 20px rgba(0,0,0,.12)'>"
            f"<div style='font-size:13px;font-weight:700;color:{color};"
            f"margin-bottom:8px;border-bottom:1px solid #f1f5f9;padding-bottom:5px'>"
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{color};margin-right:6px'></span>"
            f"{tbl}</div>"
            f"{col_rows}"
            f"</div>"
        )

        is_reranked = tbl in reranked_tables
        n_reranked_here = sum(1 for (c, _, rsc) in cols_for_tbl if rsc >= 0.65)
        radius = max(30, min(52, 30 + n_reranked_here * 4)) if is_reranked else 28

        node_cfg = {
            "id":    tbl,
            "label": tbl,
            "title": tooltip,
            "shape": "circle",
            "size":  radius,
            "shadow": {"enabled": is_reranked, "color": color + "44",
                       "size": 10, "x": 0, "y": 3},
            "borderWidth": 2 if is_reranked else 1,
        }
        if is_reranked:
            node_cfg["color"] = {
                "background": color,
                "border":     color,
                "highlight":  {"background": color, "border": "#1e293b"},
                "hover":      {"background": color, "border": "#1e293b"},
            }
            node_cfg["font"] = {"size": 13, "bold": True, "color": "#ffffff",
                                "face": "Inter,system-ui,sans-serif", "strokeWidth": 0}
            node_cfg["size"] = radius
        else:
            node_cfg["color"] = {
                "background": "#f8fafc",
                "border":     "#cbd5e1",
                "highlight":  {"background": color + "22", "border": color},
                "hover":      {"background": color + "22", "border": color},
            }
            node_cfg["font"] = {"size": 10, "bold": False, "color": "#94a3b8",
                                "face": "Inter,system-ui,sans-serif", "strokeWidth": 0}
            node_cfg["size"] = 28
        vis_nodes.append(node_cfg)

    # Build vis edges — FK joins with white-bg colours
    vis_edges = []
    for idx, (src, dst, pred) in enumerate(fk_edges):
        src_is_reranked = src in reranked_tables
        edge_color = _wb_col(src) if src_is_reranked else "#94a3b8"
        vis_edges.append({
            "id":     f"e{idx}",
            "from":   src,
            "to":     dst,
            "label":  pred.split("=")[0].strip().split(".")[-1] if "=" in pred else pred,
            "title":  (
                f"<div style='font-family:Inter;background:#ffffff;"
                f"border:1px solid #e2e8f0;border-radius:6px;"
                f"box-shadow:0 4px 12px rgba(0,0,0,0.1);"
                f"padding:8px 12px;font-size:11px'>"
                f"<b style='color:#1e40af'>FK Join</b><br>"
                f"<span style='font-family:monospace;color:#374151'>{pred}</span></div>"
            ),
            "color":  {"color": edge_color, "highlight": _wb_col(src),
                       "hover": _wb_col(src), "opacity": 1.0},
            "width":  2.5 if src_is_reranked else 1,
            "font":   {"size": 10, "color": edge_color, "face": "monospace",
                       "bold": True, "strokeWidth": 3, "strokeColor": "#ffffff",
                       "align": "middle", "background": "#ffffff"},
            "smooth": {"type": "curvedCW", "roundness": 0.25},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7, "type": "arrow"}},
            "dashes": not src_is_reranked,
        })

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    # ── 3. Stats for the footer bar ──────────────────────────────────────────
    n_tables  = len(real_tables)
    n_joins   = len(fk_edges)
    n_cols    = len(graph_cols)

    # Legend pills — only tables actually present in this query
    legend_html = ""
    label_map = {"CLAIMANT": "Claimant", "CLAIM": "Claims", "POLICY": "Policy",
                 "PAYMENT": "Payment", "ADJUSTER": "Adjuster"}
    for tbl in sorted(real_tables):
        c   = _wb_col(tbl)
        is_rt = tbl in reranked_tables
        # Match longest key first to avoid "CLAIM" swallowing "CLAIMANT"
        lbl = next(
            (v for k, v in label_map.items() if tbl.upper() == k or tbl.upper().startswith(k + "_")),
            tbl.title()
        )
        n_rk = sum(1 for (col, sc, rsc) in tbl_cols.get(tbl, []) if sc >= 0.65)
        legend_html += (
            f"<span style='display:inline-flex;align-items:center;gap:5px;"
            f"margin-right:10px;opacity:{'1' if is_rt else '0.45'}'>"
            f"<span style='width:10px;height:10px;border-radius:50%;"
            f"background:{c};display:inline-block;border:1.5px solid {c}'></span>"
            f"<span style='color:#374151;font-weight:{'600' if is_rt else '400'}'>{lbl}"
            f"<span style='color:#94a3b8;font-size:10px;margin-left:3px'>({n_rk})</span></span></span>"
        )
    # Add join info to footer
    n_reranked_total = len(reranked_scores)

    # ── 4. HTML template ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet"/>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ width:100%; height:100%; background:#ffffff; overflow:hidden; }}
  #graph {{ width:100%; height:{height}px; background:#ffffff; }}

  /* vis.js tooltip override */
  div.vis-tooltip {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    font-size: 12px;
  }}

  /* ── Footer bar ── */
  #footer {{
    position:absolute; bottom:0; left:0; right:0; z-index:10;
    background:rgba(255,255,255,0.95);
    border-top:1px solid #e2e8f0;
    padding:5px 12px;
    display:flex; align-items:center; justify-content:space-between;
    font-size:10.5px; font-family:Inter,system-ui,sans-serif;
  }}
  #footer .stats {{ color:#475569; }}
  #footer .stats b {{ color:#1e293b; }}

  /* ── Stabilised badge ── */
  #badge {{
    position:absolute; top:10px; right:12px; z-index:10;
    background:#f0fdf4; border:1px solid #86efac;
    border-radius:20px; padding:3px 10px;
    font-size:9.5px; color:#166534;
    font-family:Inter,system-ui,sans-serif;
    display:none;
  }}
</style>
</head>
<body>

<div id="graph"></div>
<div id="badge">● Stabilised — drag to rearrange</div>

<div id="footer">
  <span class="stats">
    <b>{n_tables}</b> table(s) &nbsp;·&nbsp;
    <b>{n_joins}</b> FK join(s) &nbsp;·&nbsp;
    <b>{n_cols}</b> retrieved
  </span>
  <span>{legend_html}</span>
</div>

<script>
(function() {{
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});

  var container = document.getElementById('graph');
  var options = {{
    physics: {{
      enabled: true,
      solver: 'barnesHut',
      barnesHut: {{
        gravitationalConstant: -22000,
        centralGravity: 0.35,
        springLength: 220,
        springConstant: 0.04,
        damping: 0.28,
        avoidOverlap: 0.6
      }},
      stabilization: {{
        enabled: true,
        iterations: 350,
        updateInterval: 20,
        fit: true
      }}
    }},
    interaction: {{
      hover: true,
      tooltipDelay: 80,
      zoomView: true,
      dragNodes: true,
      dragView: true,
      navigationButtons: false,
      keyboard: false
    }},
    nodes: {{
      borderWidth: 2.5,
      borderWidthSelected: 4,
      chosen: {{
        node: function(values) {{
          values.shadowSize = 20;
          values.borderWidth = 4;
        }}
      }}
    }},
    edges: {{
      width: 2.5,
      selectionWidth: 4,
      hoverWidth: 1,
      smooth: {{ type: 'curvedCW', roundness: 0.25 }},
      font: {{ size: 11, strokeWidth: 3, strokeColor: '#ffffff',
               align: 'middle', face: 'monospace' }}
    }}
  }};

  var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);

  /* Freeze physics after stabilisation so the graph stays put */
  network.once('stabilized', function() {{
    network.setOptions({{ physics: {{ enabled: false }} }});
    document.getElementById('badge').style.display = 'block';
    network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
  }});

  /* Re-enable physics while dragging a node, freeze again on release */
  network.on('dragStart', function(params) {{
    if (params.nodes.length > 0) {{
      network.setOptions({{ physics: {{ enabled: true }} }});
    }}
  }});
  network.on('dragEnd', function(params) {{
    if (params.nodes.length > 0) {{
      setTimeout(function() {{
        network.setOptions({{ physics: {{ enabled: false }} }});
      }}, 800);
    }}
  }});
}})();
</script>
</body>
</html>"""

    components.html(html, height=height + 6, scrolling=False)


# ── KG Reasoning Explorer ─────────────────────────────────────────────────────
def _render_kg_reasoning_explorer(question: str, trace) -> None:
    """
    KG Reasoning Explorer — two sections:
      1. Full Schema KG Graph  — all 4 table nodes + all their column nodes with
         query-specific highlighting (retrieved/active columns glow; inactive dim).
      2. KG Retrieval → LLM Reranking Flow — pipeline timeline card view.
    """
    import streamlit.components.v1 as components
    import json

    st.markdown("## KG Reasoning Explorer")
    st.markdown(
        "<p style='color:#475569;font-size:0.95rem;margin-top:-0.4rem'>"
        "Visual lifecycle of how the Knowledge Graph retrieved and scored columns "
        "for your query, from raw Neo4j vector search through LLM reranking to "
        "final SQL generation.</p>",
        unsafe_allow_html=True,
    )

    if not trace:
        st.info("Run a query first — the KG Reasoning Explorer will populate once "
                "pipeline trace data is available.")
        return

    retrieved_columns: list[dict] = trace.retrieved_columns or []
    join_conditions:   list[str]  = trace.join_conditions   or []
    generated_sql:     str        = trace.generated_sql     or ""

    reranked = _rerank_columns(retrieved_columns, question=question)

    # ── Section 1: Full Schema KG Graph ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. Live Knowledge Graph — Query Highlights")
    st.markdown(
        "<p style='color:#475569;font-size:0.88rem'>"
        "<b style='color:#1d4ed8'>Blue circles</b> = tables involved in this query. "
        "<b style='color:#d97706'>Yellow circles</b> = columns retrieved from those tables. "
        "Hover any node for details. FK join edges connect related tables.</p>",
        unsafe_allow_html=True,
    )

    import streamlit.components.v1 as components
    import json

    # ── Build data from trace ─────────────────────────────────────────────────
    retrieved_score: dict[tuple[str, str], float] = {}
    for col in retrieved_columns:
        tbl = str(col.get("table", "")).upper()
        nm  = str(col.get("name", col.get("column_name", ""))).upper()
        sc  = float(col.get("score", 0))
        key = (tbl, nm)
        if key not in retrieved_score or sc > retrieved_score[key]:
            retrieved_score[key] = sc

    real_tables_sql, alias_map_sql = _parse_sql_tables_and_aliases(generated_sql)
    sql_col_refs = _extract_sql_columns(generated_sql, real_tables_sql, alias_map_sql)

    # ── Only tables and columns from the SQL ─────────────────────────────────
    # Tables: those that appear in FROM / JOIN in the SQL
    sql_tables: set[str] = set(real_tables_sql)
    # Columns: only (table, col) pairs directly referenced in SQL
    sql_columns: set[tuple[str, str]] = {(t, c) for (t, c) in sql_col_refs if t in sql_tables}

    if not sql_tables:
        st.caption("No tables found in generated SQL — graph cannot be rendered.")
    else:
        # ── Build vis.js nodes & edges ────────────────────────────────────────
        vis_nodes = []
        vis_edges = []
        node_id   = 0
        table_node_ids: dict[str, int] = {}

        # TABLE nodes — big blue circles, black text
        TABLE_BG     = "#1d4ed8"   # rich blue
        TABLE_BORDER = "#1e3a8a"
        TABLE_FONT   = "#000000"   # black
        TABLE_SIZE   = 55          # large enough for label inside

        # COLUMN nodes — big yellow circles, black text
        COL_BG     = "#fbbf24"    # amber-yellow
        COL_BORDER = "#d97706"
        COL_FONT   = "#000000"

        def _make_table_tooltip(tbl: str) -> str:
            cols_here = sorted(c for (t, c) in sql_columns if t == tbl)
            col_rows = "".join(
                f"<div style='font-family:monospace;font-size:10px;color:#1d4ed8;padding:1px 0'>"
                f"· {c}"
                f"</div>"
                for c in cols_here
            ) or "<div style='color:#94a3b8;font-size:10px'>No columns retrieved</div>"
            return (
                f"<div style='font-family:Inter;background:#fff;border:1px solid #e2e8f0;"
                f"border-radius:8px;padding:10px 14px;min-width:210px;"
                f"box-shadow:0 4px 16px rgba(0,0,0,0.12);font-size:12px'>"
                f"<div style='font-weight:700;color:#1d4ed8;font-size:13px;"
                f"margin-bottom:6px;border-bottom:1px solid #e2e8f0;padding-bottom:5px'>"
                f"{tbl}</div>"
                f"<div style='color:#64748b;font-size:10px;margin-bottom:5px'>"
                f"{len(cols_here)} column(s)</div>"
                f"{col_rows}</div>"
            )

        for tbl in sorted(sql_tables):
            tbl_nid = node_id
            table_node_ids[tbl] = tbl_nid
            vis_nodes.append({
                "id":    tbl_nid,
                "label": tbl,
                "group": "table",
                "table": tbl,
                "shape": "circle",
                "size":  TABLE_SIZE,
                "color": {
                    "background": TABLE_BG,
                    "border":     TABLE_BORDER,
                    "highlight":  {"background": "#2563eb", "border": "#1e3a8a"},
                    "hover":      {"background": "#2563eb", "border": "#1e3a8a"},
                },
                "font": {
                    "size":        13,
                    "bold":        True,
                    "color":       TABLE_FONT,
                    "face":        "Inter,system-ui,sans-serif",
                    "strokeWidth": 0,
                    "vadjust":     0,
                },
                "shadow":      {"enabled": True, "color": "#1d4ed844",
                                "size": 14, "x": 0, "y": 4},
                "borderWidth": 3,
                "title":       _make_table_tooltip(tbl),
                "level":       0,
            })
            node_id += 1

        # COLUMN nodes — one per (table, col) in SQL
        for (tbl, col) in sorted(sql_columns):
            if tbl not in table_node_ids:
                continue
            tbl_nid = table_node_ids[tbl]
            sc      = retrieved_score.get((tbl, col), 0.0)
            # Shorten label so it fits inside circle (max 10 chars)
            label   = col if len(col) <= 11 else col[:10] + "…"

            vis_nodes.append({
                "id":    node_id,
                "label": label,
                "group": "column",
                "table": tbl,
                "column": col,
                "shape": "circle",
                "size":  36,
                "color": {
                    "background": COL_BG,
                    "border":     COL_BORDER,
                    "highlight":  {"background": "#fcd34d", "border": COL_BORDER},
                    "hover":      {"background": "#fcd34d", "border": COL_BORDER},
                },
                "font": {
                    "size":        9,
                    "bold":        True,
                    "color":       COL_FONT,
                    "face":        "monospace",
                    "strokeWidth": 0,
                    "vadjust":     0,
                },
                "shadow":      {"enabled": True, "color": "#fbbf2444",
                                "size": 8, "x": 0, "y": 2},
                "borderWidth": 2,
                "title": (
                    f"<div style='font-family:Inter;background:#fff;"
                    f"border:1px solid #e2e8f0;border-radius:6px;"
                    f"box-shadow:0 4px 12px rgba(0,0,0,0.1);"
                    f"padding:8px 12px;font-size:11px'>"
                    f"<span style='color:#1d4ed8;font-weight:700'>{tbl}</span>"
                    f"<span style='color:#94a3b8'>.</span>"
                    f"<span style='color:#d97706;font-weight:700'>{col}</span>"
                    f"</div>"
                ),
                "level": 1,
            })
            # Edge: table → column
            vis_edges.append({
                "id":    f"tc_{tbl_nid}_{node_id}",
                "from":  tbl_nid,
                "to":    node_id,
                "color": {"color": "#d97706cc", "highlight": "#d97706",
                          "hover": "#d97706"},
                "width":  1.8,
                "dashes": False,
                "arrows": {"to": {"enabled": False}},
                "smooth": {"type": "dynamic"},
            })
            node_id += 1

        # ── FK join edges — multi-strategy detection ─────────────────────────
        # Strategy 1: parse ON clauses and Neo4j join_conditions
        fk_edges_used = _extract_fk_edges(generated_sql, join_conditions,
                                          real_tables_sql, alias_map_sql)

        # Strategy 2: scan the entire SQL for TABLE_A.COL = TABLE_B.COL patterns
        # (catches WHERE-style joins and any case _extract_fk_edges missed)
        _jc_re2 = re.compile(
            r'\b([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)'
            r'\s*=\s*'
            r'([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)\b'
        )
        _seen2 = {frozenset({s, d}) for s, d, _ in fk_edges_used}
        for _m in _jc_re2.finditer(generated_sql.upper()):
            t1 = alias_map_sql.get(_m.group(1), _m.group(1))
            t2 = alias_map_sql.get(_m.group(3), _m.group(3))
            if t1 in real_tables_sql and t2 in real_tables_sql and t1 != t2:
                _k = frozenset({t1, t2})
                if _k not in _seen2:
                    _seen2.add(_k)
                    fk_edges_used.append((t1, t2,
                        f"{_m.group(1)}.{_m.group(2)} = {_m.group(3)}.{_m.group(4)}"))

        # Strategy 3: fuzzy schema-FK fallback — match actual table names by prefix
        # Handles "CLAIMS" matching "CLAIM", "CLAIMANT" staying "CLAIMANT", etc.
        def _fuzzy_match(candidate: str, table_set: set[str]) -> str | None:
            """Return the real table name in table_set that best matches candidate."""
            c = candidate.upper()
            if c in table_set:
                return c
            # Exact prefix match (e.g. "CLAIM" matches "CLAIMS")
            for t in table_set:
                if t.startswith(c) or c.startswith(t):
                    return t
            return None

        # Canonical FK relationships — table names are abstract prefixes here
        _SCHEMA_FK = [
            ("CLAIM",    "CLAIMANT", "CLAIMANT_ID"),
            ("CLAIM",    "POLICY",   "POLICY_ID"),
            ("PAYMENT",  "CLAIM",    "CLAIM_ID"),
            ("CLAIM",    "ADJUSTER", "ADJUSTER_ID"),
        ]
        _active_pairs = {frozenset({s, d}) for s, d, _ in fk_edges_used}
        for _src_pfx, _dst_pfx, _key_col in _SCHEMA_FK:
            _src_real = _fuzzy_match(_src_pfx, sql_tables)
            _dst_real = _fuzzy_match(_dst_pfx, sql_tables)
            if not _src_real or not _dst_real or _src_real == _dst_real:
                continue
            _pair = frozenset({_src_real, _dst_real})
            if _pair in _active_pairs:
                continue
            # Only add if both tables are in the query
            _active_pairs.add(_pair)
            fk_edges_used.append((
                _src_real, _dst_real,
                f"{_src_real}.{_key_col} = {_dst_real}.{_key_col}"
            ))

        # Build animated join edges between table nodes
        _JOIN_COLORS = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899"]
        for _idx, (_src, _dst, _pred) in enumerate(fk_edges_used):
            if _src not in table_node_ids or _dst not in table_node_ids:
                continue
            _join_col = _pred.split("=")[0].strip().split(".")[-1] if "=" in _pred else _pred
            _edge_color = _JOIN_COLORS[_idx % len(_JOIN_COLORS)]
            vis_edges.append({
                "id":    f"fk_{_src}_{_dst}",
                "from":  table_node_ids[_src],
                "to":    table_node_ids[_dst],
                "label": _join_col,
                "title": (
                    f"<div style='font-family:Inter;background:#fff;"
                    f"border:1px solid #e2e8f0;border-radius:8px;"
                    f"box-shadow:0 4px 16px rgba(0,0,0,0.12);"
                    f"padding:10px 14px;font-size:11px;min-width:180px'>"
                    f"<div style='font-weight:700;color:{_edge_color};"
                    f"font-size:12px;margin-bottom:6px;display:flex;"
                    f"align-items:center;gap:6px'>"
                    f"<span style='display:inline-block;width:8px;height:8px;"
                    f"border-radius:50%;background:{_edge_color}'></span>"
                    f"Table Join</div>"
                    f"<div style='font-family:monospace;color:#1e293b;"
                    f"font-size:11px;background:#f8fafc;padding:4px 8px;"
                    f"border-radius:4px;border:1px solid #e2e8f0'>"
                    f"{_src} ↔ {_dst}</div>"
                    f"<div style='color:#64748b;font-size:10px;margin-top:6px'>"
                    f"Key: <span style='font-family:monospace;color:#0f766e'>{_join_col}</span>"
                    f"</div></div>"
                ),
                "color": {"color": _edge_color, "highlight": _edge_color,
                          "hover": _edge_color, "opacity": 1.0},
                "width":  4,
                "dashes": False,
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.9, "type": "arrow"},
                           "from": {"enabled": True, "scaleFactor": 0.9, "type": "arrow"}},
                "font": {
                    "size": 11, "color": "#1e293b", "bold": True,
                    "face": "monospace", "strokeWidth": 3,
                    "strokeColor": "#ffffff", "align": "middle",
                    "background": "#ffffff",
                },
                "shadow": {"enabled": True, "color": _edge_color + "55",
                           "size": 8, "x": 0, "y": 0},
                "smooth": {"type": "curvedCW", "roundness": 0.18},
            })

        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)

        n_tbl  = len(sql_tables)
        n_cols = len(sql_columns)
        n_fks  = len(fk_edges_used)

        # Legend
        legend_html = ""
        for tbl in sorted(sql_tables):
            cols_count = sum(1 for (t, _) in sql_columns if t == tbl)
            legend_html += (
                f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:12px'>"
                f"<span style='width:10px;height:10px;border-radius:50%;"
                f"background:#1d4ed8;display:inline-block'></span>"
                f"<span style='color:#1e293b;font-weight:600;font-size:11px'>{tbl}</span>"
                f"<span style='color:#94a3b8;font-size:10px'>({cols_count} cols)</span></span>"
            )

        kg_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet"/>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ width:100%; height:100%; background:#f8fafc; overflow:hidden;
    font-family:Inter,-apple-system,BlinkMacSystemFont,system-ui,sans-serif; }}
  #kg {{ width:100%; height:680px; background:#f8fafc; cursor:pointer; }}
  div.vis-tooltip {{
    background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 !important;
  }}
  #footer {{
    position:absolute; bottom:0; left:0; right:0; z-index:10;
    background:rgba(255,255,255,0.97); border-top:1px solid #e2e8f0;
    padding:7px 14px; display:flex; align-items:center;
    justify-content:space-between;
    font-family:Inter,system-ui,sans-serif; font-size:10.5px;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.04);
  }}
  #badge {{
    display:none; position:absolute; top:10px; right:12px; z-index:10;
    background:#f0fdf4; border:1px solid #86efac; color:#166534;
    border-radius:20px; padding:4px 12px;
    font-size:9.5px; font-family:Inter,system-ui,sans-serif;
    box-shadow:0 1px 4px rgba(0,0,0,0.08);
  }}
  #click-hint {{
    position:absolute; top:10px; left:50%; transform:translateX(-50%); z-index:10;
    background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8;
    border-radius:20px; padding:5px 14px;
    font-size:10px; font-family:Inter,system-ui,sans-serif;
    box-shadow:0 1px 6px rgba(29,78,216,0.12);
    pointer-events:none;
    animation: pulse-hint 2s ease-in-out infinite;
  }}
  @keyframes pulse-hint {{
    0%, 100% {{ opacity:1; transform:translateX(-50%) scale(1); }}
    50% {{ opacity:0.7; transform:translateX(-50%) scale(1.03); }}
  }}
  .leg-dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
</style>
</head>
<body>
<div id="kg"></div>
<div id="badge">&#9679; Stabilised — drag to rearrange</div>
<div id="footer">
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    {legend_html}
  </div>
  <div style="display:flex;align-items:center;gap:10px;font-size:10px;color:#475569">
    <span style="display:inline-flex;align-items:center;gap:4px">
      <span class="leg-dot" style="background:#1d4ed8"></span>
      <span>Table node</span></span>
    <span style="display:inline-flex;align-items:center;gap:4px">
      <span class="leg-dot" style="background:#fbbf24"></span>
      <span>Column</span></span>
    <span style="display:inline-flex;align-items:center;gap:4px">
      <span style="display:inline-block;width:22px;height:3px;background:#6366f1;border-radius:2px;margin-top:3px"></span>
      <span>Table join</span></span>
    <span style="display:inline-flex;align-items:center;gap:4px;padding-left:8px;border-left:1px solid #e2e8f0">
    <span style="color:#64748b;padding-left:8px;border-left:1px solid #e2e8f0">
      <b style="color:#1e293b">{n_tbl}</b> table(s) &nbsp;&#183;&nbsp;
      <b style="color:#1e293b">{n_cols}</b> column(s) &nbsp;&#183;&nbsp;
      <b style="color:#1e293b">{n_fks}</b> FK join(s)</span>
  </div>
</div>
<script>
(function() {{
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var container = document.getElementById('kg');
  var options = {{
    physics: {{
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{
        gravitationalConstant: -160,
        centralGravity: 0.006,
        springLength: 200,
        springConstant: 0.04,
        damping: 0.45,
        avoidOverlap: 1.5
      }},
      stabilization: {{ enabled:true, iterations:800, updateInterval:20, fit:true }}
    }},
    interaction: {{
      hover:true, tooltipDelay:60, zoomView:true,
      dragNodes:true, dragView:true,
      navigationButtons:false, keyboard:false
    }},
    nodes: {{
      borderWidth:2.5, borderWidthSelected:4,
      chosen: {{
        node: function(values) {{ values.shadowSize = 22; values.borderWidth = 4; }}
      }}
    }},
    edges: {{
      selectionWidth:3, hoverWidth:1.2,
      font: {{ background:"#ffffff", strokeWidth:0, size:10 }}
    }}
  }};
  var network = new vis.Network(container, {{ nodes:nodes, edges:edges }}, options);

  // ── Collect FK edge ids and original states once graph is ready ──
  var joinEdgeIds = [];
  var origStates  = {{}};
  var animRunning = false;

  function _runFkAnimation() {{
    if (animRunning || joinEdgeIds.length === 0) return;
    animRunning = true;

    // Phase 1: flash all FK edges together (0 – 400 ms)
    joinEdgeIds.forEach(function(eid) {{
      edges.update({{ id: eid, width: 7,
        shadow: {{ enabled:true, color:"#fbbf24aa", size:20, x:0, y:0 }} }});
    }});

    // Phase 2 (600 ms+): highlight each edge one by one
    setTimeout(function() {{
      // Restore all to original first
      joinEdgeIds.forEach(function(eid) {{
        edges.update({{ id: eid,
          width:  origStates[eid].width  || 4,
          shadow: origStates[eid].shadow || {{ enabled:true }} }});
      }});

      var step = 0;
      var stepsPerEdge = 4;  // highlight → mid-fade → restore → pause
      var interval = setInterval(function() {{
        var edgeIdx = Math.floor(step / stepsPerEdge);
        var pulse   = step % stepsPerEdge;
        if (edgeIdx >= joinEdgeIds.length) {{
          clearInterval(interval);
          animRunning = false;
          return;
        }}
        var eid = joinEdgeIds[edgeIdx];
        if (pulse === 0) {{
          edges.update({{ id: eid, width: 9,
            shadow: {{ enabled:true, color:"#fbbf24cc", size:24, x:0, y:0 }} }});
        }} else if (pulse === 1) {{
          edges.update({{ id: eid, width: 6,
            shadow: {{ enabled:true, color:"#fbbf2477", size:14, x:0, y:0 }} }});
        }} else if (pulse === 2) {{
          var orig = origStates[eid];
          edges.update({{ id: eid,
            width:  orig.width  || 4,
            shadow: orig.shadow || {{ enabled:true }} }});
        }}
        step++;
      }}, 280);
    }}, 600);
  }}

  network.once("stabilized", function() {{
    network.setOptions({{ physics:{{ enabled:false }} }});
    document.getElementById("badge").style.display = "block";
    network.fit({{ animation:{{ duration:700, easingFunction:"easeInOutQuad" }} }});

    // Capture FK edge ids and their original visual states
    joinEdgeIds = edges.get().filter(function(e) {{
      return e.id && e.id.toString().startsWith("fk_");
    }}).map(function(e) {{ return e.id; }});

    joinEdgeIds.forEach(function(eid) {{
      var e = edges.get(eid);
      origStates[eid] = {{ color: e.color, width: e.width, shadow: e.shadow }};
    }});
  }});

  // ── Trigger animation on any canvas click (background click = no node/edge) ──
  network.on("click", function(params) {{
    // Fire on background click OR node/edge click — always impressive
    _runFkAnimation();
  }});

  network.on("dragStart", function(p) {{
    if (p.nodes.length > 0) network.setOptions({{ physics:{{ enabled:true }} }});
  }});
  network.on("dragEnd", function(p) {{
    if (p.nodes.length > 0)
      setTimeout(function() {{
        network.setOptions({{ physics:{{ enabled:false }} }});
      }}, 900);
  }});
}})();
</script>
</body>
</html>"""

        components.html(kg_html, height=706, scrolling=False)

    # ── Section 2: KG Traversal + LLM Reranking Explanation ─────────────────
    st.markdown("---")
    st.markdown("### 2. How the Graph Was Traversed & Columns Reranked")
    st.markdown(
        "<p style='color:#475569;font-size:0.88rem'>"
        "A walkthrough of every step from your question to the final SQL — "
        "what Neo4j retrieved, how the LLM reranker scored and adjusted columns, and why "
        "specific columns ended up in the query.</p>",
        unsafe_allow_html=True,
    )

    raw_cols_sorted  = sorted(retrieved_columns,
                              key=lambda c: float(c.get("score", 0)), reverse=True)
    reranked_cols    = reranked

    real_tables_sql2, alias_map_sql2 = _parse_sql_tables_and_aliases(generated_sql)
    sql_refs2 = _extract_sql_columns(generated_sql, real_tables_sql2, alias_map_sql2)

    raw_names = [
        f"{str(c.get('table','')).upper()}.{str(c.get('name', c.get('column_name',''))).upper()}"
        for c in raw_cols_sorted
    ]
    reranked_names = [
        f"{str(c.get('table','')).upper()}.{str(c.get('name', c.get('column_name',''))).upper()}"
        for c in reranked_cols
    ]

    def _rank_delta(name: str) -> int:
        r = raw_names.index(name) if name in raw_names else 999
        n = reranked_names.index(name) if name in reranked_names else 999
        return r - n

    # ── Step 1: Neo4j KG Traversal ───────────────────────────────────────────
    expl_html = (
        "<div style='font-family:Inter,system-ui,sans-serif;display:flex;"
        "flex-direction:column;gap:14px'>"
    )

    # Step 1
    top_hits = raw_cols_sorted[:5]
    top_hits_str = "".join(
        f"<span style='background:#ede9fe;border:1px solid #c4b5fd;"
        f"color:#5b21b6;font-family:monospace;font-size:0.72rem;font-weight:700;"
        f"padding:2px 7px;border-radius:5px;display:inline-block;margin:2px'>"
        f"{str(c.get('table','')).upper()}.{str(c.get('name',c.get('column_name',''))).upper()}</span>"
        for c in top_hits
    )
    more_str = (
        f"<span style='color:#94a3b8;font-size:0.72rem'>+{len(raw_cols_sorted)-5} more retrieved…</span>"
        if len(raw_cols_sorted) > 5 else ""
    )
    expl_html += (
        f"<div style='background:#faf5ff;border:1px solid #e9d5ff;"
        f"border-radius:10px;padding:14px 16px'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
        f"<div style='min-width:28px;height:28px;border-radius:50%;background:#7c3aed;"
        f"color:#fff;font-weight:800;font-size:0.85rem;display:flex;"
        f"align-items:center;justify-content:center'>1</div>"
        f"<div style='font-size:0.9rem;font-weight:700;color:#1e293b'>"
        f"Neo4j Knowledge Graph — Vector Similarity Traversal</div></div>"
        f"<p style='font-size:0.82rem;color:#374151;margin-bottom:8px'>"
        f"The user query <i>&quot;{question}&quot;</i> was embedded into a vector and used to "
        f"traverse the Neo4j Knowledge Graph. The KG stores every column in the schema "
        f"as an embedded node; the search walks the graph edges (table→column, column→column) "
        f"and retrieves the closest matching columns by cosine similarity. Top hits:</p>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px'>{top_hits_str}{more_str}</div>"
        f"</div>"
    )

    # Step 2: LLM Reranking explanation
    boosted  = [(c, _rank_delta(f"{str(c.get('table','')).upper()}.{str(c.get('name',c.get('column_name',''))).upper()}"))
                for c in reranked_cols
                if _rank_delta(f"{str(c.get('table','')).upper()}.{str(c.get('name',c.get('column_name',''))).upper()}") > 2][:4]
    demoted  = [(c, _rank_delta(f"{str(c.get('table','')).upper()}.{str(c.get('name',c.get('column_name',''))).upper()}"))
                for c in reranked_cols
                if _rank_delta(f"{str(c.get('table','')).upper()}.{str(c.get('name',c.get('column_name',''))).upper()}") < -2][:4]

    def _pill(ct: str, cn: str, delta: int, up: bool) -> str:
        bg   = "#dcfce7" if up else "#fee2e2"
        bdr  = "#86efac" if up else "#fca5a5"
        clr  = "#166534" if up else "#991b1b"
        arrow = "▲" if up else "▼"
        return (
            f"<span style='background:{bg};border:1px solid {bdr};"
            f"color:{clr};font-size:0.7rem;font-weight:700;"
            f"padding:2px 8px;border-radius:8px;font-family:monospace;"
            f"display:inline-block;margin:2px'>"
            f"{ct}.{cn} <span style='opacity:0.7'>{arrow}</span></span>"
        )

    boost_pills = "".join(
        _pill(str(c.get("table","")).upper(),
              str(c.get("name",c.get("column_name",""))).upper(),
              d, True)
        for c, d in boosted
    )
    demote_pills = "".join(
        _pill(str(c.get("table","")).upper(),
              str(c.get("name",c.get("column_name",""))).upper(),
              abs(d), False)
        for c, d in demoted
    )

    expl_html += (
        f"<div style='background:#eff6ff;border:1px solid #bfdbfe;"
        f"border-radius:10px;padding:14px 16px'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
        f"<div style='min-width:28px;height:28px;border-radius:50%;background:#1d4ed8;"
        f"color:#fff;font-weight:800;font-size:0.85rem;display:flex;"
        f"align-items:center;justify-content:center'>2</div>"
        f"<div style='font-size:0.9rem;font-weight:700;color:#1e293b'>"
        f"LLM Reranker — Business-Priority Rescoring</div></div>"
        f"<p style='font-size:0.82rem;color:#374151;margin-bottom:8px'>"
        f"The raw KG results were passed to the LLM reranker, which scores each column "
        f"on three axes: <b>(a) semantic similarity</b> to the query intent, "
        f"<b>(b) business-criticality</b> (absolute importance of an asset), "
        f"and <b>(c) query-intent match</b> — e.g. a 'reserve' query boosts "
        f"<code>RESERVE_AMT</code>, a 'fraud' query boosts <code>FRAUD_SCORE</code>. "
        f"Geographic columns (<code>STATE_CD</code>, <code>CITY_NM</code>) are "
        f"suppressed when the query has no geographic intent.</p>"
    )
    if boost_pills or demote_pills:
        expl_html += (
            f"<div style='display:flex;flex-wrap:wrap;gap:4px;margin-top:4px'>"
        )
        if boost_pills:
            expl_html += (
                f"<span style='font-size:0.72rem;color:#166534;font-weight:600;"
                f"margin-right:4px'>Promoted:</span>{boost_pills}"
            )
        if demote_pills:
            expl_html += (
                f"<span style='font-size:0.72rem;color:#991b1b;font-weight:600;"
                f"margin-right:4px;margin-left:8px'>Demoted:</span>{demote_pills}"
            )
        expl_html += "</div>"
    expl_html += "</div>"

    # Step 3: SQL column selection
    sql_col_list = sorted(
        f"{t}.{c}" for (t, c) in sql_refs2 if t in real_tables_sql2
    )
    sql_col_pills = "".join(
        f"<span style='background:#ccfbf1;border:1px solid #5eead4;"
        f"color:#0f766e;font-family:monospace;font-size:0.72rem;font-weight:700;"
        f"padding:2px 8px;border-radius:5px;display:inline-block;margin:2px'>"
        f"✓ {fc}</span>"
        for fc in sql_col_list
    ) or "<span style='color:#94a3b8;font-size:0.78rem'>SELECT * (all columns)</span>"

    fk_used = _extract_fk_edges(generated_sql, join_conditions,
                                real_tables_sql2, alias_map_sql2)
    fk_str  = " → ".join(f"{s} ⟷ {d}" for s, d, _ in fk_used) if fk_used else "None (single-table query)"

    expl_html += (
        f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;"
        f"border-radius:10px;padding:14px 16px'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
        f"<div style='min-width:28px;height:28px;border-radius:50%;background:#059669;"
        f"color:#fff;font-weight:800;font-size:0.85rem;display:flex;"
        f"align-items:center;justify-content:center'>3</div>"
        f"<div style='font-size:0.9rem;font-weight:700;color:#1e293b'>"
        f"Final Column Selection &amp; SQL Generation</div></div>"
        f"<p style='font-size:0.82rem;color:#374151;margin-bottom:8px'>"
        f"Reranked columns above the relevance threshold were passed to the SQL generator. "
        f"The LLM resolved the join path "
        f"(<b style='color:#1d4ed8'>{fk_str}</b>) "
        f"and grounded the SELECT / WHERE / ORDER BY clauses using only "
        f"columns verified to exist in the schema. The {len(sql_col_list)} column(s) "
        f"below are those that made it into the final SQL:</p>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px'>{sql_col_pills}</div>"
        f"</div>"
    )

    expl_html += "</div>"
    st.markdown(expl_html, unsafe_allow_html=True)


def _build_rerank_reason(col_name: str, question: str, delta: int, change_type: str) -> str:
    """Generate a short human-readable reason for why a column was boosted, dropped, or kept."""
    cn = col_name.upper()
    q  = question.lower()
    if change_type == "boosted":
        if cn in _PRIORITY_COLS:
            return "Priority column; promoted by business-critical rule"
        for keyword, boosts in _INTENT_BOOST:
            if keyword in q and cn in {k.upper() for k in boosts}:
                return f"Intent boost: query contains «{keyword}»"
        return f"Promoted {delta} positions by reranker"
    elif change_type == "dropped":
        GEO_COLS = {"STATE_CD", "CITY_NM", "ZIP_CD", "COUNTY_CD"}
        if cn in GEO_COLS and not _has_geo_intent(question):
            return "Geo column suppressed — no geographic intent detected"
        return "Score below reranked relevance threshold for this query"
    else:
        if cn in _PRIORITY_COLS:
            return "Stable: high-priority core column"
        return "Rank unchanged after reranking"


def _build_drop_reason(col_name: str, score: float, question: str) -> str:
    """Generate a short explanation for why a column was filtered out before SQL generation."""
    cn = col_name.upper()
    GEO_COLS = {"STATE_CD", "CITY_NM", "ZIP_CD", "COUNTY_CD"}
    if cn in GEO_COLS and not _has_geo_intent(question):
        return "Geographic column filtered — query has no location context"
    if score < 0.70:
        return f"Low semantic similarity ({score:.3f}) — below SQL grounding threshold"
    if score < 0.75:
        return f"Below display threshold (0.75) — borderline relevance ({score:.3f})"
    generic_suffixes = ("_CD", "_FLG", "_IND", "_NM", "_DT")
    if any(cn.endswith(s) for s in generic_suffixes) and cn not in _PRIORITY_COLS:
        return "Generic metadata column; not needed for this specific query"
    return "Not referenced in SQL and below priority threshold"


# ── AI Summary: generate from actual result DataFrame ────────────────────────
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


# ── Results area ──────────────────────────────────────────────────────────────
# Only render pipeline results when the last action was a real data query
# (not a definitional answer or irrelevant-query warning).
_show_results = (
    st.session_state.history
    and not st.session_state.get("definitional_query")
    and not st.session_state.get("irrelevant_query")
)

# ── KG Reasoning Explorer tab ─────────────────────────────────────────────────
if st.session_state.get("active_tab") == "kg_explorer":
    if _show_results:
        last_q_kg, last_r_kg = st.session_state.history[0]
        _render_kg_reasoning_explorer(last_q_kg, last_r_kg.trace)
    else:
        _render_kg_reasoning_explorer("", None)

elif _show_results:
    last_q, last_r = st.session_state.history[0]

    st.markdown(f"### Results for: *{last_q}*")

    # ── Status bar ────────────────────────────────────────────────────────────
    cols = st.columns(4)
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
        score = f"{last_r.trace.max_retrieval_score:.2%}"
        st.markdown(f"<div class='metric-card'><b>KG Score</b><br>{score}</div>",
                    unsafe_allow_html=True)

    # ── Error message ─────────────────────────────────────────────────────────
    if not last_r.success:
        st.error(f"Query failed: {last_r.error}")

    # ── Results dataframe ─────────────────────────────────────────────────────
    if last_r.df is not None and not last_r.df.empty:
        st.dataframe(last_r.df, use_container_width=True, height=400)

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
        with st.expander("Query Trace", expanded=True):
            t = last_r.trace

            # ── Stage 1: HyDE expansion (trimming done upstream) ──────────────
            st.markdown("**Stage 1 — HyDE Expanded Query:**")
            st.text_area("", t.expanded_query or "", height=80,
                         label_visibility="collapsed", key="hyde_out",
                         disabled=True)

            # ── Stage 2: reranked & deduplicated columns ───────────────────────
            st.markdown(f"**Stage 2 — Retrieved Columns:**")
            if t.retrieved_columns:
                reranked = _rerank_columns(t.retrieved_columns, question=last_q)
                # Filter out low-relevance columns from the trace display.
                # Columns scoring below 0.60 are typically FK-chain noise that
                # the retriever pulled in but the SQL generator won't actually use.
                DISPLAY_SCORE_THRESHOLD = 0.75
                reranked_filtered = [
                    c for c in reranked
                    if c.get("score", 0) >= DISPLAY_SCORE_THRESHOLD
                ] or reranked[:10]  # fallback: always show at least top-10
                df_cols = pd.DataFrame(reranked_filtered)
                # Normalise column names: support both "name"/"table" (dataclass)
                # and any legacy "column_name" key so the display is always clean.
                display_cols = []
                rename_map: dict[str, str] = {}
                for preferred, fallback in [("table", None), ("name", "column_name"), ("description", None), ("score", None)]:
                    if preferred in df_cols.columns:
                        display_cols.append(preferred)
                    elif fallback and fallback in df_cols.columns:
                        display_cols.append(fallback)
                        rename_map[fallback] = preferred
                df_display = df_cols[display_cols].rename(columns=rename_map)
                # Format score as percentage for readability
                if "score" in df_display.columns:
                    df_display = df_display.copy()
                    df_display["score"] = df_display["score"].apply(
                        lambda s: f"{float(s):.1%}" if s is not None and s != "" else s
                    )
                st.dataframe(df_display, use_container_width=True, height=250)

            # ── Stage 3: SQL-aware joins only ─────────────────────────────────
            st.markdown("**Stage 3 — Join Conditions (present in generated SQL):**")
            sql_joins = _extract_sql_joins(t.generated_sql, t.join_conditions or [])
            if sql_joins:
                for jc in sql_joins:
                    st.code(jc, language="sql")
            else:
                st.caption("No joins required.")

            # ── Live Traversal Graph ──────────────────────────────────────────
            st.markdown("**Live Traversal Graph:**")
            _trace_reranked = _rerank_columns(t.retrieved_columns or [], question=last_q)
            _render_traversal_graph(
                sql=t.generated_sql or "",
                retrieved_columns=t.retrieved_columns or [],
                join_conditions=t.join_conditions or [],
                question=last_q,
                height=430,
                reranked_columns=_trace_reranked,
            )

            # ── Stage 4+5: Generated SQL ──────────────────────────────────────
            st.markdown(f"**Stage 4+5 — Generated SQL** "
                        f"(repair attempts: {t.sql_repair_attempts}):")
            st.markdown(
                f"<div class='sql-box'>{t.generated_sql}</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── History ───────────────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("### Query History")
        for i, (hq, hr) in enumerate(st.session_state.history[1:], 1):
            icon = "+" if hr.success else "x"
            rows = len(hr.df) if hr.df is not None else 0
            with st.expander(f"[{icon}] {hq}  —  {rows} rows · {hr.trace.elapsed_seconds}s"):
                if hr.df is not None and not hr.df.empty:
                    st.dataframe(hr.df, use_container_width=True, height=200)
                elif not hr.success:
                    st.error(hr.error)

else:
    # Welcome state — only shown when not in KG explorer and no results yet
    if st.session_state.get("active_tab") != "kg_explorer":
        st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #666;">
        <h3>Welcome</h3>
        <p>Type a question above about your P&amp;C insurance data to get started.</p>
        <p>Your question will be processed through the 5-stage Knowledge Graph pipeline:<br>
        <b>HyDE Expansion &rarr; KG Retrieval &rarr; Join Paths &rarr; SQL Generation &rarr; Execution</b></p>
    </div>
    """, unsafe_allow_html=True)