"""
pipeline/query_router.py — Decides which swim lane a question belongs to.

Routing rule (business intent, not phrasing):
  SQL          -> entity/attribute-centric questions: filters, aggregations,
                  reporting, and FIXED-depth joins (one known hop, e.g.
                  "claims with their policy").
  Cypher / KG  -> questions where relationships, multi-hop connections, or
                  UNKNOWN-depth traversals are central to the answer
                  ("claimants who share a policy", "claims connected through
                  the same adjuster", "chain of payments across a claimant's
                  claims", "shortest path between X and Y", etc.)

Approach: fast, explainable keyword/pattern heuristics first (cheap, and
gives the demo a deterministic story to point at). Falls back to an LLM
classifier only when the heuristics are inconclusive.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)

SQL_LANE = "SQL"
CYPHER_LANE = "CYPHER"

# ── Signals that point at relationships / multi-hop / unknown depth ──────────
_CYPHER_PATTERNS = [
    r"\bshare[sd]?\b.*\b(policy|policies|claim|claims|adjuster|agent|address)\b",
    r"\bin common\b",
    r"\bsame (adjuster|agent|policy|address|claimant)\b",
    r"\bconnected\b", r"\bconnection[s]?\b",
    r"\brelated to\b", r"\brelationship[s]?\b",
    r"\bnetwork\b",
    r"\bchain[s]?\b",
    r"\bmulti[- ]?hop\b",
    r"\bpath[s]? (between|from|to|connecting)\b",
    r"\bshortest path\b",
    r"\btraversal\b", r"\btraverse\b",
    r"\bhow (is|are) .* (connected|linked|related)\b",
    r"\blinked (to|via|through)\b",
    r"\bthrough (the )?(same|shared)\b",
    r"\bvia (the )?(same|shared)\b",
    r"\bwho else\b",
    r"\bfriend[- ]?of[- ]?friend\b",
    r"\b\d+[- ]?(hop|hops|degree|degrees)\b",
    r"\bwithin \d+ (hops?|steps?|degrees?)\b",
    r"\bunknown depth\b",
    r"\bindirect(ly)?\b",
    r"\bcluster[s]?\b",
]

# ── Signals that point squarely at fixed-schema entity/attribute work ────────
_SQL_PATTERNS = [
    r"\b(count|sum|avg|average|total|min|max)\b",
    r"\btop \d+\b",
    r"\bgroup by\b", r"\bbreakdown\b", r"\bby (state|status|type|line of business|month|year)\b",
    r"\blist\b", r"\bshow\b", r"\bfind\b", r"\bfilter\b",
    r"\breport\b",
    r"\bgreater than\b|\bless than\b|\babove\b|\bbelow\b|\bbetween\b",
    r"\bwith (a |an )?(status|amount|premium|deductible|score)\b",
]

_CYPHER_RE = re.compile("|".join(_CYPHER_PATTERNS), re.IGNORECASE)
_SQL_RE = re.compile("|".join(_SQL_PATTERNS), re.IGNORECASE)

_ROUTER_SYSTEM = (
    "You are a query-routing classifier for a P&C insurance analytics system "
    "that can answer questions two ways:\n\n"
    "SQL       -> best for entity/attribute-centric questions: filters, "
    "aggregations (count/sum/avg/top-N), reporting, and FIXED-depth joins "
    "(a single, known hop such as 'claims with their policy').\n"
    "CYPHER    -> best when relationships, multi-hop connections, or "
    "UNKNOWN-depth traversals are central to answering the question — e.g. "
    "'claimants who share a policy with another claimant', 'claims connected "
    "through the same adjuster', 'chain of payments across a claimant's "
    "claims', shortest/variable-length paths between entities.\n\n"
    "Respond with exactly one word: SQL or CYPHER."
)


@dataclass
class RouteDecision:
    lane: str
    reason: str
    method: str  # "heuristic" | "llm" | "default"


class QueryRouter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._gpt = get_gpt_client(cfg)

    def route(self, question: str) -> RouteDecision:
        print(f"\n[QueryRouter] Routing question: {repr(question)}")

        cypher_hit = _CYPHER_RE.search(question)
        sql_hit = _SQL_RE.search(question)

        if cypher_hit and not sql_hit:
            reason = (
                f"Matched relationship/multi-hop signal ('{cypher_hit.group(0)}') "
                "→ routed to Cypher / Knowledge Graph lane."
            )
            print(f"[QueryRouter] {reason}")
            return RouteDecision(lane=CYPHER_LANE, reason=reason, method="heuristic")

        if sql_hit and not cypher_hit:
            reason = (
                f"Matched entity/attribute signal ('{sql_hit.group(0)}') "
                "→ routed to SQL lane."
            )
            print(f"[QueryRouter] {reason}")
            return RouteDecision(lane=SQL_LANE, reason=reason, method="heuristic")

        if cypher_hit and sql_hit:
            # Both fired (e.g. "count claimants who share a policy") — the
            # relationship signal takes precedence because the *hard* part
            # of the question is the unknown-depth traversal, not the count.
            reason = (
                f"Both signals matched, but relationship signal "
                f"('{cypher_hit.group(0)}') dominates → Cypher / Knowledge Graph lane."
            )
            print(f"[QueryRouter] {reason}")
            return RouteDecision(lane=CYPHER_LANE, reason=reason, method="heuristic")

        # ── Inconclusive: ask the LLM ──────────────────────────────────────
        try:
            raw = chat_complete(
                self._gpt, self.cfg.openai_deployment_name,
                _ROUTER_SYSTEM, f"Question: {question}",
                temperature=0.0, max_tokens=5,
            ).strip().upper()
            lane = CYPHER_LANE if "CYPHER" in raw else SQL_LANE
            reason = f"No strong keyword signal — LLM classifier chose {lane}."
            print(f"[QueryRouter] {reason}")
            return RouteDecision(lane=lane, reason=reason, method="llm")
        except Exception as e:
            logger.warning("Router LLM classification failed: %s", e)
            print(f"[QueryRouter] LLM classification failed ({e}) — defaulting to SQL lane.")
            return RouteDecision(
                lane=SQL_LANE,
                reason="Routing classifier unavailable — defaulted to SQL lane.",
                method="default",
            )
