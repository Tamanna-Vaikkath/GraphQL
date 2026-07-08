"""
pipeline/cypher_generator.py — Cypher lane: Knowledge Graph query generation.

Mirrors pipeline/sql_generator.py but targets the *business data* graph
built by kg/graph_ingest.py:

    (:Claimant)-[:FILED]->(:Claim)-[:COVERED_BY]->(:Policy)
    (:Claim)-[:HAS_PAYMENT]->(:Payment)
    (:Claim)-[:SAME_ADJUSTER_AS]->(:Claim)
    (:Policy)-[:SAME_AGENT_AS]->(:Policy)

This lane is used when the question is dominated by relationships,
multi-hop connections, or unknown-depth traversals — the class of question
plain SQL joins handle poorly because the number of hops isn't known ahead
of time.
"""
from __future__ import annotations

import logging
import re

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)


class CypherValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


GRAPH_SCHEMA_BLOCK = """
Knowledge Graph schema (Neo4j property graph of real insurance data):

Nodes:
  (:Claimant {claimant_id, name, dob, gender, state, address, phone,
              attorney_represented, claim_count, fraud_risk_score})
  (:Policy   {policy_id, policy_number, insured_name, effective_date,
              expiration_date, line_of_business, state, premium_amount,
              deductible_amount, agent_id, status})
  (:Claim    {claim_id, status, loss_date, report_date, loss_type,
              incurred_amount, reserve_amount, adjuster_id, close_date,
              litigation_flag})
  (:Payment  {payment_id, payment_date, gross_amount, net_amount, status,
              type, payee_name, check_number, void_reason})

Relationships:
  (:Claimant)-[:FILED]->(:Claim)
  (:Claim)-[:COVERED_BY]->(:Policy)
  (:Claim)-[:HAS_PAYMENT]->(:Payment)
  (:Claim)-[:SAME_ADJUSTER_AS]->(:Claim)   -- undirected in practice; only one direction stored
  (:Policy)-[:SAME_AGENT_AS]->(:Policy)    -- undirected in practice; only one direction stored

Status code meanings:
  Claim.status:  O=Open, C=Closed, P=Pending, D=Denied
  Policy.status: AC=Active, CN=Cancelled, EX=Expired
  Payment.status: IS=Issued, CL=Cleared, VD=Voided, PD=Pending
  attorney_represented / litigation_flag: 'Y' or 'N'

For relationships that could go either direction (SAME_ADJUSTER_AS,
SAME_AGENT_AS), match without a direction arrow, e.g.
  (a:Claim)-[:SAME_ADJUSTER_AS]-(b:Claim)
"""

_CYPHER_SYSTEM = f"""You are a precise Cypher query generator for Neo4j.
Generate a single Cypher query only. Return ONLY the Cypher statement —
no explanation, no markdown fences, no comments.

{GRAPH_SCHEMA_BLOCK}

Rules:
- Use variable-length patterns (e.g. *1..4 or shortestPath) when the
  question implies an unknown number of hops or a "connection" between
  entities rather than one specific relationship.
- Always RETURN concrete scalar/property values (not whole nodes) so
  results can be rendered in a table — e.g. RETURN c.name AS claimant_name.
- Use DISTINCT when a traversal could revisit the same entity via multiple
  paths.
- Always include a LIMIT (default 100) unless the question asks for a
  single aggregate value.
- Only use labels, relationship types, and properties defined in the schema
  above. Do not invent new ones.
"""

_FEWSHOT = """
Example 1
Question: Which claimants share the same policy as another claimant?
Cypher:
MATCH (c1:Claimant)-[:FILED]->(:Claim)-[:COVERED_BY]->(p:Policy)<-[:COVERED_BY]-(:Claim)<-[:FILED]-(c2:Claimant)
WHERE c1.claimant_id < c2.claimant_id
RETURN DISTINCT c1.name AS claimant_1, c2.name AS claimant_2, p.policy_number AS shared_policy
LIMIT 100

Example 2
Question: Find claims connected through the same adjuster, up to 3 hops away.
Cypher:
MATCH path = (a:Claim)-[:SAME_ADJUSTER_AS*1..3]-(b:Claim)
WHERE a.claim_id <> b.claim_id
RETURN DISTINCT a.claim_id AS claim_a, b.claim_id AS claim_b, length(path) AS hops
LIMIT 100

Example 3
Question: Show the chain of payments across all claims filed by claimant 'John Smith'.
Cypher:
MATCH (c:Claimant {name: 'John Smith'})-[:FILED]->(cl:Claim)-[:HAS_PAYMENT]->(p:Payment)
RETURN cl.claim_id AS claim_id, p.payment_id AS payment_id, p.payment_date AS payment_date,
       p.gross_amount AS gross_amount, p.status AS status
ORDER BY cl.claim_id, p.payment_date
LIMIT 100

Example 4
Question: What is the shortest connection path between policy PL-2021-00042 and claimant Jane Doe?
Cypher:
MATCH (p:Policy {policy_number: 'PL-2021-00042'}), (c:Claimant {name: 'Jane Doe'})
MATCH path = shortestPath((p)-[*..6]-(c))
RETURN [n IN nodes(path) | coalesce(n.name, n.claim_id, n.policy_number, n.payment_id)] AS path_nodes,
       length(path) AS hops
LIMIT 1
"""

_ALLOWED_LABELS = {"Claimant", "Policy", "Claim", "Payment"}
_ALLOWED_RELS = {"FILED", "COVERED_BY", "HAS_PAYMENT", "SAME_ADJUSTER_AS", "SAME_AGENT_AS"}


def _validate_cypher(cypher: str) -> None:
    """Lightweight guardrail: block destructive ops and unknown labels."""
    upper = cypher.upper()
    for forbidden in ("DELETE", "REMOVE", "CREATE", "MERGE", "SET", "DROP"):
        if re.search(rf"\b{forbidden}\b", upper):
            raise CypherValidationError(
                f"Generated Cypher contains a write/destructive clause ({forbidden}), "
                "which is not permitted for read-only KG queries."
            )
    used_labels = set(re.findall(r":([A-Z][A-Za-z_]*)", cypher))
    unknown = {l for l in used_labels if l not in _ALLOWED_LABELS and l not in _ALLOWED_RELS}
    if unknown:
        raise CypherValidationError(
            f"Generated Cypher references unknown label(s)/relationship(s): {sorted(unknown)}. "
            "Please rephrase your question using Claims, Claimants, Policies, or Payments."
        )


class CypherGenerator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)
        self._system = _CYPHER_SYSTEM + "\n" + _FEWSHOT
        print("[CypherGenerator INIT] ready (graph schema: Claimant/Policy/Claim/Payment)")

    def generate(self, question: str) -> str:
        print(f"\n[Cypher lane] CypherGenerator.generate()")
        print(f"  Question: {question}")

        cypher = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            self._system, f"Question: {question}",
            temperature=0.0, max_tokens=600,
        )
        cypher = (
            cypher.strip()
            .removeprefix("```cypher").removeprefix("```")
            .removesuffix("```").strip()
        )
        print(f"[Cypher lane] Generated Cypher:\n{cypher}")

        _validate_cypher(cypher)
        return cypher
