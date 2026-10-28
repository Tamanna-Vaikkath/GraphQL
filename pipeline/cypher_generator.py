"""
pipeline/cypher_generator.py — Cypher lane: Knowledge Graph query generation.

Mirrors pipeline/sql_generator.py but targets the *business data* graph
built by kg/graph_ingest.py:

    (:Claimant)-[:FILED]->(:Claim)-[:COVERED_BY]->(:Policy)
    (:Claim)-[:HAS_PAYMENT]->(:Payment)
    (:Claim)-[:HANDLED_BY]->(:Adjuster)
    (:Claim)-[:SAME_ADJUSTER_AS]->(:Claim)
    (:Policy)-[:SAME_AGENT_AS]->(:Policy)

This lane is used when the question is dominated by relationships,
multi-hop connections, or unknown-depth traversals — the class of question
plain SQL joins handle poorly because the number of hops isn't known ahead
of time.

ADJUSTER CLUSTERING
--------------------
The :Adjuster node (a real entity, not a precomputed pairwise edge) lets the
LLM-generated Cypher perform genuine multi-hop traversal to find *clusters*
of claims connected through a shared adjuster (or other shared entities —
agent, payee) rather than only single-hop pairwise matches. In addition to
LLM-generated Cypher, `find_claim_clusters()` below runs a deterministic,
Python-side connected-components pass over the graph for cases where a
robust, complete cluster listing (rather than one generated query) is
wanted.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)


class CypherValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


from pathlib import Path

import yaml

from kg.ontology import load_ontology, load_schema_block

# ── The ontology is the single source for the schema, the few-shots and the validator ──────────────────
_ONTO = load_ontology()
# Fallback copy (from the YAML file). CypherGenerator prefers the copy stored IN Neo4j (:OntClass / :ONT_REL).
GRAPH_SCHEMA_BLOCK = _ONTO.schema_block()

_RULES = """Rules:
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
- NEVER use the result of an aggregating function (collect(), count(),
  sum(), etc.) as part of a graph pattern / MATCH clause — e.g.
  `collect(DISTINCT x)<-[:REL]-(y)` is invalid Cypher syntax. Aggregate
  ONLY after all MATCH/WHERE pattern-matching for that WITH/RETURN is done.
- If the question involves grouping claims by MORE THAN ONE kind of shared
  entity (e.g. "shared adjusters, agents, OR payees"), write it as separate
  MATCH...WITH...RETURN blocks combined with UNION (one per shared-entity
  type), each ending in its own aggregation — do not try to combine
  different relationship traversals and aggregations into a single MATCH
  pattern.
- NEVER use query parameters (e.g. $name, $id). Inline every literal value
  directly into the Cypher text — the query is executed with no parameter
  binding.
- shortestPath() and allShortestPaths() must wrap EXACTLY ONE relationship
  pattern (a single, untyped, variable-length hop like -[*..6]- is fine).
  NEVER chain multiple relationships — typed or not — inside a
  shortestPath()/allShortestPaths() call (e.g.
  shortestPath((a)-[:REL1]->()<-[:REL2]-(b)) is invalid Cypher). If the
  question mentions a specific intermediate entity type (e.g. "through a
  shared adjuster"), still use a single untyped variable-length pattern —
  do not hardcode the intermediate hop's relationship type(s).
"""


def _build_system(schema_block: str) -> str:
    return (
        "You are a precise Cypher query generator for Neo4j.\n"
        "Generate a single Cypher query only. Return ONLY the Cypher statement —\n"
        "no explanation, no markdown fences, no comments.\n\n"
        f"{schema_block}\n\n{_RULES}"
    )


def _load_fewshot() -> str:
    """Few-shot examples come from ontology/fewshot.yaml; tests/run_cq_tests.py --fewshot executes every one of them."""
    path = Path(__file__).resolve().parent.parent / "ontology" / "fewshot.yaml"
    ex = yaml.safe_load(path.read_text(encoding="utf-8"))["examples"]
    return "\n" + "\n".join(f"Example {i}\nQuestion: {e['q']}\nCypher:\n{e['cypher'].strip()}\n" for i, e in enumerate(ex, 1))


_FEWSHOT = _load_fewshot()
_ALLOWED_LABELS = _ONTO.allowed_labels
_ALLOWED_RELS = _ONTO.allowed_rels

# Guardrail: generated Cypher must never use query parameters (e.g. "$state"
# or "$1"). CypherExecutor.execute() runs the statement as
# `session.run(cypher)` with no parameter dict, so a parameterized query
# fails at the database with a "missing parameter" error instead of
# returning results — all literal values must be inlined into the Cypher
# text itself.
_NO_PARAMS_RULE = re.compile(r"\$\w+")


def _find_balanced_parens(text: str, open_idx: int) -> str:
    """Given the index of an opening '(', return the substring up to its
    matching closing ')'."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return text[open_idx + 1:]


def _check_shortest_path_single_relationship(cypher: str) -> None:
    """
    shortestPath()/allShortestPaths() only accept a pattern with exactly
    one relationship (typed or untyped, optionally variable-length). Catch
    the common LLM mistake of chaining two relationships through an
    intermediate node — e.g.
    shortestPath((c1)-[:HANDLED_BY]->(:Adjuster)<-[:HANDLED_BY]-(c2)) — as
    a fast, deterministic check rather than round-tripping to Neo4j.
    """
    for fn in ("shortestPath", "allShortestPaths"):
        for m in re.finditer(rf"{fn}\s*\(", cypher):
            open_idx = m.end() - 1
            inner = _find_balanced_parens(cypher, open_idx)
            rel_count = len(re.findall(r"\[[^\]]*\]", inner))
            if rel_count > 1:
                raise CypherValidationError(
                    f"Generated Cypher calls {fn}(...) with {rel_count} relationships "
                    f"in the pattern: {fn}({inner}). {fn}() only accepts a SINGLE "
                    "relationship pattern (a single untyped, variable-length hop like "
                    "-[*..6]- is fine) — it cannot chain multiple relationships through "
                    "an intermediate node. Use a single variable-length relationship and "
                    "let it discover the connecting path instead of hardcoding each hop."
                )


_STRING_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_NODE_TXT = r"\(\s*\w*\s*(?::\s*\w+)?[^()]*\)"
# node  <-[ ... ]-  node   (overlap-safe via lookahead so chains like (a)-[:R]->(b)-[:S]->(c) yield both hops)
_HOP_RE = re.compile(r"(?=(" + _NODE_TXT + r")\s*(<?)-\s*\[([^\]]*)\]\s*-(>?)\s*(" + _NODE_TXT + r"))")


def _ontology_checks(cypher: str) -> None:
    """Ground generated Cypher in the ontology: relationship DIRECTION / domain-range, then property names."""
    text = _STRING_RE.sub("''", cypher)
    # 1. relationship domain/range + direction (only where both end nodes carry a label and the hop has one plain type)
    for m in _HOP_RE.finditer(text):
        left, back, relbody, fwd, right = m.groups()
        lcls = re.match(r"\(\s*\w*\s*:\s*(\w+)", left)
        rcls = re.match(r"\(\s*\w*\s*:\s*(\w+)", right)
        typ = re.match(r"\s*\w*\s*:\s*(\w+)\s*(?:\*.*)?$", relbody)
        if not (lcls and rcls and typ) or "|" in relbody:
            continue
        lc, rc, rel = lcls.group(1), rcls.group(1), typ.group(1)
        if lc not in _ONTO.classes or rc not in _ONTO.classes or rel not in _ALLOWED_RELS:
            continue                                # unknown names are reported by the label check
        if back and not fwd:
            frm, to = rc, lc
        elif fwd and not back:
            frm, to = lc, rc
        else:                                       # undirected: accept either orientation
            if _ONTO.edge_allowed(rel, lc, rc) or _ONTO.edge_allowed(rel, rc, lc):
                continue
            frm, to = lc, rc
        if not _ONTO.edge_allowed(rel, frm, to):
            declared = ", ".join(f"(:{r['from']})-[:{rel}]->(:{r['to']})" for r in _ONTO.rels if r["rel"] == rel)
            raise CypherValidationError(f"The ontology does not allow (:{frm})-[:{rel}]->(:{to}). Declared: {declared}.")
    # 2. property names: variable -> label(s), then every var.prop and inline {prop: ...}
    var_labels = {}
    for nm in re.finditer(r"\(\s*(\w+)\s*:\s*(\w+)\s*(\{[^}]*\})?", text):
        var, lab, inline = nm.group(1), nm.group(2), nm.group(3)
        if lab in _ONTO.classes:
            var_labels.setdefault(var, set()).add(lab)
            for key in re.findall(r"(\w+)\s*:", inline or ""):
                if key not in _ONTO.props_any(lab):
                    raise CypherValidationError(f"Property '{key}' is not declared on :{lab} in the ontology.")
    for pm in re.finditer(r"\b(\w+)\.(\w+)\b", text):
        var, prop = pm.group(1), pm.group(2)
        if var in var_labels:
            allowed = set().union(*(_ONTO.props_any(l) for l in var_labels[var]))
            if prop not in allowed:
                raise CypherValidationError(
                    f"Property '{prop}' is not declared on :{'/'.join(sorted(var_labels[var]))} in the ontology "
                    f"(declared: {', '.join(sorted(allowed))}).")


def _validate_cypher(cypher: str) -> None:
    """Guardrail: block writes, enforce the ontology (labels, relationship direction, properties), forbid parameters."""
    upper = cypher.upper()
    for forbidden in ("DELETE", "REMOVE", "CREATE", "MERGE", "SET", "DROP"):
        if re.search(rf"\b{forbidden}\b", upper):
            raise CypherValidationError(
                f"Generated Cypher contains a write/destructive clause ({forbidden}), "
                "which is not permitted for read-only KG queries."
            )
    used_labels = set(re.findall(r":([A-Z][A-Za-z_]*)", _STRING_RE.sub("''", cypher)))
    unknown = {l for l in used_labels if l not in _ALLOWED_LABELS and l not in _ALLOWED_RELS}
    if unknown:
        raise CypherValidationError(
            f"Generated Cypher references unknown label(s)/relationship(s): {sorted(unknown)}. "
            "Use only classes and relationships defined in the ontology."
        )
    param_hit = _NO_PARAMS_RULE.search(cypher)
    if param_hit:
        raise CypherValidationError(
            f"Generated Cypher uses a query parameter ({param_hit.group(0)}), "
            "but this query is executed without a parameter binding. All values "
            "must be inlined as literals in the Cypher text."
        )
    _ontology_checks(cypher)
    _check_shortest_path_single_relationship(cypher)


class CypherGenerator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)
        block, source = load_schema_block(cfg)          # prefers the ontology stored IN Neo4j; falls back to the YAML file
        self._schema_block, self.schema_source = block, source
        self._system = _build_system(block) + "\n" + _FEWSHOT
        print(f"[CypherGenerator INIT] ready — schema block built from the ontology ({source}), v{_ONTO.version}, "
              f"{len(_ONTO.concrete())} classes, {len(_ONTO.rels)} relationship types")

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


# ── True multi-hop cluster analysis (deterministic, not LLM-generated) ───────

def find_claim_clusters(
    cfg: Config, min_cluster_size: int = 2,
) -> List[Dict[str, Any]]:
    """
    Deterministic, Python-side companion to Example 5 above: walks the graph
    to find connected components ("clusters") of claims that are transitively
    linked through a shared adjuster (:HANDLED_BY -> :Adjuster), a shared
    agent (via each claim's :COVERED_BY policy), or a shared payment payee.

    This demonstrates genuine multi-hop graph analysis — claims A and B can
    end up in the same cluster even without a direct edge between them, as
    long as a chain of shared-entity connections links them (e.g. A and B
    share an adjuster, and B and C share an agent, so A/B/C form one
    cluster). It runs the connectivity as three focused Cypher queries and
    unions the results with a lightweight union-find, since building this as
    a single generated Cypher statement is far less reliable than a fixed,
    tested traversal for a capability this central to the demo.
    """
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    edges: List[Tuple[str, str, str]] = []
    try:
        with driver.session(database=cfg.neo4j_database) as session:
            # Claims that share the same adjuster
            for r in session.run("""
                MATCH (c1:Claim)-[:HANDLED_BY]->(a:Adjuster)<-[:HANDLED_BY]-(c2:Claim)
                WHERE c1.claim_id < c2.claim_id
                RETURN c1.claim_id AS claim_a, c2.claim_id AS claim_b
            """):
                edges.append((r["claim_a"], r["claim_b"], "adjuster"))

            # Claims connected through the same agent (via their policy)
            for r in session.run("""
                MATCH (c1:Claim)-[:COVERED_BY]->(p1:Policy),
                      (c2:Claim)-[:COVERED_BY]->(p2:Policy)
                WHERE c1.claim_id < c2.claim_id
                  AND p1.agent_id IS NOT NULL AND p1.agent_id = p2.agent_id
                RETURN c1.claim_id AS claim_a, c2.claim_id AS claim_b
            """):
                edges.append((r["claim_a"], r["claim_b"], "agent"))

            # Claims connected through the same payment payee
            for r in session.run("""
                MATCH (c1:Claim)-[:HAS_PAYMENT]->(p1:Payment),
                      (c2:Claim)-[:HAS_PAYMENT]->(p2:Payment)
                WHERE c1.claim_id < c2.claim_id
                  AND p1.payee_name IS NOT NULL AND p1.payee_name = p2.payee_name
                RETURN c1.claim_id AS claim_a, c2.claim_id AS claim_b
            """):
                edges.append((r["claim_a"], r["claim_b"], "payee"))
    finally:
        driver.close()

    # ── Union-find over all shared-entity edges, regardless of edge type ────
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    reasons: Dict[Tuple[str, str], set] = {}
    for a, b, edge_type in edges:
        union(a, b)
        reasons.setdefault(tuple(sorted((a, b))), set()).add(edge_type)

    components: Dict[str, set] = {}
    for node in parent:
        components.setdefault(find(node), set()).add(node)

    clusters: List[Dict[str, Any]] = []
    for root, members in components.items():
        if len(members) < min_cluster_size:
            continue
        connections = [
            {"claim_a": a, "claim_b": b, "shared_via": sorted(via)}
            for (a, b), via in reasons.items()
            if a in members and b in members
        ]
        clusters.append({
            "claim_ids": sorted(members),
            "size": len(members),
            "connections": connections,
        })

    clusters.sort(key=lambda c: c["size"], reverse=True)
    print(f"[find_claim_clusters] Found {len(clusters)} cluster(s) "
          f"(min_size={min_cluster_size}) from {len(edges)} shared-entity edges.")
    return clusters