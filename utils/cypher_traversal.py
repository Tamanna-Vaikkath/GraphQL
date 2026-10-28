"""
utils/cypher_traversal.py — builds the "Live Traversal Graph" model for the
Cypher / Knowledge-Graph lane.

The original Cypher-lane graph was a fixed schema diagram with a few labels
highlighted. This module makes it specific to the question that was asked.
Two views are produced:

1. "Query pattern"  — parsed from the generated Cypher: exactly the node
   labels and relationship types the query walks, in order, with the filters
   (e.g. {claim_id: 1001}), the properties it returns, hop ranges of
   variable-length traversals, and any aggregation.

2. "Matched graph"  — the REAL nodes and relationships behind the answer,
   read back from Neo4j:
     * for path queries (shortestPath / MATCH p = ...) the path(s) themselves;
     * otherwise, the entities named in the result rows plus their direct
       connections over the relationship types the query uses.
   Nodes that appear in the result carry a dark ring; the rest are shown as
   connection context. Every node lists its properties (codes decoded) and
   every relationship carries a plain-English sentence explaining the link.

Neo4j access is isolated in fetch_from_neo4j(); build_cypher_model() takes the
fetcher as an argument so the model can be built (and tested) without a
database.
"""
from __future__ import annotations

import ast
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from utils.traversal_viz import (
    circled, entity_colors, inspector, make_edge, make_node, make_view,
)

# ─────────────────────────────────────────────────────────────────────────────
# Domain knowledge used for explanations
# ─────────────────────────────────────────────────────────────────────────────
LABEL_INFO = {
    "Claimant": "A person who filed one or more claims.",
    "Claim": "One insurance claim — the hub of the graph.",
    "Policy": "The insurance policy that covers a claim.",
    "Payment": "A payment issued against a claim.",
    "Adjuster": "The adjuster handling one or more claims.",
    "LossCause": "A cause of loss (e.g. water damage, rear-end collision) with its peril group.",
    "Cluster": "A detected group of related claims (overlaps, catastrophe event, or shared-payee ring).",
    "ClaimEvent": "One event in a claim's timeline (reported, reserve change, payment, closed...).",
}
REL_INFO = {
    "FILED": "A claimant filed a claim.",
    "COVERED_BY": "A claim is covered by a policy.",
    "HAS_PAYMENT": "A payment was issued against a claim.",
    "HANDLED_BY": "A claim is handled by an adjuster. Two claims sharing an adjuster are connected through this node.",
    "SAME_ADJUSTER_AS": "Shortcut edge: two claims are handled by the same adjuster.",
    "SAME_AGENT_AS": "Shortcut edge: two policies were sold by the same agent.",
    "CAUSED_BY": "The claim's loss was caused by this cause of loss.",
    "IN_CLUSTER": "The claim belongs to this detected cluster of related claims.",
    "OVERLAPS_WITH": "The two claims overlap / collide: same policy and/or claimant with loss dates within 30 days.",
    "SIMILAR_TO": "The second claim is one of the most similar claims to the first (ranked, score 0-1).",
    "HAS_EVENT": "The claim has this event in its timeline.",
    "NEXT": "The next event in time within the same claim.",
}
_STATUS = {
    ("Claim", "status"): {"O": "Open", "C": "Closed", "P": "Pending", "D": "Denied"},
    ("Policy", "status"): {"AC": "Active", "CN": "Cancelled", "EX": "Expired"},
    ("Payment", "status"): {"IS": "Issued", "CL": "Cleared", "VD": "Voided", "PD": "Paid/Pending"},
    ("Payment", "type"): {"INDEM": "Indemnity", "MED": "Medical", "EXP": "Expense"},
    ("ClaimEvent", "from_status"): {"O": "Open", "C": "Closed", "P": "Pending", "D": "Denied"},
    ("ClaimEvent", "to_status"): {"O": "Open", "C": "Closed", "P": "Pending", "D": "Denied"},
}
_MONEY = ("amount", "incurred", "reserve", "premium", "deductible", "total_incurred")
_PROP_ORDER = {
    "Claim": ["claim_id", "status", "loss_date", "report_date", "close_date", "loss_type", "loss_cause", "severity",
              "incurred_amount", "reserve_amount", "adjuster_id", "litigation_flag", "loss_state", "loss_description"],
    "Claimant": ["claimant_id", "name", "state", "attorney_represented", "fraud_risk_score", "claim_count"],
    "Policy": ["policy_id", "policy_number", "insured_name", "line_of_business", "state", "status", "premium_amount",
               "effective_date", "expiration_date", "agent_id"],
    "Payment": ["payment_id", "payment_date", "gross_amount", "net_amount", "status", "type", "payee_name", "void_reason"],
    "Adjuster": ["adjuster_id"],
    "LossCause": ["cause_code", "description", "loss_type", "peril_group"],
    "Cluster": ["cluster_id", "cluster_type", "label", "risk_level", "claim_count", "total_incurred",
                "first_loss_date", "last_loss_date", "span_days", "link_basis"],
    "ClaimEvent": ["seq", "date", "type", "from_status", "to_status", "incurred_at_event", "reserve_at_event", "note"],
}
_HIDDEN_PROPS = {"_policy_id", "_claimant_id", "_claim_id", "embedding"}
CORE_CONTEXT_RELS = ["FILED", "COVERED_BY", "HANDLED_BY", "CAUSED_BY", "IN_CLUSTER"]
MAX_NODES = 90
PER_SEED = 8


# ─────────────────────────────────────────────────────────────────────────────
# Cypher pattern parsing
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PNode:
    key: str
    var: str = ""
    label: str = ""
    props: str = ""
    where: List[str] = field(default_factory=list)
    returned: List[str] = field(default_factory=list)


@dataclass
class PEdge:
    idx: int
    src: str
    dst: str
    types: List[str]
    hops: str = ""
    directed: bool = True
    var: str = ""


@dataclass
class Pattern:
    nodes: "OrderedDict[str, PNode]" = field(default_factory=OrderedDict)
    edges: List[PEdge] = field(default_factory=list)
    shortest: bool = False
    path_var: str = ""
    aggregates: List[str] = field(default_factory=list)
    returns: List[str] = field(default_factory=list)
    order_by: str = ""
    limit: str = ""
    cross_where: List[str] = field(default_factory=list)
    n_returns: int = 0
    has_union: bool = False
    has_with: bool = False


_KEYWORDS = r"(?:WHERE|WITH|RETURN|MATCH|OPTIONAL|UNION|ORDER|LIMIT|UNWIND|CALL|SKIP)"
_MATCH_RE = re.compile(rf"(?:OPTIONAL\s+)?MATCH\s+(.*?)(?=\b{_KEYWORDS}\b|$)", re.I | re.S)
_NODE = r"\(\s*(?P<var>[A-Za-z_]\w*)?\s*(?P<labels>(?::\s*[A-Za-z_]\w*\s*)*)(?P<nprops>\{[^{}]*\})?\s*\)"
_REL = r"(?P<l><)?-\s*(?:\[(?P<body>[^\]]*)\])?\s*-(?P<r>>)?"
_TOKEN_RE = re.compile(rf"(?P<node>{_NODE})|(?P<rel>{_REL})")
_REL_BODY = re.compile(
    r"\s*(?P<var>[A-Za-z_]\w*)?\s*(?P<types>(?::\s*[A-Za-z_]\w*\s*(?:\|\s*:?\s*[A-Za-z_]\w*\s*)*)?)"
    r"\s*(?P<hops>\*\s*\d*\s*(?:\.\.\s*\d*)?)?\s*(?P<props>\{.*\})?\s*$", re.S)


def _split_top(text: str, sep: str = ",") -> List[str]:
    out, depth, cur, quote = [], 0, [], ""
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _last_segment(cypher: str, keyword: str, stops: str) -> str:
    """Text after the LAST top-level `keyword` up to one of `stops` (regex alternation)."""
    m = list(re.finditer(rf"\b{keyword}\b", cypher, re.I))
    if not m:
        return ""
    rest = cypher[m[-1].end():]
    s = re.search(rf"\b(?:{stops})\b", rest, re.I)
    return (rest[: s.start()] if s else rest).strip()


def parse_cypher(cypher: str) -> Pattern:
    pat = Pattern()
    clean = re.sub(r"//.*", "", cypher or "")
    up = clean.upper()
    pat.has_union = " UNION " in f" {up} "
    pat.has_with = bool(re.search(r"\bWITH\b", clean, re.I))
    pat.shortest = bool(re.search(r"\b(?:all)?shortestPath\s*\(", clean, re.I))
    pm = re.search(r"\b([A-Za-z_]\w*)\s*=\s*(?:all)?(?:shortestPath\s*)?\(", clean, re.I)
    pat.path_var = pm.group(1) if pm else ""

    anon = 0
    for mm in _MATCH_RE.finditer(clean):
        text = re.sub(r"\b(?:all)?shortestPath\s*\(", "(", mm.group(1), flags=re.I)
        text = re.sub(r"\b[A-Za-z_]\w*\s*=\s*(?=\()", "", text)
        prev_end = 0
        prev_node: Optional[str] = None
        pending: Optional[dict] = None
        for tok in _TOKEN_RE.finditer(text):
            gap = text[prev_end:tok.start()]
            prev_end = tok.end()
            if "," in gap:
                prev_node, pending = None, None
            if tok.group("node") is not None:
                var = tok.group("var") or ""
                labels = [x.strip() for x in tok.group("labels").split(":") if x.strip()]
                label = labels[0] if labels else ""
                if var and var in pat.nodes:
                    node = pat.nodes[var]
                    if label and not node.label:
                        node.label = label
                    if tok.group("nprops") and not node.props:
                        node.props = tok.group("nprops")
                else:
                    key = var or f"_anon{anon}"
                    anon += 1
                    node = PNode(key=key, var=var, label=label, props=tok.group("nprops") or "")
                    pat.nodes[key] = node
                if prev_node is not None and pending is not None:
                    a, b = prev_node, node.key
                    if pending["l"] and not pending["r"]:
                        a, b = b, a
                    pat.edges.append(PEdge(len(pat.edges), a, b, pending["types"], pending["hops"],
                                           directed=bool(pending["l"] or pending["r"]), var=pending["var"]))
                prev_node, pending = node.key, None
            else:
                body = tok.group("body") or ""
                bm = _REL_BODY.match(body)
                types, hops, rvar = [], "", ""
                if bm:
                    types = [t.strip() for t in re.split(r"[|:]", bm.group("types") or "") if t.strip()]
                    hops = re.sub(r"\s+", "", bm.group("hops") or "")
                    rvar = bm.group("var") or ""
                pending = {"l": tok.group("l"), "r": tok.group("r"), "types": types, "hops": hops, "var": rvar}

    # WHERE conditions -> attach to the variable they mention
    for wm in re.finditer(rf"\bWHERE\b(.*?)(?=\b(?:WITH|RETURN|MATCH|OPTIONAL|UNION|ORDER|LIMIT)\b|$)", clean, re.I | re.S):
        for cond in re.split(r"\bAND\b", wm.group(1), flags=re.I):
            cond = " ".join(cond.split())
            if not cond:
                continue
            vs = {v for v in re.findall(r"\b([A-Za-z_]\w*)\.\w+", cond) if v in pat.nodes}
            if len(vs) == 1:
                pat.nodes[next(iter(vs))].where.append(cond)
            else:
                pat.cross_where.append(cond)

    # RETURN items
    ret = _last_segment(clean, "RETURN", "ORDER BY|LIMIT|UNION|SKIP")
    for item in _split_top(ret):
        item = " ".join(item.split())
        if not item:
            continue
        pat.returns.append(item)
        if re.search(r"\b(?:collect|count|sum|avg|min|max|size)\s*\(", item, re.I):
            pat.aggregates.append(item)
        alias = re.search(r"\bAS\s+(\w+)\s*$", item, re.I)
        for v, p in re.findall(r"\b([A-Za-z_]\w*)\.(\w+)", item):
            if v in pat.nodes:
                pat.nodes[v].returned.append(f"{p}" + (f" AS {alias.group(1)}" if alias else ""))
    pat.n_returns = len(re.findall(r"\bRETURN\b", clean, re.I))
    ob = re.search(r"\bORDER\s+BY\b(.*?)(?=\bLIMIT\b|\bSKIP\b|$)", clean, re.I | re.S)
    pat.order_by = " ".join(ob.group(1).split()) if ob else ""
    lm = re.findall(r"\bLIMIT\s+(\d+)", clean, re.I)
    pat.limit = lm[-1] if lm else ""
    return pat


def path_evidence_query(cypher: str, pat: Pattern, limit: int = 12) -> Optional[str]:
    """For a query that names a path variable (MATCH p = ... / shortestPath),
    rewrite the RETURN so that it returns the path itself. Only done for
    simple single-RETURN queries with no WITH / UNION, where the path variable
    is guaranteed to be in scope."""
    if not pat.path_var or pat.has_union or pat.has_with or pat.n_returns != 1:
        return None
    idx = [m.start() for m in re.finditer(r"\bRETURN\b", cypher, re.I)]
    if not idx:
        return None
    head = cypher[: idx[-1]].rstrip()
    q = f"{head}\nRETURN {pat.path_var} AS evidence_path LIMIT {limit}"
    return q


# ─────────────────────────────────────────────────────────────────────────────
# Seeds from result rows
# ─────────────────────────────────────────────────────────────────────────────
def _flatten(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        out: List[Any] = []
        for x in v:
            out.extend(_flatten(x))
        return out
    if isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]"):
        try:
            return _flatten(ast.literal_eval(v))
        except Exception:
            return [v]
    if isinstance(v, dict):
        return []
    try:
        import pandas as pd
        if pd.isna(v):
            return []
    except Exception:
        pass
    return [v]


def _as_int(v: Any) -> Optional[int]:
    try:
        f = float(v)
        return int(f) if f == int(f) else None
    except Exception:
        return None


def extract_seeds(df) -> Dict[str, List[Any]]:
    seeds: Dict[str, List[Any]] = {k: [] for k in (
        "claim_ids", "claimant_ids", "claimant_names", "policy_numbers", "policy_ids",
        "adjuster_ids", "payment_ids", "cluster_ids", "cause_codes", "event_ids")}
    if df is None or getattr(df, "empty", True):
        return seeds
    head = df.head(60)
    for col in head.columns:
        lc = str(col).lower()
        vals = [x for v in head[col] for x in _flatten(v)]
        if not vals:
            continue
        ints = [i for i in (_as_int(x) for x in vals) if i is not None]
        strs = [str(x) for x in vals if isinstance(x, str)]
        if "claimant" in lc:
            if "id" in lc:
                seeds["claimant_ids"] += ints
            else:
                seeds["claimant_names"] += strs
        elif "claim" in lc and ("id" in lc or re.search(r"claim(s|_[a-z0-9]+)?$", lc)):
            seeds["claim_ids"] += ints
        elif lc in ("name", "insured_name") and strs:
            seeds["claimant_names"] += strs if lc == "name" else []
        elif "policy" in lc:
            (seeds["policy_numbers"] if strs else seeds["policy_ids"]).extend(strs or ints)
        elif "adjuster" in lc and "id" in lc:
            seeds["adjuster_ids"] += ints
        elif "payment" in lc and "id" in lc:
            seeds["payment_ids"] += ints
        elif "cluster" in lc and "id" in lc:
            seeds["cluster_ids"] += ints
        elif ("cause" in lc or lc == "loss_cause") and strs and all(re.fullmatch(r"[A-Z_]{4,}", s) for s in strs):
            seeds["cause_codes"] += strs
        elif "event" in lc and "id" in lc:
            seeds["event_ids"] += ints
    caps = {"claim_ids": 30}
    for k, v in seeds.items():
        seeds[k] = list(dict.fromkeys(v))[: caps.get(k, 20)]
    return seeds


def _has_seeds(seeds: Dict[str, List[Any]]) -> bool:
    return any(seeds.values())


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j fetch (thin, isolated)
# ─────────────────────────────────────────────────────────────────────────────
_SEED_QUERY = """
MATCH (n)
WHERE (n:Claim AND n.claim_id IN $claim_ids)
   OR (n:Claimant AND (n.claimant_id IN $claimant_ids OR n.name IN $claimant_names))
   OR (n:Policy AND (n.policy_number IN $policy_numbers OR n.policy_id IN $policy_ids))
   OR (n:Adjuster AND n.adjuster_id IN $adjuster_ids)
   OR (n:Payment AND n.payment_id IN $payment_ids)
   OR (n:Cluster AND n.cluster_id IN $cluster_ids)
   OR (n:LossCause AND n.cause_code IN $cause_codes)
   OR (n:ClaimEvent AND n.event_id IN $event_ids)
WITH n LIMIT $max_seeds
WITH collect(n) AS seeds
UNWIND seeds AS s
CALL {
  WITH s
  OPTIONAL MATCH (s)-[r]-(m)
  WHERE type(r) IN $rel_types
  RETURN r, m LIMIT $per_seed
}
RETURN s, r, m
"""


def _node_dict(n) -> Dict[str, Any]:
    return {"id": n.element_id, "labels": list(n.labels), "props": dict(n)}


def _rel_dict(r) -> Dict[str, Any]:
    return {"id": r.element_id, "type": r.type, "start": r.start_node.element_id,
            "end": r.end_node.element_id, "props": dict(r)}


def fetch_from_neo4j(cfg, seeds: Dict[str, List[Any]], rel_types: List[str],
                     path_query: Optional[str] = None) -> Dict[str, Any]:
    """Return {"nodes": [...], "rels": [...], "seed_ids": [...], "paths": [[node ids]], "mode": str}."""
    from neo4j import GraphDatabase
    from neo4j.graph import Path

    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    try:
        with driver.session(database=cfg.neo4j_database) as session:
            if path_query:
                result = session.run(path_query)
                records = list(result)
                graph = result.graph()
                paths = []
                for rec in records:
                    for v in rec.values():
                        if isinstance(v, Path):
                            paths.append([n.element_id for n in v.nodes])
                nodes = [_node_dict(n) for n in graph.nodes]
                rels = [_rel_dict(r) for r in graph.relationships]
                if nodes:
                    return {"nodes": nodes, "rels": rels, "seed_ids": [n["id"] for n in nodes],
                            "paths": paths, "mode": "path"}
            params = {k: seeds.get(k, []) for k in (
                "claim_ids", "claimant_ids", "claimant_names", "policy_numbers", "policy_ids",
                "adjuster_ids", "payment_ids", "cluster_ids", "cause_codes", "event_ids")}
            params.update(rel_types=rel_types, per_seed=PER_SEED, max_seeds=30)
            result = session.run(_SEED_QUERY, **params)
            seed_ids: List[str] = []
            for rec in result:
                s = rec["s"]
                if s is not None and s.element_id not in seed_ids:
                    seed_ids.append(s.element_id)
            graph = result.graph()
            return {"nodes": [_node_dict(n) for n in graph.nodes],
                    "rels": [_rel_dict(r) for r in graph.relationships],
                    "seed_ids": seed_ids, "paths": [], "mode": "seeds"}
    finally:
        driver.close()


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_value(label: str, key: str, v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    codes = _STATUS.get((label, key))
    if codes and str(v) in codes:
        return f"{v} ({codes[str(v)]})"
    if isinstance(v, (int, float)) and any(m in key for m in _MONEY):
        return f"${v:,.2f}"
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _prop_rows(label: str, props: Dict[str, Any]) -> List[List[str]]:
    order = _PROP_ORDER.get(label, [])
    keys = [k for k in order if k in props] + [k for k in props if k not in order and k not in _HIDDEN_PROPS]
    return [[k, _fmt_value(label, k, props[k])] for k in keys if k not in _HIDDEN_PROPS]


def caption(labels: List[str], p: Dict[str, Any]) -> str:
    lab = labels[0] if labels else "Node"
    try:
        if lab == "Claim":
            return f"Claim {p.get('claim_id')}"
        if lab == "Claimant":
            return str(p.get("name") or f"Claimant {p.get('claimant_id')}")
        if lab == "Policy":
            return str(p.get("policy_number") or f"Policy {p.get('policy_id')}")
        if lab == "Payment":
            return f"Payment {p.get('payment_id')}"
        if lab == "Adjuster":
            return f"Adjuster {p.get('adjuster_id')}"
        if lab == "LossCause":
            return str(p.get("cause_code"))
        if lab == "Cluster":
            return f"Cluster {p.get('cluster_id')}"
        if lab == "ClaimEvent":
            return f"{p.get('seq')}. {str(p.get('type', '')).replace('_', ' ').title()}"
    except Exception:
        pass
    return lab


def _who(n: Dict[str, Any]) -> str:
    return caption(n["labels"], n["props"])


def edge_sentence(rel: Dict[str, Any], a: Dict[str, Any], b: Dict[str, Any]) -> str:
    t, p = rel["type"], rel["props"]
    A, B = _who(a), _who(b)
    if t == "FILED":
        return f"{A} filed {B}."
    if t == "COVERED_BY":
        return f"{A} is covered by policy {B}."
    if t == "HAS_PAYMENT":
        return f"{A} has payment {b['props'].get('payment_id')} ({_fmt_value('Payment', 'gross_amount', b['props'].get('gross_amount'))} to {b['props'].get('payee_name', '?')})."
    if t == "HANDLED_BY":
        return f"{A} is handled by {B}."
    if t == "SAME_ADJUSTER_AS":
        return f"{A} and {B} are handled by the same adjuster ({a['props'].get('adjuster_id')})."
    if t == "SAME_AGENT_AS":
        return f"{A} and {B} were sold by the same agent ({a['props'].get('agent_id')})."
    if t == "CAUSED_BY":
        return f"{A} was caused by {B}: {b['props'].get('description', '')}."
    if t == "IN_CLUSTER":
        return f"{A} belongs to {B} ({b['props'].get('cluster_type')}, risk {b['props'].get('risk_level')})" + \
               (" — it is the anchor claim." if p.get("is_anchor") else ".")
    if t == "OVERLAPS_WITH":
        return (f"{A} and {B} overlap ({p.get('overlap_type')}): losses {p.get('days_apart')} day(s) apart, "
                f"score {p.get('score')}{' — flagged as a collision' if p.get('collision') else ''}.")
    if t == "SIMILAR_TO":
        return f"{B} is similar to {A} (rank {p.get('rank')}, score {p.get('score')}): {p.get('basis', '')}."
    if t == "HAS_EVENT":
        return f"{A} has timeline event {B}: {b['props'].get('note', '')}"
    if t == "NEXT":
        return f"After {A} came {B}."
    return f"{A} —{t}→ {B}."


# ─────────────────────────────────────────────────────────────────────────────
# Pattern view
# ─────────────────────────────────────────────────────────────────────────────
def _build_pattern_view(pat: Pattern, evidence: Optional[Dict[str, Any]], question: str,
                        n_result_rows: Optional[int]) -> Dict[str, Any]:
    if not pat.nodes:
        return make_view("pattern", "Query pattern", [], [],
                         empty_msg="Could not read a node/relationship pattern from the Cypher.")
    ev_counts: Counter = Counter()
    ev_rel_counts: Counter = Counter()
    if evidence:
        for n in evidence["nodes"]:
            for lb in n["labels"]:
                ev_counts[lb] += 1
        for r in evidence["rels"]:
            ev_rel_counts[r["type"]] += 1

    order: List[str] = []
    for e in pat.edges:
        for k in (e.src, e.dst):
            if k not in order:
                order.append(k)
    for k in pat.nodes:
        if k not in order:
            order.append(k)
    number = {k: i + 1 for i, k in enumerate(order)}

    nodes, edges = [], []
    for k in order:
        pn = pat.nodes[k]
        lab = pn.label or "Node"
        colors = entity_colors(lab)
        title = f"{lab}" + (f" ({pn.var})" if pn.var else "")
        sections: List[Dict[str, Any]] = [
            {"title": "What this node is", "items": [LABEL_INFO.get(lab, "A node in the knowledge graph.")]},
            {"title": "Role in this query", "rows": [
                ["Variable", pn.var or "(anonymous — not referenced again)"],
                ["Traversal step", f"#{number[k]}"]]}]
        flt = ([f"inline: {pn.props}"] if pn.props else []) + pn.where
        if flt:
            sections.append({"title": "Filters on this node", "items": flt})
        if pn.returned:
            sections.append({"title": "Properties returned", "items": pn.returned})
        if ev_counts.get(lab):
            sections.append({"title": "Matched data", "rows": [[f"{lab} nodes retrieved", str(ev_counts[lab])]]})
        n = make_node(k, f"{circled(number[k])} {lab}" + (f"\n{pn.var}" if pn.var else ""), colors,
                      inspector(title, "Node pattern", f"Step {number[k]}", sections), size=36, kind="pattern-node")
        n["legendKey"] = lab
        nodes.append(n)

    for e in pat.edges:
        types = e.types or []
        tname = "|".join(types) if types else "(any relationship)"
        first = types[0] if types else ""
        hop = e.hops or ""
        hop_txt = ""
        if hop:
            m = re.fullmatch(r"\*(\d*)(?:\.\.(\d*))?", hop)
            lo, hi = (m.group(1), m.group(2)) if m else ("", "")
            hop_txt = (f"variable length: {lo or '1'} to {hi or 'any number of'} hops" if ".." in hop
                       else f"exactly {lo} hops" if lo else "variable length: any number of hops")
        src, dst = pat.nodes[e.src], pat.nodes[e.dst]
        sentence = REL_INFO.get(first, "Traverses relationships of this kind." if types else
                                "Follows ANY relationship type, so the database discovers the connecting nodes itself.")
        sections = [
            {"title": "What this relationship means", "items": [sentence]},
            {"title": "How the query walks it", "rows": [
                ["From", f"{src.label or '?'}" + (f" ({src.var})" if src.var else "")],
                ["To", f"{dst.label or '?'}" + (f" ({dst.var})" if dst.var else "")],
                ["Direction", "directed →" if e.directed else "either direction"],
            ] + ([["Hops", hop_txt]] if hop_txt else [])},
        ]
        if ev_rel_counts.get(first):
            sections.append({"title": "Matched data", "rows": [["Relationships retrieved", str(ev_rel_counts[first])]]})
        label = (":" + tname if types else "") + (f" {hop}" if hop else "")
        edges.append(make_edge(
            f"pe{e.idx}", e.src, e.dst, label.strip() or "(any)",
            inspector(f":{tname}", f"{src.label or '?'} → {dst.label or '?'}", "Relationship", sections),
            color="#7c3aed", width=3 if hop else 2.4, dashed=bool(hop), arrows=e.directed,
            note=sentence, curve=0.18 * (e.idx % 3), kind="pattern-edge"))

    # summary + steps
    summary: List[List[str]] = []
    if question:
        summary.append(["Question", question])
    summary.append(["Node labels", ", ".join(dict.fromkeys(p.label for p in pat.nodes.values() if p.label))])
    rels_used = list(dict.fromkeys(t for e in pat.edges for t in (e.types or ["(any)"])))
    if rels_used:
        summary.append(["Relationships", ", ".join(rels_used)])
    if pat.shortest:
        summary.append(["Traversal", "shortest path between two entities"])
    elif any(e.hops for e in pat.edges):
        summary.append(["Traversal", "variable-length (unknown depth)"])
    if pat.aggregates:
        summary.append(["Aggregation", "; ".join(pat.aggregates)])
    if pat.order_by:
        summary.append(["Order by", pat.order_by])
    if pat.limit:
        summary.append(["Limit", pat.limit])
    if n_result_rows is not None:
        summary.append(["Rows returned", f"{n_result_rows:,}"])
    steps: List[Dict[str, Any]] = []
    first_key = order[0]
    fn = pat.nodes[first_key]
    steps.append({"text": f"Start at {fn.label or 'a node'}" + (f" ({fn.var})" if fn.var else "")
                          + (f" where {fn.props}" if fn.props else ""), "nodes": [first_key], "edges": []})
    for e in pat.edges:
        src, dst = pat.nodes[e.src], pat.nodes[e.dst]
        tname = "|".join(e.types) if e.types else "any relationship"
        steps.append({"text": f"Follow :{tname}{(' ' + e.hops) if e.hops else ''} from {src.label or '?'} to {dst.label or '?'}"
                              f" — {REL_INFO.get(e.types[0], '') if e.types else 'the database finds the connecting path'}",
                      "nodes": [e.src, e.dst], "edges": [f"pe{e.idx}"]})
    for k in order:
        for c in pat.nodes[k].where:
            steps.append({"text": f"Filter {pat.nodes[k].label}: {c}", "nodes": [k], "edges": []})
    for c in pat.cross_where:
        steps.append({"text": f"Compare across nodes: {c}", "nodes": [], "edges": []})
    if pat.aggregates:
        steps.append({"text": "Aggregate: " + "; ".join(pat.aggregates), "nodes": list(order), "edges": []})
    steps.append({"text": "Return: " + ", ".join(pat.returns[:6]) + (" …" if len(pat.returns) > 6 else "")
                          + (f"  (limit {pat.limit})" if pat.limit else ""),
                  "nodes": list(order), "edges": [f"pe{e.idx}" for e in pat.edges]})
    legend = []
    seen = set()
    for k in order:
        lab = pat.nodes[k].label
        if lab and lab not in seen:
            seen.add(lab)
            legend.append((lab, entity_colors(lab)))
    return make_view(
        "pattern", "Query pattern", nodes, edges,
        description="The node labels and relationship types the generated Cypher walks, in order. "
                    "Dashed relationship = variable-length (unknown depth) traversal.",
        summary=summary, steps=steps, legend=legend,
        stats=f"{len(nodes)} node pattern(s) · {len(edges)} relationship pattern(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Matched graph view
# ─────────────────────────────────────────────────────────────────────────────
def _rows_fallback(df, n_rows: int) -> Dict[str, Any]:
    nodes, edges = [], []
    if df is None or df.empty:
        return make_view("matched", "Matched graph", [], [], empty_msg="The query returned no rows, so there is nothing to trace.")
    sample = df.head(15).astype(object)
    nodes.append(make_node("hub", f"Result set\n{n_rows:,} row(s)", ("#0f172a", "#0f172a"),
                           inspector("Result set", "No graph entities could be identified in the result", "Result",
                                     [{"title": "Columns", "items": [str(c) for c in df.columns]}]), size=34))
    for i, (_, row) in enumerate(sample.iterrows()):
        first = next((str(v) for v in row.values if isinstance(v, str)), f"row {i + 1}")
        nodes.append(make_node(f"r{i}", first if len(first) < 20 else first[:19] + "…", ("#cbd5e1", "#475569"),
                               inspector(f"Result row {i + 1}", "", "Row",
                                         [{"title": "Values", "rows": [[str(c), str(v)] for c, v in row.items()]}]),
                               size=20, small_font=True, in_result=True))
        edges.append(make_edge(f"re{i}", "hub", f"r{i}", "", inspector("row of result", ""), color="#94a3b8",
                               width=1, arrows=False))
    return make_view("matched", "Matched graph", nodes, edges,
                     description="The result rows do not carry graph identifiers (ids / names), so each row is shown as a node.",
                     steps=[{"text": f"{n_rows:,} row(s) come out of the traversal", "nodes": ["hub"], "edges": []}],
                     legend=[("Result row", ("#cbd5e1", "#475569"))],
                     stats=f"{min(15, n_rows)} of {n_rows:,} rows shown")


def _build_matched_view(pat: Pattern, ev: Optional[Dict[str, Any]], df, error: str = "") -> Dict[str, Any]:
    n_rows = 0 if df is None else len(df)
    if ev is None:
        msg = ("Could not load the matched graph from Neo4j" + (f": {error}" if error else ".")
               if error else "No entities could be identified from the result rows to look up in the graph.")
        if not error and df is not None and not df.empty:
            return _rows_fallback(df, n_rows)
        return make_view("matched", "Matched graph", [], [], empty_msg=msg)
    nodes_raw = {n["id"]: n for n in ev["nodes"]}
    if not nodes_raw:
        return make_view("matched", "Matched graph", [], [], empty_msg="No matching nodes were found in the graph.")
    seed_ids = set(ev.get("seed_ids") or [])
    rels = ev["rels"]
    # cap nodes, keeping seeds first
    keep = list(dict.fromkeys(list(seed_ids) + [i for i in nodes_raw if i not in seed_ids]))[:MAX_NODES]
    keep_set = set(keep)
    rels = [r for r in rels if r["start"] in keep_set and r["end"] in keep_set]
    # de-duplicate undirected duplicates of the same edge id
    rels = list({r["id"]: r for r in rels}.values())

    degree: Counter = Counter()
    for r in rels:
        degree[r["start"]] += 1
        degree[r["end"]] += 1
    seed_neighbours: Dict[str, List[str]] = defaultdict(list)
    for r in rels:
        if r["start"] in seed_ids and r["end"] not in seed_ids:
            seed_neighbours[r["end"]].append(f"{r['type']} from {_who(nodes_raw[r['start']])}")
        if r["end"] in seed_ids and r["start"] not in seed_ids:
            seed_neighbours[r["start"]].append(f"{r['type']} to {_who(nodes_raw[r['end']])}")

    nodes, edges = [], []
    for nid in keep:
        n = nodes_raw[nid]
        lab = n["labels"][0] if n["labels"] else "Node"
        colors = entity_colors(lab)
        in_res = nid in seed_ids
        sections: List[Dict[str, Any]] = [
            {"title": "Why it is in this graph", "items": [
                "It is part of the query result (dark ring)." if in_res else
                "Connection context: " + "; ".join(seed_neighbours.get(nid, ["linked to a result node"])[:3])]},
            {"title": "Properties", "rows": _prop_rows(lab, n["props"])},
            {"title": "What this node is", "items": [LABEL_INFO.get(lab, "")]}]
        node = make_node(nid, caption(n["labels"], n["props"]), colors,
                         inspector(caption(n["labels"], n["props"]), f":{':'.join(n['labels'])}",
                                   "In result" if in_res else "Context", sections),
                         size=24 if in_res else 17, in_result=in_res, small_font=True, kind="entity")
        node["legendKey"] = lab
        nodes.append(node)
    for i, r in enumerate(rels):
        a, b = nodes_raw[r["start"]], nodes_raw[r["end"]]
        sent = edge_sentence(r, a, b)
        sections = [{"title": "What this link says", "items": [sent]},
                    {"title": "Relationship type", "items": [f":{r['type']} — {REL_INFO.get(r['type'], '')}"]}]
        if r["props"]:
            sections.append({"title": "Relationship properties", "rows": [[k, _fmt_value("", k, v)] for k, v in r["props"].items()]})
        sections.append({"title": "From → To", "rows": [["From", _who(a)], ["To", _who(b)]]})
        dashed = r["type"] in ("SIMILAR_TO", "OVERLAPS_WITH", "SAME_ADJUSTER_AS", "SAME_AGENT_AS")
        edges.append(make_edge(f"re{i}", r["start"], r["end"], f":{r['type']}",
                               inspector(f":{r['type']}", f"{_who(a)} → {_who(b)}", "Relationship", sections),
                               color="#7c3aed" if not dashed else "#db2777", width=1.8, dashed=dashed, note=sent,
                               kind=r["type"]))

    # steps
    steps: List[Dict[str, Any]] = []
    paths = ev.get("paths") or []
    if paths:
        for pi, p in enumerate(paths[:3]):
            ids = [x for x in p if x in keep_set]
            hop_edges = []
            for a, b in zip(p, p[1:]):
                for j, r in enumerate(rels):
                    if {r["start"], r["end"]} == {a, b}:
                        hop_edges.append(f"re{j}")
                        break
            names = " → ".join(_who(nodes_raw[x]) for x in ids if x in nodes_raw)
            steps.append({"text": f"Path {pi + 1} ({max(0, len(p) - 1)} hop(s)): {names}", "nodes": ids, "edges": hop_edges})
    else:
        by_label: Dict[str, List[str]] = defaultdict(list)
        for nid in keep:
            if nid in seed_ids:
                by_label[nodes_raw[nid]["labels"][0]].append(nid)
        for lab, ids in by_label.items():
            steps.append({"text": f"{len(ids)} {lab} node(s) matched by the query", "nodes": ids, "edges": []})
        by_type: Dict[str, List[str]] = defaultdict(list)
        for j, r in enumerate(rels):
            by_type[r["type"]].append(f"re{j}")
        for t, eids in by_type.items():
            touched = sorted({x for j, r in enumerate(rels) if f"re{j}" in eids for x in (r["start"], r["end"])})
            steps.append({"text": f"{len(eids)} :{t} link(s) — {REL_INFO.get(t, '')}", "nodes": touched, "edges": eids})
    legend = []
    seen = set()
    for nid in keep:
        lab = nodes_raw[nid]["labels"][0] if nodes_raw[nid]["labels"] else "Node"
        if lab not in seen:
            seen.add(lab)
            legend.append((lab, entity_colors(lab)))
    summary = [["Nodes in graph", str(len(nodes))], ["In result", str(len([n for n in keep if n in seed_ids]))],
               ["Relationships", str(len(edges))]]
    if len(nodes_raw) > MAX_NODES:
        summary.append(["Note", f"Showing the first {MAX_NODES} of {len(nodes_raw)} nodes."])
    return make_view(
        "matched", "Matched graph", nodes, edges,
        description=("The path(s) found in the graph." if paths else
                     "Real nodes named in the result plus their direct connections. Dark ring = in the result; "
                     "smaller nodes are connection context. Dashed pink = derived analytics links."),
        summary=summary, steps=steps, legend=legend,
        stats=f"{len(nodes)} nodes · {len(edges)} relationships · {ev.get('mode', '')}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
Fetcher = Callable[[Dict[str, List[Any]], List[str], Optional[str]], Optional[Dict[str, Any]]]


def build_cypher_model(cypher: str, df=None, question: str = "", fetch: Optional[Fetcher] = None) -> Dict[str, Any]:
    """fetch(seeds, rel_types, path_query) -> evidence dict (see fetch_from_neo4j) or None."""
    pat = parse_cypher(cypher)
    seeds = extract_seeds(df)
    used_rels = {t for e in pat.edges for t in e.types}
    if re.search(r"\bfind_claim_clusters\b|Deterministic connected-components", cypher or ""):
        used_rels |= {"HANDLED_BY", "COVERED_BY"}
    rel_types = sorted(set(CORE_CONTEXT_RELS) | used_rels)
    path_q = path_evidence_query(cypher, pat)

    ev, err = None, ""
    if fetch is not None and (path_q or _has_seeds(seeds)):
        try:
            ev = fetch(seeds, rel_types, path_q)
        except Exception as e:                       # Neo4j down, permissions, etc.
            err = str(e).splitlines()[0][:200] if str(e) else e.__class__.__name__
    n_rows = None if df is None else len(df)
    views = [_build_pattern_view(pat, ev, question, n_rows)]
    try:
        views.append(_build_matched_view(pat, ev, df, err))
    except Exception as e:
        views.append(make_view("matched", "Matched graph", [], [], empty_msg=f"Could not build matched view: {e}"))
    return {"views": views}
