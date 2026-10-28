"""
kg/graph_ingest_gl.py — ontology-driven ingest of the Commercial GL business graph.

Flow (nothing is hard-coded per label):

    Delta / SQLite tables ──(gl_mapping.yaml)──> in-memory GraphModel ──(gl_claims.yaml)──> validate ──> Neo4j
                                                                                     │
                                          :OntClass / :ONT_REL metadata written alongside, so the prompt
                                          and validators can be built FROM the graph.

What "the ontology is real" means here:
  * the graph can only contain classes, properties and relationships the ontology declares
  * relationship domain/range is checked (Payment -PAYEE-> Policy is rejected; Payment -PAYEE-> Attorney is allowed)
  * a change to either YAML changes the graph version, which triggers a rebuild
  * every node/edge is derivable from Delta tables alone, and reconcile() proves the graph still matches them

Public API (same names the pipeline already calls): graph_is_built(cfg), ensure_graph_ingested(cfg, db_path=None)
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kg.ontology import Ontology, load_ontology, write_ontology_metadata

logger = logging.getLogger(__name__)
BATCH = 1000


class OntologyViolation(Exception):
    """The data does not conform to the ontology (undeclared edge, dangling reference, duplicate id...)."""

    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__(f"{len(problems)} ontology violation(s): " + "; ".join(problems[:8]) + (" ..." if len(problems) > 8 else ""))


@dataclass
class GraphModel:
    nodes: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)                 # class -> id -> props
    edges: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = field(default_factory=dict)    # (rel, from_cls, to_cls) -> [(from_id, to_id)]
    problems: List[str] = field(default_factory=list)
    source_label: str = ""

    def node_counts(self) -> Dict[str, int]:
        return {c: len(v) for c, v in self.nodes.items()}

    def edge_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for (rel, f, t), pairs in self.edges.items():
            out[f"{f}-[{rel}]->{t}"] = len(pairs)
        return out

    def id_checksum(self, cls: str) -> str:
        return hashlib.sha1("|".join(sorted(self.nodes.get(cls, {}))).encode()).hexdigest()[:10]


# ------------------------------------------------------------------------------------------------ type casting
def _cast(v: Any, t: Optional[str]) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    if t == "float":
        return float(v)
    if t == "int":
        return int(float(v))
    if t == "bool":
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "t", "yes", "y")
        return bool(v)
    if t == "str":
        return str(v)
    return v


# ------------------------------------------------------------------------------------------------ build + validate
def build_model(onto: Ontology, src) -> GraphModel:
    """Read every mapped table once and assemble nodes + edges. Reports problems; does not raise."""
    m = GraphModel(source_label=getattr(src, "label", ""))
    cache: Dict[str, List[Dict[str, Any]]] = {}

    def rows(table: str) -> List[Dict[str, Any]]:
        if table not in cache:
            cache[table] = src.rows(table)
        return cache[table]

    id_owner: Dict[str, str] = {}
    for cls, nm in onto.mapping["nodes"].items():
        tbl, idc = nm["table"], nm["id"]
        data = rows(tbl)
        if data:
            missing = [c for c in [idc, *nm.get("props", {}).values()] if c not in data[0]]
            if missing:
                m.problems.append(f"table {tbl} (class {cls}) is missing column(s) {missing}")
                continue
        bucket: Dict[str, Dict[str, Any]] = m.nodes.setdefault(cls, {})
        for r in data:
            nid = r.get(idc)
            if nid is None:
                m.problems.append(f"{tbl}: row with null id for class {cls}")
                continue
            nid = str(nid)
            fam = (onto.ancestors(cls) or [cls])[-1]            # ids are unique across a whole class family (e.g. every Party role)
            if (fam, nid) in id_owner:
                other = id_owner[(fam, nid)]
                m.problems.append(f"duplicate id {nid} in class {cls}" if other == cls else f"id {nid} is used by two {fam} roles ({other} and {cls})")
                continue
            id_owner[(fam, nid)] = cls
            props = {}
            for p, col in nm.get("props", {}).items():
                val = _cast(r.get(col), (nm.get("cast") or {}).get(p))
                if val is not None:
                    props[p] = val
            bucket[nid] = props

    for rm in onto.mapping["relationships"]:
        rel, fcls, tcls, tbl = rm["rel"], rm["from"], rm["to"], rm["table"]
        data = rows(tbl)
        if data and (rm["from_id"] not in data[0] or rm["to_id"] not in data[0]):
            m.problems.append(f"table {tbl} lacks column {rm['from_id']} or {rm['to_id']} for {rel}")
            continue
        targets = [tcls] if not onto.classes[tcls].get("abstract") else [c for c in onto.concrete() if onto.is_a(c, tcls)]
        seen = set()
        for r in data:
            f, t = r.get(rm["from_id"]), r.get(rm["to_id"])
            if f is None or t is None:
                continue
            f, t = str(f), str(t)
            if f not in m.nodes.get(fcls, {}):
                m.problems.append(f"dangling {rel}: {fcls} {f} (table {tbl}) does not exist")
                continue
            actual = next((c for c in targets if t in m.nodes.get(c, {})), None)
            if actual is None:
                m.problems.append(f"dangling {rel}: {tcls} {t} (table {tbl}, from {f}) does not exist")
                continue
            if not onto.edge_allowed(rel, fcls, actual):
                m.problems.append(f"edge {fcls}-[{rel}]->{actual} is not allowed by the ontology")
                continue
            if (f, t) in seen:
                continue
            seen.add((f, t))
            m.edges.setdefault((rel, fcls, actual), []).append((f, t))
    return m


def validate_model(onto: Ontology, model: GraphModel) -> List[str]:
    probs = list(model.problems)
    for cls, nodes in model.nodes.items():
        if cls not in onto.classes:
            probs.append(f"class {cls} is not in the ontology")
        elif onto.classes[cls].get("abstract") and nodes:
            probs.append(f"abstract class {cls} has {len(nodes)} direct instances")
        declared = set(onto.props(cls)) if cls in onto.classes else set()
        for nid, props in nodes.items():
            bad = set(props) - declared
            if bad:
                probs.append(f"{cls} {nid}: undeclared properties {sorted(bad)}")
                break
        for key, vals in onto.enums.items():
            c, p = key.split(".")
            if c != cls:
                continue
            offenders = {n: pr[p] for n, pr in nodes.items() if p in pr and pr[p] not in vals}
            if offenders:
                ex = next(iter(offenders.items()))
                probs.append(f"{key}: {len(offenders)} value(s) outside the enum {vals}, e.g. {ex[0]}={ex[1]!r}")
    return probs


# ------------------------------------------------------------------------------------------------ write
def _batches(seq: List[Any], n: int = BATCH) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def write_model(session, onto: Ontology, model: GraphModel, *, wipe: bool = True, constraints: bool = True) -> Dict[str, int]:
    if constraints:
        for c in onto.classes:
            session.run(f"CREATE CONSTRAINT {c.lower()}_id IF NOT EXISTS FOR (n:{c}) REQUIRE n.id IS UNIQUE")
    if wipe:
        owned = list(onto.classes)
        session.run("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) DETACH DELETE n", labels=owned)
    n_nodes = n_edges = 0
    for cls, nodes in model.nodes.items():
        lab = ":".join(onto.labels(cls))
        rows = [dict(id=i, props=p) for i, p in nodes.items()]
        for b in _batches(rows):
            session.run(f"UNWIND $rows AS r MERGE (n:{lab} {{id: r.id}}) SET n += r.props", rows=b)
        n_nodes += len(rows)
    for (rel, fcls, tcls), pairs in model.edges.items():
        rows = [dict(f=f, t=t) for f, t in pairs]
        for b in _batches(rows):
            session.run(f"UNWIND $rows AS r MATCH (a:{fcls} {{id: r.f}}) MATCH (b:{tcls} {{id: r.t}}) MERGE (a)-[:{rel}]->(b)", rows=b)
        n_edges += len(rows)
    return {"nodes": n_nodes, "edges": n_edges}


# ------------------------------------------------------------------------------------------------ reconcile
def reconcile(onto: Ontology, model: GraphModel, session) -> Dict[str, Any]:
    """Compare what Delta says (model, freshly built from the tables) with what Neo4j actually holds."""
    checks: List[Dict[str, Any]] = []
    for cls in onto.concrete():
        want = model.node_counts().get(cls, 0)
        rec = list(session.run(f"MATCH (n:{cls}) RETURN count(n) AS c, collect(n.id) AS ids"))
        got, ids = (rec[0]["c"], rec[0]["ids"]) if rec else (0, [])
        want_ck = model.id_checksum(cls)
        got_ck = hashlib.sha1("|".join(sorted(ids)).encode()).hexdigest()[:10]
        checks.append(dict(kind="node", name=cls, delta=want, graph=got, ok=(want == got and want_ck == got_ck)))
    for (rel, f, t), pairs in model.edges.items():
        rec = list(session.run(f"MATCH (:{f})-[r:{rel}]->(:{t}) RETURN count(r) AS c"))
        got = rec[0]["c"] if rec else 0
        checks.append(dict(kind="edge", name=f"{f}-[{rel}]->{t}", delta=len(pairs), graph=got, ok=len(pairs) == got))
    bad = [c for c in checks if not c["ok"]]
    return dict(ok=not bad, checked=len(checks), drift=bad, ontology_version=onto.version, signature=onto.signature())


# ------------------------------------------------------------------------------------------------ orchestration
def ingest_into(session, onto: Ontology, src, *, wipe: bool = True, constraints: bool = True, guard=None, strict: bool = True) -> Dict[str, Any]:
    """Core: build -> validate -> (guard) -> write -> metadata -> reconcile. `session` is a Neo4j session (or compatible)."""
    t0 = time.time()
    map_errs = onto.validate_mapping()
    if map_errs:
        raise OntologyViolation(["mapping: " + e for e in map_errs])
    model = build_model(onto, src)
    problems = validate_model(onto, model)
    if problems and strict:
        raise OntologyViolation(problems)
    if guard:
        guard(session)
    stats = write_model(session, onto, model, wipe=wipe, constraints=constraints)
    write_ontology_metadata(session, onto)
    session.run("MERGE (m:GraphMeta {name: 'business_graph_version'}) SET m.value = $v", v=onto.graph_version())
    session.run("MERGE (m:GraphMeta {name: 'last_ingest'}) SET m.value = $v, m.source = $s, m.nodes = $n, m.edges = $e",
                v=time.strftime("%Y-%m-%dT%H:%M:%S"), s=model.source_label, n=stats["nodes"], e=stats["edges"])
    report = reconcile(onto, model, session)
    return dict(stats=stats, problems=problems, reconcile=report, seconds=round(time.time() - t0, 1), node_counts=model.node_counts(),
                edge_counts=model.edge_counts(), source=model.source_label)


def _driver(cfg):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))


def graph_is_built(cfg) -> bool:
    """True when Neo4j holds a graph built from THIS ontology+mapping version (any YAML change => rebuild)."""
    onto = load_ontology()
    driver = _driver(cfg)
    try:
        with driver.session(database=cfg.neo4j_database) as s:
            rec = s.run("MATCH (m:GraphMeta {name: 'business_graph_version'}) RETURN m.value AS v").single()
            n = s.run("MATCH (c:Claim) RETURN count(c) AS c").single()["c"]
            return bool(rec and rec["v"] == onto.graph_version() and n > 0)
    except Exception as e:  # noqa: BLE001
        logger.warning("graph_is_built() check failed: %s", e)
        return False
    finally:
        driver.close()


def ingest(cfg=None, db_path: str = None) -> Dict[str, Any]:
    from config import load_config
    from kg.graph_ingest import _make_source          # reads Delta when DATA_BACKEND=databricks, SQLite otherwise
    from utils.safety import assert_neo4j_is_dev_or_empty, stamp_dev_marker
    cfg = cfg or load_config()
    onto = load_ontology()
    src = _make_source(cfg, db_path)
    print(f"[graph_ingest_gl] ontology v{onto.version} ({onto.signature()}) — reading tables from {src.label}")
    driver = _driver(cfg)
    try:
        with driver.session(database=cfg.neo4j_database) as s:
            def guard(session):
                assert_neo4j_is_dev_or_empty(session, cfg.neo4j_uri, "rebuild the GL business graph in")
            out = ingest_into(s, onto, src, guard=guard)
            stamp_dev_marker(s)
        rep = out["reconcile"]
        print(f"[graph_ingest_gl] {out['stats']['nodes']} nodes, {out['stats']['edges']} edges in {out['seconds']}s | "
              f"reconcile: {'OK' if rep['ok'] else 'DRIFT'} ({rep['checked']} checks)")
        return out
    finally:
        src.close()
        driver.close()


def ensure_graph_ingested(cfg, db_path: str = None) -> bool:
    """Build the graph if it is missing or was built from a different ontology version. True if a build happened."""
    if graph_is_built(cfg):
        print("[graph_ingest_gl] GL graph already present for this ontology version — skipping ingest.")
        return False
    print("[graph_ingest_gl] GL graph missing or ontology changed — rebuilding from the configured data backend...")
    ingest(cfg, db_path)
    return True


def reconcile_live(cfg=None) -> Dict[str, Any]:
    """Drift check against the CURRENT source tables (e.g. after Delta changed): is the graph stale?"""
    from config import load_config
    from kg.graph_ingest import _make_source
    cfg = cfg or load_config()
    onto = load_ontology()
    src = _make_source(cfg)
    driver = _driver(cfg)
    try:
        model = build_model(onto, src)
        with driver.session(database=cfg.neo4j_database) as s:
            return reconcile(onto, model, s)
    finally:
        src.close()
        driver.close()


if __name__ == "__main__":
    import json
    print(json.dumps(ingest()["reconcile"], indent=2, default=str))
