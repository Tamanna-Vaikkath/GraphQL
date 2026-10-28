"""
kg/ontology.py — the ontology as a RUNTIME artifact.

Loads ontology/gl_claims.yaml (what the business means) and ontology/gl_mapping.yaml (where it lives in Delta) and
exposes everything the rest of the app needs:

    onto = load_ontology()
    onto.labels("Attorney")              -> ["Attorney", "Party"]          (roles inherit from Party)
    onto.props("Attorney")               -> ["name", "firm", "barId"]      (inherited + own)
    onto.edge_allowed("PAYEE", "Payment", "Attorney")   -> True            (range Party covers subclasses)
    onto.validate_mapping()              -> [] or a list of problems
    onto.schema_block()                  -> text for the Cypher-generation prompt
    onto.allowed_labels / onto.allowed_rels              -> drive the Cypher validator
    onto.signature()                     -> hash; a change to either file triggers a graph rebuild

The same schema block can be rendered from the :OntClass / :ONT_REL metadata stored IN Neo4j
(schema_block_from_neo4j), so the prompt can be built from the graph itself, exactly as the ontology design intends.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
CAST_TYPES = {"float", "int", "bool", "str"}


def _clean(p: str) -> str:
    return p.rstrip("*")


class Ontology:
    def __init__(self, ont: Dict[str, Any], mapping: Dict[str, Any]):
        self.ont, self.mapping = ont, mapping
        self.version: str = str(ont["version"])
        self.classes: Dict[str, Dict[str, Any]] = ont["classes"]
        self.rels: List[Dict[str, str]] = ont["relationships"]
        self.cqs: List[Dict[str, Any]] = ont.get("competency_questions", [])
        self.enums: Dict[str, List[str]] = ont.get("enums", {})
        self.rules: List[str] = ont.get("query_rules", [])

    # ---- hierarchy -------------------------------------------------------------------------------------------
    def parent(self, cls: str) -> Optional[str]:
        return self.classes[cls].get("subClassOf")

    def ancestors(self, cls: str) -> List[str]:
        out, p = [], self.parent(cls)
        while p:
            out.append(p)
            p = self.parent(p)
        return out

    def labels(self, cls: str) -> List[str]:
        """Neo4j labels for an instance of cls: itself plus every ancestor (roles carry the Party label)."""
        return [cls] + self.ancestors(cls)

    def is_a(self, cls: str, other: str) -> bool:
        return cls == other or other in self.ancestors(cls)

    def concrete(self) -> List[str]:
        return [c for c, d in self.classes.items() if not d.get("abstract")]

    def subclasses(self, cls: str) -> List[str]:
        return [c for c in self.classes if self.parent(c) == cls]

    # ---- properties ------------------------------------------------------------------------------------------
    def props(self, cls: str) -> List[str]:
        inherited = self.props(self.parent(cls)) if self.parent(cls) else []
        return inherited + [_clean(p) for p in self.classes[cls].get("props", [])]

    def props_any(self, cls: str) -> set:
        """Properties valid on cls OR any of its subclasses (for variables typed with an abstract label such as :Party)."""
        out = set(self.props(cls)) | {"id"}
        for sub in self.subclasses(cls):
            out |= self.props_any(sub)
        return out

    def temporal_props(self, cls: str) -> List[str]:
        return [_clean(p) for p in self.classes[cls].get("props", []) if p.endswith("*")]

    # ---- relationships ---------------------------------------------------------------------------------------
    def edge_allowed(self, rel: str, from_cls: str, to_cls: str) -> bool:
        for r in self.rels:
            if r["rel"] == rel and self.is_a(from_cls, r["from"]) and self.is_a(to_cls, r["to"]):
                return True
        return False

    @property
    def allowed_labels(self) -> set:
        return set(self.classes)

    @property
    def allowed_rels(self) -> set:
        return {r["rel"] for r in self.rels}

    # ---- identity of this ontology+mapping (a change => the graph is rebuilt) ---------------------------------
    def signature(self) -> str:
        blob = json.dumps([self.ont, self.mapping], sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]

    def graph_version(self) -> str:
        return f"gl-{self.version}-{self.signature()}"

    # ---- mapping validation ----------------------------------------------------------------------------------
    def validate_mapping(self) -> List[str]:
        errs: List[str] = []
        nodes, rmaps = self.mapping.get("nodes", {}), self.mapping.get("relationships", [])
        for c in self.concrete():
            if c not in nodes:
                errs.append(f"class {c} is concrete but has no node mapping")
        for c, m in nodes.items():
            if c not in self.classes:
                errs.append(f"mapping declares unknown class {c}")
                continue
            if self.classes[c].get("abstract"):
                errs.append(f"abstract class {c} must not be mapped")
            if not m.get("table") or not m.get("id"):
                errs.append(f"{c}: mapping needs table and id")
            declared = set(self.props(c))
            for p in m.get("props", {}):
                if p not in declared:
                    errs.append(f"{c}.{p} is mapped but not declared in the ontology")
            for p, t in (m.get("cast") or {}).items():
                if t not in CAST_TYPES:
                    errs.append(f"{c}.{p}: unknown cast type {t}")
        seen = set()
        for r in rmaps:
            key = (r["rel"], r["from"], r["to"])
            if key in seen:
                errs.append(f"relationship mapping {key} appears twice")
            seen.add(key)
            if not any(d["rel"] == r["rel"] and d["from"] == r["from"] and d["to"] == r["to"] for d in self.rels):
                errs.append(f"mapped relationship {key} is not declared in the ontology")
        for d in self.rels:
            if (d["rel"], d["from"], d["to"]) not in seen:
                errs.append(f"ontology relationship {(d['rel'], d['from'], d['to'])} has no mapping")
        for key in self.enums:
            c, p = key.split(".")
            if c not in self.classes or p not in self.props(c):
                errs.append(f"enum {key} refers to an undeclared property")
        return errs

    # ---- records (shared by the YAML renderer and the Neo4j renderer) ----------------------------------------
    def records(self) -> Tuple[List[dict], List[dict]]:
        classes = [dict(name=c, parent=self.parent(c), abstract=bool(d.get("abstract")), domain=d["domain"], definition=d["def"],
                        acord=d.get("acord", ""), status=d.get("status", ""), props=self.props(c), temporal=self.temporal_props(c))
                   for c, d in self.classes.items()]
        rels = [dict(rel=r["rel"], frm=r["from"], to=r["to"]) for r in self.rels]
        return classes, rels

    def schema_block(self) -> str:
        classes, rels = self.records()
        return render_schema_block(self.version, classes, rels, self.rules, self.enums)


def render_schema_block(version: str, classes: List[dict], rels: List[dict], rules: Iterable[str], enums: Dict[str, List[str]]) -> str:
    out = [f"Knowledge Graph schema — Commercial GL Claims ontology v{version} (Neo4j property graph).", "",
           "Nodes (each also has a unique string property `id`):"]
    abstract = [c for c in classes if c["abstract"]]
    for c in classes:
        if c["abstract"]:
            continue
        labels = c["name"] + (f":{c['parent']}" if c["parent"] else "")
        props = ", ".join(c["props"])
        tmp = f"   // time-varying: {', '.join(c['temporal'])} (current value)" if c["temporal"] else ""
        out.append(f"  (:{labels} {{{props}}}){tmp}")
    for a in abstract:
        subs = [c["name"] for c in classes if c["parent"] == a["name"]]
        out.append(f"Abstract class {a['name']}: every {', '.join(subs)} node also carries the :{a['name']} label.")
    out += ["", "Relationships (direction matters):"]
    out += [f"  (:{r['frm']})-[:{r['rel']}]->(:{r['to']})" for r in rels]
    if enums:
        out += ["", "Allowed values:"]
        out += [f"  {k}: {' | '.join(v)}" for k, v in enums.items()]
    out += ["", "Ontology rules:"]
    out += [f"- {r}" for r in rules]
    out += ["- Use ONLY the labels, relationship types and properties listed above."]
    return "\n".join(out)


def load_ontology(root: Path = ROOT) -> Ontology:
    ont = yaml.safe_load((root / "ontology" / "gl_claims.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((root / "ontology" / "gl_mapping.yaml").read_text(encoding="utf-8"))
    return Ontology(ont, mapping)


# ---------------------------------------------------------------------------------------------------------------
# Ontology stored IN the graph (so prompts / validators can read it from Neo4j, not from a file)
# ---------------------------------------------------------------------------------------------------------------
def write_ontology_metadata(session, onto: Ontology) -> None:
    classes, rels = onto.records()
    session.run("MATCH (n) WHERE n:OntClass OR n:Ontology OR n:CompetencyQuestion OR n:OntRule OR n:OntEnum DETACH DELETE n")
    session.run("MERGE (o:Ontology {name: 'gl_claims'}) SET o.version = $v, o.signature = $s", v=onto.version, s=onto.signature())
    session.run("""UNWIND $rows AS c MERGE (k:OntClass {name: c.name})
                   SET k.parent = c.parent, k.abstract = c.abstract, k.domain = c.domain, k.definition = c.definition,
                       k.acord = c.acord, k.status = c.status, k.properties = c.props, k.temporal = c.temporal""", rows=classes)
    session.run("""UNWIND $rows AS c WITH c WHERE c.parent IS NOT NULL
                   MATCH (a:OntClass {name: c.name}), (b:OntClass {name: c.parent}) MERGE (a)-[:SUBCLASS_OF]->(b)""", rows=classes)
    session.run("""UNWIND $rows AS r MATCH (a:OntClass {name: r.frm}), (b:OntClass {name: r.to})
                   MERGE (a)-[:ONT_REL {name: r.rel}]->(b)""", rows=rels)
    session.run("UNWIND $rows AS r MERGE (x:OntRule {idx: r.idx}) SET x.text = r.text",
                rows=[dict(idx=i, text=t) for i, t in enumerate(onto.rules)])
    session.run("UNWIND $rows AS e MERGE (x:OntEnum {key: e.key}) SET x.vals = e.vals",
                rows=[dict(key=k, vals=v) for k, v in onto.enums.items()])
    session.run("""UNWIND $rows AS q MERGE (c:CompetencyQuestion {id: q.id}) SET c.text = q.q, c.capability = q.capability""", rows=onto.cqs)
    session.run("""UNWIND $rows AS q UNWIND q.touches AS t MATCH (c:CompetencyQuestion {id: q.id}), (k:OntClass {name: t})
                   MERGE (c)-[:TOUCHES]->(k)""", rows=onto.cqs)


def schema_block_from_session(session) -> Optional[str]:
    """Render the LLM schema block from the ontology metadata stored in Neo4j. None if no ontology is stored."""
    meta = list(session.run("MATCH (o:Ontology) RETURN o.version AS v"))
    if not meta:
        return None
    version = meta[0]["v"]
    crow = list(session.run("MATCH (k:OntClass) RETURN k.name AS name, k.parent AS parent, k.abstract AS abstract, k.domain AS domain, "
                            "k.definition AS definition, k.acord AS acord, k.status AS status, k.properties AS props, k.temporal AS temporal"))
    rrow = list(session.run("MATCH (a:OntClass)-[r:ONT_REL]->(b:OntClass) RETURN a.name AS frm, r.name AS rel, b.name AS to"))
    rules = [r["t"] for r in session.run("MATCH (x:OntRule) RETURN x.text AS t ORDER BY x.idx")]
    enums = {r["k"]: r["v"] for r in session.run("MATCH (x:OntEnum) RETURN x.key AS k, x.vals AS v")}
    # keep the declared order: the YAML order is the canonical order, graph order is not guaranteed
    onto = load_ontology()
    order = {n: i for i, n in enumerate(onto.classes)}
    rel_order = {(r["from"], r["rel"], r["to"]): i for i, r in enumerate(onto.rels)}
    classes = sorted([dict(c) for c in crow], key=lambda c: order.get(c["name"], 999))
    rels = sorted([dict(r) for r in rrow], key=lambda r: rel_order.get((r["frm"], r["rel"], r["to"]), 999))
    enum_order = {k: i for i, k in enumerate(onto.enums)}
    enums = dict(sorted(enums.items(), key=lambda kv: enum_order.get(kv[0], 999)))
    return render_schema_block(version, classes, rels, rules, enums)


def load_schema_block(cfg=None) -> Tuple[str, str]:
    """Best available schema block. Prefers the ontology stored in Neo4j (so the prompt is built FROM the graph),
    falls back to the YAML file. Returns (block, source)."""
    if cfg is not None:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
            try:
                with driver.session(database=cfg.neo4j_database) as s:
                    block = schema_block_from_session(s)
                    if block:
                        return block, "neo4j"
            finally:
                driver.close()
        except Exception:  # noqa: BLE001 — Neo4j unreachable / no ontology yet: fall back to the file
            pass
    return load_ontology().schema_block(), "yaml"


if __name__ == "__main__":
    o = load_ontology()
    problems = o.validate_mapping()
    print(f"ontology v{o.version} sig={o.signature()}  classes={len(o.classes)} concrete={len(o.concrete())} rels={len(o.rels)}")
    print("mapping problems:", problems or "none")
