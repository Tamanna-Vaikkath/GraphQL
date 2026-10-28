"""The Cypher validator must ACCEPT every tested query and REJECT queries that break the ontology.  Run: python tests/test_cypher_validator.py"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.cypher_generator import _validate_cypher, CypherValidationError  # noqa: E402

ok = bad = 0


def accepts(name, cy):
    global ok, bad
    try:
        _validate_cypher(cy); ok += 1; print(f"  PASS  accepts: {name}")
    except CypherValidationError as e:
        bad += 1; print(f"  FAIL  accepts: {name} -> {e}")


def rejects(name, cy, needle):
    global ok, bad
    try:
        _validate_cypher(cy); bad += 1; print(f"  FAIL  rejects: {name} (was accepted)")
    except CypherValidationError as e:
        if needle.lower() in str(e).lower():
            ok += 1; print(f"  PASS  rejects: {name}")
        else:
            bad += 1; print(f"  FAIL  rejects: {name} wrong reason -> {e}")


for i, e in enumerate(yaml.safe_load((ROOT / "ontology/fewshot.yaml").read_text())["examples"], 1):
    accepts(f"fewshot {i}", e["cypher"])
for t in yaml.safe_load((ROOT / "ontology/cq_tests.yaml").read_text())["tests"]:
    accepts(f"{t['id']} (params inlined)", re.sub(r"\$\w+", "'x'", t["cypher"]))

rejects("wrong direction", "MATCH (a:Attorney)-[:COUNSEL]->(l:Litigation) RETURN l.id", "does not allow")
rejects("wrong domain/range", "MATCH (c:Claim)-[:COUNSEL]->(a:Attorney) RETURN a.name", "does not allow")
rejects("undeclared property (dot)", "MATCH (a:Attorney) RETURN a.favouriteColour", "not declared")
rejects("undeclared property (inline)", "MATCH (c:Claim {severity: 'high'}) RETURN c.id", "not declared")
rejects("property from the wrong class", "MATCH (c:Claim) RETURN c.firm", "not declared")
rejects("unknown label", "MATCH (x:Vehicle) RETURN x.id", "unknown label")
rejects("unknown relationship", "MATCH (a:Claim)-[:OWNS]->(b:Policy) RETURN a.id", "unknown label")
rejects("write clause", "MATCH (c:Claim) SET c.status = 'x' RETURN c", "write")
rejects("query parameter", "MATCH (c:Claim {id: $id}) RETURN c.id", "parameter")

# things that must NOT be rejected
accepts("Party role property via abstract label", "MATCH (p:Payment)-[:PAYEE]->(x:Party) RETURN x.name")
accepts("undirected hop", "MATCH (a:Attorney)-[:COUNSEL]-(l:Litigation) RETURN a.id")
accepts("string literal containing a dot", "MATCH (c:Claim) WHERE c.summary CONTAINS 'a.b' RETURN c.id")
print(f"\n{ok}/{ok + bad} passed")
sys.exit(1 if bad else 0)
