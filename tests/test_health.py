"""The health check must tell the situations apart.  Run: python tests/test_health.py  (needs falkordblite)"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from falkor_session import FalkorSession            # noqa: E402
from kg.graph_ingest import _make_source            # noqa: E402
from kg.graph_ingest_gl import ingest_into          # noqa: E402
from kg.ontology import load_ontology               # noqa: E402
# The health page must NOT depend on other (possibly out-of-date) files: poison the module the old version imported.
sys.modules["tools.neo4j_reset_for_gl"] = None
from utils.health import diagnose, neo4j_health, registry_health  # noqa: E402

onto = load_ontology(); passed = 0; total = 0
def check(name, cond):
    global passed, total
    total += 1; passed += bool(cond); print(f"  {'PASS' if cond else 'FAIL'}  {name}")
def run(session, backend="databricks", sql=None):
    h = {"backend": backend, "neo4j": neo4j_health(session, onto), "neo4j_error": None, "sql": sql, "sql_error": None}
    return h, diagnose(h)
def has(findings, level, needle): return any(l == level and needle.lower() in m.lower() for l, m in findings)

# 1. empty instance
s = FalkorSession(path="/tmp/h1.db", graph="h1")
_, f = run(s); check("empty Neo4j -> 'EMPTY' error", has(f, "error", "EMPTY"))

# 2. OLD demo's graph (nodes without id, old schema graph)
s = FalkorSession(path="/tmp/h2.db", graph="h2")
s.run("CREATE (:Claim {claim_id:'1'}), (:Policy {policy_id:'P'}), (:Table {name:'CLAIMS'}), (:Table {name:'POLICY'})")
_, f = run(s)
check("old-demo graph -> 'NONE are GL nodes' error", has(f, "error", "none are gl nodes"))
check("old-demo schema graph is flagged (upper-case names)", has(f, "error", "upper-case") or has(f, "error", "OLD demo's schema graph"))

# 3. working demo's graph
s = FalkorSession(path="/tmp/h3.db", graph="h3")
s.run("CREATE (:Claim {claim_id:'1'}), (:GraphMeta {name:'environment', value:'dev'})")
_, f = run(s); check("'dev' marker -> working-demo error", has(f, "error", "WORKING demo"))

# 4. GL graph ingested but schema graph missing
s = FalkorSession(path="/tmp/h4.db", graph="h4")
ingest_into(s, onto, _make_source(SimpleNamespace(data_backend="sqlite", sqlite_db_path=str(ROOT / "database/gl_claims.db"))), wipe=False, constraints=False)
h, f = run(s)
check("GL graph present and current", has(f, "ok", "present and current"))
check("missing schema graph is flagged", has(f, "error", "schema graph") and has(f, "error", "MISSING"))
check("sanity check finds the planted attorney", has(f, "ok", "Hargrove"))

# 5. fully healthy
s.run("CREATE (:Table {name:'attorney'}), (:Table {name:'claim'})")
h, f = run(s, sql={"target": "graphql.insurance", "counts": {"claim": 144}, "error": None})
check("healthy -> all ok + hint about the generated query", all(l == "ok" for l, _ in f) and has(f, "ok", "Generated Cypher"))

# 6. upper-case schema graph is FINE (the registry normalises case) -> no false alarm
s.run("MATCH (t:Table) DELETE t"); s.run("CREATE (:Table {name:'ATTORNEY'}), (:Table {name:'CLAIM'})")
_, f = run(s, sql={"target": "graphql.insurance", "counts": {"claim": 144}, "error": None})
check("upper-case schema graph is NOT flagged", not any(l == "error" for l, _ in f))
_src = (ROOT / "utils" / "health.py").read_text()
check("health.py has no import of tools/ (the stale-file KeyError you hit cannot recur)", "from tools" not in _src and "import tools" not in _src)

# 7. what the RUNNING app's registry knows
class Reg:
    def __init__(self, names, concepts, aliases): self.table_names, self._c, self._custom_aliases = names, concepts, aliases
    def build_concept_map(self): return self._c
old = Reg(["CLAIMS", "POLICY", "PAYMENT", "CLAIMANT"], {"claim": [], "claims": [], "policy": []}, {})
new = Reg(["ATTORNEY", "CLAIM", "LITIGATION"], {"attorney": [], "attorneys": [], "suits": [], "open suit": [], "claim": [], "claims": []}, {"attorney": "ATTORNEY"})
h, f = run(s); h["registry"] = registry_health(old); f = diagnose(h)
check("stale registry (old 4 tables) -> 'redeploy / restart' error", has(f, "error", "RUNNING app") and has(f, "error", "restart"))
h["registry"] = registry_health(new); f = diagnose(h)
check("registry with GL tables + aliases -> ok", has(f, "ok", "including `attorney`"))
check("the attorney question is covered deterministically (no LLM needed)", registry_health(new)["uncovered_words"]["Which attorneys have more than 10 open suits?"] == [])
print(f"\n{passed}/{total} passed"); sys.exit(0 if passed == total else 1)