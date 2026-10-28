"""The reset tool must NEVER wipe the working demo's graph.  Run: python tests/test_neo4j_guard.py  (needs: pip install falkordblite)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from falkor_session import FalkorSession                     # noqa: E402
from tools.neo4j_reset_for_gl import inspect_graph, wipe_all  # noqa: E402
from utils.safety import ProductionGuardError, assert_neo4j_is_dev_or_empty  # noqa: E402

passed = 0
def check(name, cond):
    global passed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"); passed += bool(cond)

s = FalkorSession(path="/tmp/guard1.db", graph="g1")
check("empty graph -> EMPTY", inspect_graph(s)["verdict"] == "EMPTY")

s = FalkorSession(path="/tmp/guard2.db", graph="g2")
s.run("CREATE (:Claim {claim_id:'C1'}), (:Claimant {claimant_id:'X'}), (:GraphMeta {name:'environment', value:'dev'})")
rep = inspect_graph(s)
check("graph stamped 'dev' -> DEV_DEMO", rep["verdict"] == "DEV_DEMO")
try:
    wipe_all(s); refused = False
except RuntimeError:
    refused = True
check("wipe REFUSES the working demo's graph", refused)
check("...and nothing was deleted", inspect_graph(s)["total"] == 3)
try:
    assert_neo4j_is_dev_or_empty(s, ""); guard_ok = False
except ProductionGuardError:
    guard_ok = True
check("GL ingest guard also refuses the 'dev' graph", guard_ok)

s = FalkorSession(path="/tmp/guard3.db", graph="g3")
s.run("UNWIND range(1,12000) AS i CREATE (:Claim {claim_id: i})")
check("unmarked data -> UNMARKED", inspect_graph(s)["verdict"] == "UNMARKED")
try:
    assert_neo4j_is_dev_or_empty(s, ""); g = False
except ProductionGuardError:
    g = True
check("GL ingest guard refuses unmarked data until it is reset", g)
out = wipe_all(s)
after = inspect_graph(s)
check("wipe removes all 12000 nodes in batches", out["nodes_removed"] == 12000 and after["total"] == 1)
check("...then stamps the GL marker -> verdict GL", after["verdict"] == "GL")
assert_neo4j_is_dev_or_empty(s, ""); check("GL ingest guard now allows it", True)
print(f"\n{passed}/10 passed"); sys.exit(0 if passed == 10 else 1)
