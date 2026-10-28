"""
tools/neo4j_reset_for_gl.py — look inside the Neo4j instance BEFORE the GL build touches it.

    python tools/neo4j_reset_for_gl.py                              # READ-ONLY: prints what is in the database + a verdict
    python tools/neo4j_reset_for_gl.py --wipe --confirm-host <host> # wipe the OLD demo's graph so the GL build can use it

Verdicts
    EMPTY      nothing there                         -> the GL build can use it
    GL         already stamped by the GL build        -> the GL build can use it (re-ingest replaces its own data)
    DEV_DEMO   stamped 'dev' = your WORKING demo     -> NEVER wiped by this tool, with or without flags. Use another instance.
    UNMARKED   has data but no marker                -> the older demo (or something unknown). Wipe needs --wipe AND
                                                        --confirm-host <the host part of NEO4J_URI> typed exactly.
A wipe deletes ALL nodes and relationships, drops every constraint and non-lookup index, then stamps the GL marker.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.safety import DEV_MARKER_NAME, DEV_MARKER_VALUE, _norm_uri, stamp_dev_marker  # noqa: E402


def inspect_graph(session) -> Dict[str, Any]:
    total = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    markers = [r["v"] for r in session.run("MATCH (m:GraphMeta {name: $n}) RETURN m.value AS v", n=DEV_MARKER_NAME)]
    rows = list(session.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c, count(n.id) AS with_id ORDER BY c DESC"))
    labels = {r["l"]: r["c"] for r in rows}
    labels_with_id = {r["l"]: r["with_id"] for r in rows}
    versions = [r["v"] for r in session.run("MATCH (m:GraphMeta {name: 'business_graph_version'}) RETURN m.value AS v")]
    if total == 0:
        verdict = "EMPTY"
    elif "dev" in markers:
        verdict = "DEV_DEMO"
    elif DEV_MARKER_VALUE in markers:
        verdict = "GL"
    else:
        verdict = "UNMARKED"
    return dict(total=total, markers=markers, labels=labels, labels_with_id=labels_with_id, business_graph_version=versions, verdict=verdict)


def print_report(rep: Dict[str, Any], uri: str = "") -> None:
    print(f"Neo4j: {_norm_uri(uri) or '(session)'}")
    print(f"  nodes: {rep['total']}   marker(s): {rep['markers'] or 'none'}   business graph version: {rep['business_graph_version'] or 'none'}")
    print("    (GL nodes carry an `id` property; old-demo nodes do not)")
    for l, c in list(rep["labels"].items())[:25]:
        print(f"    :{l:<20} {c:>6} nodes, {rep['labels_with_id'].get(l, 0):>6} with id")
    msg = {
        "EMPTY": "OK - empty. The GL build can use this instance.",
        "GL": "OK - already a GL-build graph. Re-running the ingest replaces it.",
        "DEV_DEMO": "STOP - this is your WORKING demo's graph (marker 'dev'). The GL build must use a different Neo4j instance. Nothing here will be touched.",
        "UNMARKED": "CAUTION - data without a marker. If this is the OLD demo's graph you may reset it with --wipe --confirm-host <host>.",
    }[rep["verdict"]]
    print(f"  VERDICT: {rep['verdict']}  — {msg}")


def wipe_all(session) -> Dict[str, int]:
    """Delete everything in batches, then drop constraints and non-lookup indexes, then stamp the GL marker."""
    rep = inspect_graph(session)
    if rep["verdict"] == "DEV_DEMO":
        raise RuntimeError("Refusing: this database carries the 'dev' marker (your working demo).")
    removed = 0
    while True:
        n = session.run("MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c").single()["c"]
        removed += n
        if n == 0:
            break
    dropped_c = dropped_i = 0
    try:
        for r in list(session.run("SHOW CONSTRAINTS YIELD name RETURN name")):
            session.run(f"DROP CONSTRAINT `{r['name']}` IF EXISTS")
            dropped_c += 1
        for r in list(session.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name")):
            session.run(f"DROP INDEX `{r['name']}` IF EXISTS")
            dropped_i += 1
    except Exception as e:  # noqa: BLE001 — e.g. SHOW not available; schema objects are recreated with IF NOT EXISTS anyway
        print(f"  (could not list/drop constraints and indexes: {str(e)[:100]})")
    stamp_dev_marker(session)
    return dict(nodes_removed=removed, constraints_dropped=dropped_c, indexes_dropped=dropped_i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true")
    ap.add_argument("--confirm-host", default="")
    args = ap.parse_args()
    from config import load_config
    from neo4j import GraphDatabase
    cfg = load_config()
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    with driver.session(database=cfg.neo4j_database) as s:
        rep = inspect_graph(s)
        print_report(rep, cfg.neo4j_uri)
        if args.wipe:
            if rep["verdict"] == "DEV_DEMO":
                sys.exit("Refusing to wipe: marked as the working demo's graph.")
            if rep["verdict"] == "UNMARKED" and args.confirm_host.strip().lower() != _norm_uri(cfg.neo4j_uri).split(":")[0]:
                sys.exit(f"UNMARKED graph: re-run with --confirm-host {_norm_uri(cfg.neo4j_uri).split(':')[0]}  (typed exactly) to confirm the wipe.")
            print("Wiping...", wipe_all(s))
    driver.close()


if __name__ == "__main__":
    main()