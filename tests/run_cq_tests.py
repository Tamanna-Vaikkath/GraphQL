"""
tests/run_cq_tests.py — the competency questions as automated acceptance tests.

    python tests/run_cq_tests.py               # against the Neo4j in your .env / app config (graph must already be ingested)
    python tests/run_cq_tests.py --embedded    # no Neo4j needed: ingest into an embedded openCypher engine first (pip install falkordblite)
    python tests/run_cq_tests.py --fewshot     # also execute every few-shot example used in the NLQ prompt

Each CQ's Cypher result is compared with the SQL ground truth computed independently by data/generate_gl_data.py.
Exit code 0 only if everything passes, so this can gate a deployment.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

GT = json.loads((ROOT / "database" / "gl_ground_truth.json").read_text())


def _rows(session, cypher, params=None):
    return [dict(r) for r in session.run(cypher, **(params or {}))]


CHECKS = {
    "CQ1": lambda rows: (sorted(r["feature_id"] for r in rows) == GT["CQ1"],
                         f"{len(rows)} features, attorneys={sorted({r['attorney'] for r in rows})}, expected {len(GT['CQ1'])}"),
    "CQ2": lambda rows: (sorted(r["provider_id"] for r in rows) == GT["CQ2"], f"providers={[r['provider_id'] for r in rows]}, expected {GT['CQ2']}"),
    "CQ3": lambda rows: (sorted({r["claim_id"] for r in rows}) == GT["CQ3"] and
                         all(GT["CQ3_development"][r["feature_id"]] == [r["initial_reserve"], r["current_reserve"]] for r in rows if r["feature_id"] in GT["CQ3_development"]),
                         f"{len({r['claim_id'] for r in rows})} claims (expected {len(GT['CQ3'])}); hero development "
                         f"{next(((r['initial_reserve'], r['current_reserve']) for r in rows if r['feature_id'] == 'FEA-1001-BI'), None)}"),
    "CQ4a": lambda rows: (len(rows) == 1 and rows[0]["amount"] == GT["CQ4"]["known_2025-06-30"], f"{rows[0]['amount'] if rows else None} (expected {GT['CQ4']['known_2025-06-30']})"),
    "CQ4b": lambda rows: (len(rows) == 1 and rows[0]["amount"] == GT["CQ4"]["known_2025-12-31_asof_2025-06-30"], f"{rows[0]['amount'] if rows else None} (expected {GT['CQ4']['known_2025-12-31_asof_2025-06-30']})"),
    "CQ4c": lambda rows: (len(rows) == 1 and rows[0]["amount"] == GT["CQ4"]["current"], f"{rows[0]['amount'] if rows else None} (expected {GT['CQ4']['current']})"),
    "CQ5": lambda rows: ({r["location_id"]: r["occurrences"] for r in rows} == GT["CQ5"], f"{ {r['location_id']: r['occurrences'] for r in rows} }, expected {GT['CQ5']}"),
    "CQ6": lambda rows: (sorted(r["fact_id"] for r in rows) == GT["CQ6_expected_facts"] and all(r["payment_run"] and r["extraction_run"] for r in rows),
                         f"{len(rows)} supporting facts with run + model lineage (expected {len(GT['CQ6_expected_facts'])})"),
    "CQ7": lambda rows: (sorted([[r["tpa_claim"], r["carrier_claim"], r["occurrence_id"]] for r in rows], key=lambda x: x[2]) == GT["CQ7"], f"{len(rows)} pairs, expected {len(GT['CQ7'])}"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedded", action="store_true")
    ap.add_argument("--fewshot", action="store_true")
    args = ap.parse_args()

    from kg.ontology import load_ontology
    from kg.graph_ingest import _make_source
    from kg.graph_ingest_gl import build_model, validate_model, ingest_into, reconcile
    from types import SimpleNamespace

    onto = load_ontology()
    results = []

    def record(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")

    print(f"Ontology v{onto.version} ({onto.signature()})")
    errs = onto.validate_mapping()
    record("ontology/mapping consistent", not errs, "; ".join(errs[:3]))

    src = _make_source(SimpleNamespace(data_backend="sqlite", sqlite_db_path=str(ROOT / "database" / "gl_claims.db")))
    model = build_model(onto, src)
    viol = validate_model(onto, model)
    record("data conforms to ontology", not viol, f"{sum(model.node_counts().values())} nodes, {sum(model.edge_counts().values())} edges" + (f"; {viol[:2]}" if viol else ""))

    if args.embedded:
        from falkor_session import FalkorSession
        session = FalkorSession()
        out = ingest_into(session, onto, src, wipe=False, constraints=False)
        record("graph ingested (embedded engine)", True, f"{out['stats']}")
        close = lambda: None
    else:
        from config import load_config
        from neo4j import GraphDatabase
        cfg = load_config()
        driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
        session = driver.session(database=cfg.neo4j_database)
        close = lambda: (session.close(), driver.close())

    meta = list(session.run("MATCH (m:GraphMeta {name: 'business_graph_version'}) RETURN m.value AS v"))
    have = meta[0]["v"] if meta else None
    if have != onto.graph_version():
        print(f"\nSTOP: the graph in this Neo4j is not the GL graph for this ontology (found version {have!r}, expected {onto.graph_version()!r}).")
        print("The GL ingest has not run against THIS database. Run databricks/02_ingest_and_verify.py (step 4b), or tools/neo4j_reset_for_gl.py to see what is in it.")
        close()
        sys.exit(2)
    rep = reconcile(onto, model, session)
    record("graph == source tables (reconcile)", rep["ok"], f"{rep['checked']} checks" + (f", drift in {[d['name'] for d in rep['drift'][:3]]}" if rep["drift"] else ""))

    tests = yaml.safe_load((ROOT / "ontology" / "cq_tests.yaml").read_text())["tests"]
    for t in tests:
        params = dict(t.get("params") or {})
        if params.get("payment_id") == "__PAYMENT__":
            params["payment_id"] = GT["CQ6_sample_payment"]
        try:
            rows = _rows(session, t["cypher"], params)
            ok, detail = CHECKS[t["id"]](rows)
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"error: {str(e)[:160]}"
        record(t["id"], bool(ok), detail)

    if args.fewshot:
        ex = yaml.safe_load((ROOT / "ontology" / "fewshot.yaml").read_text())["examples"]
        for i, e in enumerate(ex, 1):
            if e.get("neo4j_only") and args.embedded:
                results.append((f"fewshot {i}", True, "skipped"))
                print(f"  SKIP  fewshot {i}: {e['q'][:44]:44s} (Neo4j-only syntax; run without --embedded)")
                continue
            try:
                rows = _rows(session, e["cypher"])
                record(f"fewshot {i}: {e['q'][:44]}", len(rows) > 0, f"{len(rows)} rows")
            except Exception as err:  # noqa: BLE001
                record(f"fewshot {i}: {e['q'][:44]}", False, f"error: {str(err)[:140]}")
    close()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()