"""
utils/health.py — one-click diagnosis of "the app says no rows".

collect_health(cfg) gathers the facts (Delta table counts, what is really inside Neo4j, whether the schema graph and the
ontology-driven business graph exist); diagnose(h) turns them into ordered, plain-language findings with the action to take.
The Neo4j part takes a session so it can be unit-tested against an embedded engine (tests/test_health.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

GL_LABELS = ["Claim", "Feature", "Attorney", "Litigation", "Reserve", "Payment", "Chunk", "ExtractedFact"]
SQL_TABLES = ["claim", "feature", "attorney", "litigation", "reserve_state", "payment", "text_chunk"]


def sql_health(cfg) -> Dict[str, Any]:
    from utils.db_connector import qualify, run_query
    out: Dict[str, Any] = {"target": qualify(cfg, "<table>").replace(".<table>", ""), "counts": {}, "error": None}
    for t in SQL_TABLES:
        try:
            df = run_query(cfg, f"SELECT COUNT(*) AS c FROM {qualify(cfg, t)}")
            out["counts"][t] = int(df.iloc[0, 0])
        except Exception as e:  # noqa: BLE001
            out["counts"][t] = None
            out["error"] = out["error"] or f"{t}: {str(e)[:160]}"
    return out


def _inspect(session) -> Dict[str, Any]:
    """What is in this Neo4j database? Self-contained on purpose: the health page must work even if other files are out of date."""
    from utils.safety import DEV_MARKER_NAME, DEV_MARKER_VALUE
    total = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    markers = [r["v"] for r in session.run("MATCH (m:GraphMeta {name: $n}) RETURN m.value AS v", n=DEV_MARKER_NAME)]
    rows = list(session.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c, count(n.id) AS with_id ORDER BY c DESC"))
    versions = [r["v"] for r in session.run("MATCH (m:GraphMeta {name: 'business_graph_version'}) RETURN m.value AS v")]
    verdict = ("EMPTY" if total == 0 else "DEV_DEMO" if "dev" in markers else "GL" if DEV_MARKER_VALUE in markers else "UNMARKED")
    return dict(total=total, markers=markers, labels={r["l"]: r["c"] for r in rows}, labels_with_id={r["l"]: r["with_id"] for r in rows},
                business_graph_version=versions, verdict=verdict)


def registry_health(registry) -> Dict[str, Any]:
    """What does the RUNNING app's schema registry know? (It is loaded once at startup.)"""
    if registry is None:
        return {"present": False}
    names = list(getattr(registry, "table_names", []) or [])
    cm = registry.build_concept_map()
    from pipeline.hyde_expander import _GENERIC_QUERY_TERMS
    import re
    stop = frozenset({"show", "list", "get", "find", "which", "what", "have", "has", "are", "is", "the", "a", "an", "for", "of", "with", "and", "or",
                      "in", "on", "to", "from", "by", "where", "that", "me", "all", "each", "how", "many"}) | _GENERIC_QUERY_TERMS
    probes = {}
    for q in ["Which attorneys have more than 10 open suits?", "show open claims", "What has been paid under each coverage part?"]:
        ql = q.lower()
        cov = " ".join(c for c in cm if c in ql)
        probes[q] = [w for w in re.findall(r"[a-z]+", ql) if w not in stop and w not in cov and not w.isdigit()]
    return {"present": True, "table_count": len(names), "tables": names, "has_attorney": "ATTORNEY" in [n.upper() for n in names],
            "custom_aliases": len(getattr(registry, "_custom_aliases", {}) or {}), "uncovered_words": probes}


def neo4j_health(session, onto) -> Dict[str, Any]:
    rep = _inspect(session)
    h: Dict[str, Any] = {
        "verdict": rep["verdict"], "total_nodes": rep["total"], "markers": rep["markers"],
        "graph_version": (rep["business_graph_version"] or [None])[0], "expected_version": onto.graph_version(),
        "labels": rep["labels"], "labels_with_id": rep["labels_with_id"],
    }
    h["table_nodes"] = [r["n"] for r in session.run("MATCH (t:Table) RETURN t.name AS n ORDER BY n")]
    h["ontology_classes_in_graph"] = session.run("MATCH (k:OntClass) RETURN count(k) AS c").single()["c"]
    gl = {}
    for lab in GL_LABELS:
        r = session.run(f"MATCH (n:{lab}) RETURN count(n) AS c, count(n.id) AS with_id").single()
        gl[lab] = {"nodes": r["c"], "with_id": r["with_id"]}
    h["gl_counts"] = gl
    r = session.run("MATCH (a:Attorney)<-[:COUNSEL]-(l:Litigation) WHERE l.stage <> 'closed' "
                    "RETURN a.name AS name, count(DISTINCT l) AS open_suits ORDER BY open_suits DESC LIMIT 1").single()
    h["top_attorney"] = dict(r) if r else None
    return h


def collect_health(cfg, registry=None) -> Dict[str, Any]:
    from kg.ontology import load_ontology
    onto = load_ontology()
    h: Dict[str, Any] = {"backend": cfg.data_backend, "ontology_version": onto.version, "neo4j": None, "neo4j_error": None, "sql": None, "sql_error": None,
                         "registry": None}
    try:
        h["registry"] = registry_health(registry)
    except Exception as e:  # noqa: BLE001
        h["registry"] = {"present": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}
    try:
        h["sql"] = sql_health(cfg)
    except Exception as e:  # noqa: BLE001
        h["sql_error"] = str(e)[:200]
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
        try:
            with driver.session(database=cfg.neo4j_database) as s:
                h["neo4j"] = neo4j_health(s, onto)
        finally:
            driver.close()
    except Exception as e:  # noqa: BLE001
        h["neo4j_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        h["neo4j_error_is_connection"] = type(e).__name__ in ("ServiceUnavailable", "AuthError", "ConfigurationError", "SessionExpired", "DriverError") or "auth" in str(e).lower()
    return h


def diagnose(h: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Ordered findings: (level, message). level is 'error' | 'warn' | 'ok'."""
    f: List[Tuple[str, str]] = []
    if h.get("neo4j_error"):
        hint = ("  -> check NEO4J_URI / credentials in the app's secret resources." if h.get("neo4j_error_is_connection")
                else "  -> Neo4j answered, but reading it failed inside the app (not a credentials problem). Send me this line.")
        f.append(("error", f"Could not read Neo4j: {h['neo4j_error']}{hint}"))
    n = h.get("neo4j")
    if n:
        gl_ids = sum(v["with_id"] for v in n["gl_counts"].values())
        if n["verdict"] == "DEV_DEMO":
            f.append(("error", "This Neo4j instance holds your WORKING demo's graph (marker 'dev'). The GL app must use a different instance."))
        elif n["total_nodes"] == 0:
            f.append(("error", "Neo4j is EMPTY: the GL graphs have not been built. Run databricks/02_ingest_and_verify.py (or use the button below for the business graph)."))
        elif gl_ids == 0:
            f.append(("error", f"Neo4j holds {n['total_nodes']} nodes but NONE are GL nodes (no `id` property on Claim/Attorney/...). "
                               "It still contains the OLD demo's graph, so every Cypher query returns 0 rows. "
                               "Run databricks/02_ingest_and_verify.py with WIPE_OLD_GRAPH = True (and the host typed in CONFIRM_HOST), then restart the app."))
        elif n["graph_version"] != n["expected_version"]:
            f.append(("warn", f"The graph was built from ontology version {n['graph_version']!r} but this app expects {n['expected_version']!r}. Re-run the ingest."))
        else:
            f.append(("ok", f"GL business graph present and current ({n['graph_version']})."))
        if not n["table_nodes"]:
            f.append(("error", "The schema graph (:Table/:Column) is MISSING, so the SQL lane cannot pick tables. Run kg.build_graph (notebook 02, step 4a)."))
        else:
            tn = n["table_nodes"]
            sample = ", ".join(tn[:6]) + (" ..." if len(tn) > 6 else "")
            if "attorney" not in [t.lower() for t in tn]:
                f.append(("error", f"Schema graph does not contain the GL tables (found: {sample}). It is the OLD demo's schema graph. Rebuild it (notebook 02, step 4a)."))
            else:
                f.append(("ok", f"Schema graph present: {len(tn)} tables."))
        if gl_ids and n["top_attorney"]:
            f.append(("ok", f"Sanity check: top attorney by open suits = {n['top_attorney']['name']} ({n['top_attorney']['open_suits']})."))
        elif gl_ids:
            f.append(("warn", "GL nodes exist but no Attorney<-COUNSEL-Litigation pattern was found: relationships may be missing. Re-run the ingest."))
    r = h.get("registry")
    if r is not None:
        if not r.get("present"):
            f.append(("warn", f"Could not inspect the app's schema registry: {r.get('error', 'not available')}"))
        elif not r["has_attorney"]:
            f.append(("error", f"The RUNNING app loaded a schema with {r['table_count']} tables and NO `attorney` table ({', '.join(r['tables'][:8])}...). "
                               "It read graphql.insurance before the GL tables were loaded, or from a stale state. -> Redeploy / restart the app so it re-reads the schema. "
                               "This is why questions say 'attorneys' is not in the database."))
        else:
            unc = {q: w for q, w in r["uncovered_words"].items() if w}
            f.append(("ok", f"The running app knows {r['table_count']} tables including `attorney`, with {r['custom_aliases']} business aliases loaded."))
            if r["custom_aliases"] == 0:
                f.append(("warn", "No business aliases were loaded (kg/aliases.json missing or unreadable), so grounding relies on the LLM and may refuse valid questions."))
    if h.get("sql_error"):
        f.append(("error", f"Cannot query Delta: {h['sql_error']}"))
    s = h.get("sql")
    if s:
        zero = [t for t, c in s["counts"].items() if c in (0, None)]
        if s.get("error") and len(zero) == len(s["counts"]):
            f.append(("error", f"No GL table is readable in {s['target']}: {s['error']}  -> run databricks/01_load_gl_tables.py and check the app's permissions."))
        elif zero:
            f.append(("error", f"Empty or unreadable tables in {s['target']}: {zero}. Run databricks/01_load_gl_tables.py."))
        else:
            f.append(("ok", f"Delta tables readable in {s['target']} (claim = {s['counts']['claim']} rows)."))
    if f and all(lvl == "ok" for lvl, _ in f):
        f.append(("ok", "Both stores look healthy. If a question still returns 0 rows, open 'Generated Cypher' / 'Show Query Trace' on that question and send it to me: the generated query is then the issue, not the data."))
    return f