"""
kg/graph_ingest.py — Builds the *business data* Knowledge Graph in Neo4j.

IMPORTANT DISTINCTION
----------------------
kg/build_graph.py builds a *schema metadata* graph (:Table / :Column nodes +
:REFERENCES edges) that is used purely to help the SQL lane discover joins.

This module builds a completely separate *business data* graph — real
Claimant / Policy / Claim / Payment / Adjuster nodes carrying the actual
row data, connected by real relationships:

    (:Claimant)-[:FILED]->(:Claim)-[:COVERED_BY]->(:Policy)
    (:Claim)-[:HAS_PAYMENT]->(:Payment)
    (:Claim)-[:HANDLED_BY]->(:Adjuster)

The explicit :Adjuster node (rather than only a precomputed pairwise
SAME_ADJUSTER_AS edge) is what lets the Cypher lane perform *true* multi-hop
graph traversal — e.g. clustering claims that are transitively connected
through a shared adjuster, shared agent, or shared payee — instead of
relying on a shortcut edge that was already pairwise-flattened at ingest
time.

This is the graph the Cypher lane queries and traverses. It is what makes
relationship / multi-hop / unknown-depth questions ("claimants who share a
policy", "claims connected through the same adjuster", "chains of payments
across a claimant's claims", etc.) answerable with real Cypher execution
instead of SQL joins.

Run once (or whenever the SQLite data changes):
    python kg/graph_ingest.py [path/to/db]

The pipeline also calls `ensure_graph_ingested()` lazily on first Cypher
query so the demo works out-of-the-box without a manual step.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

from config import Config, load_config
from utils.safety import assert_neo4j_is_dev_or_empty, stamp_dev_marker

logger = logging.getLogger(__name__)

# Node label used as a "has the business graph been built?" marker.
_MARKER_LABEL = "Claim"

# Bump this whenever the shape of the business graph changes, so an older DEV
# graph is rebuilt automatically. (Only ever rebuilds a database that carries
# the dev marker or is empty — see utils/safety.py.)
GRAPH_VERSION = "2-analytics"

# Every label this module owns (and may therefore wipe when rebuilding).
_OWNED_LABELS = ["Claimant", "Policy", "Claim", "Payment", "Adjuster",
                 "LossCause", "Cluster", "ClaimEvent"]


def _clean_value(v: Any) -> Any:
    """Make a Delta/pandas value safe for the Neo4j driver (no Decimal / NaN / NaT / date objects)."""
    import datetime as _dt
    import decimal as _dec
    if v is None:
        return None
    if isinstance(v, _dec.Decimal):
        return float(v)
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    try:
        if v != v:           # NaN / NaT
            return None
    except Exception:        # noqa: BLE001
        pass
    if hasattr(v, "item"):   # numpy scalar
        try:
            return v.item()
        except Exception:    # noqa: BLE001
            return v
    return v


class _SqliteSource:
    """Reads the source tables from a local SQLite file."""
    label = "SQLite"

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def rows(self, table: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(f"SELECT * FROM {table}").fetchall()]

    def rows_optional(self, table: str) -> List[Dict[str, Any]]:
        """[] when the table does not exist (e.g. the original 4-table demo database)."""
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        return self.rows(table) if exists else []

    def close(self) -> None:
        self.conn.close()


class _DeltaSource:
    """Reads the source tables from Unity Catalog / Delta (catalog.schema.TABLE) through the
    same SQL Warehouse connection the SQL lane uses -> ONE source of truth for both lanes.
    Column names are upper-cased so the ingest code is identical for both backends."""
    label = "Databricks (Delta)"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def rows(self, table: str) -> List[Dict[str, Any]]:
        from utils.db_connector import run_query, qualify
        df = run_query(self.cfg, f"SELECT * FROM {qualify(self.cfg, table)}")
        return [{str(k).upper(): _clean_value(v) for k, v in rec.items()}
                for rec in df.to_dict("records")]

    def rows_optional(self, table: str) -> List[Dict[str, Any]]:
        try:
            return self.rows(table)
        except Exception as e:  # noqa: BLE001  (table missing in this schema)
            logger.info("Optional table %s not readable (%s) - treating as empty", table, str(e)[:120])
            return []

    def close(self) -> None:
        pass


def _make_source(cfg: Config, db_path: str = None):
    if getattr(cfg, "data_backend", "sqlite") == "databricks":
        return _DeltaSource(cfg)
    return _SqliteSource(db_path or cfg.sqlite_db_path)


def graph_is_built(cfg: Config) -> bool:
    """Cheap check: does the business graph already have Claim nodes?"""
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    try:
        with driver.session(database=cfg.neo4j_database) as session:
            rec = session.run(f"MATCH (n:{_MARKER_LABEL}) RETURN count(n) AS c").single()
            if not (rec and rec["c"] and rec["c"] > 0):
                return False
            ver = session.run(
                "MATCH (m:GraphMeta {name: 'business_graph_version'}) RETURN m.value AS v").single()
            return bool(ver and ver["v"] == GRAPH_VERSION)
    except Exception as e:
        logger.warning("graph_is_built() check failed: %s", e)
        return False
    finally:
        driver.close()


def ensure_graph_ingested(cfg: Config, db_path: str = None) -> bool:
    """Build the business graph if it doesn't exist yet. Returns True if a
    build was performed, False if the graph already existed."""
    if graph_is_built(cfg):
        print("[graph_ingest] Business KG already present — skipping ingest.")
        return False
    print("[graph_ingest] Business KG not found — ingesting from the configured data backend now...")
    ingest(db_path or cfg.sqlite_db_path, cfg)
    return True


def ingest(db_path: str, cfg: Config = None) -> None:
    cfg = cfg or load_config()
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    src = _make_source(cfg, db_path)
    print(f"[graph_ingest] Reading source tables from {src.label}")

    try:
        claimants = src.rows("CLAIMANT")
        policies  = src.rows("POLICY")
        claims    = src.rows("CLAIMS")
        payments  = src.rows("PAYMENT")

        # Extension tables (only present in the dev database)
        loss_causes = src.rows_optional("LOSS_CAUSE")
        loss_detail = src.rows_optional("CLAIM_LOSS_DETAIL")
        events      = src.rows_optional("CLAIM_EVENT")
        overlaps    = src.rows_optional("CLAIM_OVERLAP")
        clusters    = src.rows_optional("CLAIM_CLUSTER")
        members     = src.rows_optional("CLAIM_CLUSTER_MEMBER")
        similar     = src.rows_optional("CLAIM_SIMILARITY")

        with driver.session(database=cfg.neo4j_database) as session:
            # SAFETY: only ever delete/rebuild a database that is empty or was
            # created by the dev build. A production graph is refused here,
            # BEFORE anything is deleted.
            assert_neo4j_is_dev_or_empty(session, cfg.neo4j_uri, "rebuild the business graph in")

            # Wipe only the business-graph labels; leave the schema-metadata
            # graph (:Table / :Column) untouched.
            session.run(
                "MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) DETACH DELETE n",
                labels=_OWNED_LABELS,
            )
            print("[graph_ingest] Cleared prior business graph nodes.")

            # Constraints for fast MERGE / lookups 
            for label, key in [
                ("Claimant", "claimant_id"), ("Policy", "policy_id"),
                ("Claim", "claim_id"), ("Payment", "payment_id"),
                ("Adjuster", "adjuster_id"),
                ("LossCause", "cause_code"), ("Cluster", "cluster_id"),
                ("ClaimEvent", "event_id"),
            ]:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                    f"REQUIRE n.{key} IS UNIQUE"
                )

            # Nodes 
            session.run("""
                UNWIND $rows AS r
                MERGE (c:Claimant {claimant_id: r.CLAIMANT_ID})
                SET c.name = r.CLAIMANT_NM, c.dob = r.DOB, c.gender = r.GENDER_CD,
                    c.state = r.STATE_CD, c.address = r.ADDRESS_LINE1,
                    c.phone = r.CONTACT_PHONE, c.attorney_represented = r.ATTY_REP_FLG,
                    c.claim_count = r.CLAIM_COUNT, c.fraud_risk_score = r.FRAUD_RISK_SCRE
            """, rows=claimants)
            print(f"[graph_ingest] Claimant nodes: {len(claimants)}")

            session.run("""
                UNWIND $rows AS r
                MERGE (p:Policy {policy_id: r.POLICY_ID})
                SET p.policy_number = r.POLICY_NBR, p.insured_name = r.INSURED_NM,
                    p.effective_date = r.POL_EFF_DT, p.expiration_date = r.POL_EXP_DT,
                    p.line_of_business = r.LINE_OF_BUSNSS, p.state = r.STATE_CD,
                    p.premium_amount = r.PREMIUM_AMT, p.deductible_amount = r.DEDUCTIBLE_AMT,
                    p.agent_id = r.AGENT_ID, p.status = r.POL_STAT_CD
            """, rows=policies)
            print(f"[graph_ingest] Policy nodes: {len(policies)}")

            session.run("""
                UNWIND $rows AS r
                MERGE (c:Claim {claim_id: r.CLAIM_ID})
                SET c.status = r.CLM_STAT_CD, c.loss_date = r.LOSS_DT,
                    c.report_date = r.REPORT_DT, c.loss_type = r.LOSS_TYPE_CD,
                    c.incurred_amount = r.INCURRED_AMT, c.reserve_amount = r.RESERVE_AMT,
                    c.adjuster_id = r.ADJUSTER_ID, c.close_date = r.CLOSE_DT,
                    c.litigation_flag = r.LITIGATION_FLG,
                    c._policy_id = r.POLICY_ID, c._claimant_id = r.CLAIMANT_ID
            """, rows=claims)
            print(f"[graph_ingest] Claim nodes: {len(claims)}")

            session.run("""
                UNWIND $rows AS r
                MERGE (p:Payment {payment_id: r.PAYMENT_ID})
                SET p.payment_date = r.PMT_DT, p.gross_amount = r.PMT_AMT_GROSS,
                    p.net_amount = r.PMT_AMT_NET, p.status = r.PMT_STAT_CD,
                    p.type = r.PMT_TYPE_CD, p.payee_name = r.PAYEE_NM,
                    p.check_number = r.CHK_NBR, p.void_reason = r.VOID_RSN_CD,
                    p._claim_id = r.CLAIM_ID
            """, rows=payments)
            print(f"[graph_ingest] Payment nodes: {len(payments)}")

            # Adjuster nodes 
            # Derived from the distinct ADJUSTER_ID values on CLAIMS, since
            # there's no standalone ADJUSTER table in the source SQLite db.
            # A real node (rather than only a flattened SAME_ADJUSTER_AS edge)
            # is what enables true multi-hop traversal / clustering later.
            adjuster_ids = sorted({
                r["ADJUSTER_ID"] for r in claims if r.get("ADJUSTER_ID")
            })
            session.run("""
                UNWIND $ids AS aid
                MERGE (a:Adjuster {adjuster_id: aid})
            """, ids=adjuster_ids)
            print(f"[graph_ingest] Adjuster nodes: {len(adjuster_ids)}")

            # Relationships 
            session.run("""
                MATCH (cl:Claim) WHERE cl._claimant_id IS NOT NULL
                MATCH (cm:Claimant {claimant_id: cl._claimant_id})
                MERGE (cm)-[:FILED]->(cl)
            """)
            session.run("""
                MATCH (cl:Claim) WHERE cl._policy_id IS NOT NULL
                MATCH (p:Policy {policy_id: cl._policy_id})
                MERGE (cl)-[:COVERED_BY]->(p)
            """)
            session.run("""
                MATCH (pm:Payment) WHERE pm._claim_id IS NOT NULL
                MATCH (cl:Claim {claim_id: pm._claim_id})
                MERGE (cl)-[:HAS_PAYMENT]->(pm)
            """)
            session.run("""
                MATCH (cl:Claim) WHERE cl.adjuster_id IS NOT NULL
                MATCH (a:Adjuster {adjuster_id: cl.adjuster_id})
                MERGE (cl)-[:HANDLED_BY]->(a)
            """)
            # Same-adjuster / same-agent edges make multi-hop "who else..."
            # questions answerable without unknown intermediate keys.
            session.run("""
                MATCH (a:Claim), (b:Claim)
                WHERE a.adjuster_id IS NOT NULL AND a.adjuster_id = b.adjuster_id
                  AND id(a) < id(b)
                MERGE (a)-[:SAME_ADJUSTER_AS]->(b)
            """)
            session.run("""
                MATCH (a:Policy), (b:Policy)
                WHERE a.agent_id IS NOT NULL AND a.agent_id = b.agent_id
                  AND id(a) < id(b)
                MERGE (a)-[:SAME_AGENT_AS]->(b)
            """)
            print("[graph_ingest] Relationships created: FILED, COVERED_BY, "
                  "HAS_PAYMENT, HANDLED_BY, SAME_ADJUSTER_AS, SAME_AGENT_AS")

            # ── DEV EXTENSION: loss cause, clusters, overlaps, similarity, timeline ──
            if loss_causes:
                session.run("""
                    UNWIND $rows AS r
                    MERGE (l:LossCause {cause_code: r.LOSS_CAUSE_CD})
                    SET l.description = r.LOSS_CAUSE_DESC, l.loss_type = r.LOSS_TYPE_CD,
                        l.peril_group = r.PERIL_GROUP_CD
                """, rows=loss_causes)
                print(f"[graph_ingest] LossCause nodes: {len(loss_causes)}")

            if loss_detail:
                session.run("""
                    UNWIND $rows AS r
                    MATCH (c:Claim {claim_id: r.CLAIM_ID})
                    SET c.loss_cause = r.LOSS_CAUSE_CD, c.loss_description = r.LOSS_DESC,
                        c.loss_state = r.LOSS_STATE_CD, c.severity = r.SEVERITY_CD
                    WITH c, r
                    MATCH (l:LossCause {cause_code: r.LOSS_CAUSE_CD})
                    MERGE (c)-[:CAUSED_BY]->(l)
                """, rows=loss_detail)
                print(f"[graph_ingest] CAUSED_BY edges: {len(loss_detail)}")

            if clusters:
                session.run("""
                    UNWIND $rows AS r
                    MERGE (k:Cluster {cluster_id: r.CLUSTER_ID})
                    SET k.cluster_type = r.CLUSTER_TYPE_CD, k.label = r.CLUSTER_LABEL,
                        k.claim_count = r.CLAIM_CNT, k.total_incurred = r.TOTAL_INCURRED_AMT,
                        k.first_loss_date = r.FIRST_LOSS_DT, k.last_loss_date = r.LAST_LOSS_DT,
                        k.span_days = r.SPAN_DAYS, k.risk_level = r.RISK_LEVEL_CD,
                        k.link_basis = r.LINK_BASIS_TXT
                """, rows=clusters)
                session.run("""
                    UNWIND $rows AS r
                    MATCH (c:Claim {claim_id: r.CLAIM_ID})
                    MATCH (k:Cluster {cluster_id: r.CLUSTER_ID})
                    MERGE (c)-[m:IN_CLUSTER]->(k)
                    SET m.link_degree = r.LINK_DEGREE, m.is_anchor = (r.ANCHOR_FLG = 'Y')
                """, rows=members)
                print(f"[graph_ingest] Cluster nodes: {len(clusters)}, IN_CLUSTER edges: {len(members)}")

            if overlaps:
                session.run("""
                    UNWIND $rows AS r
                    MATCH (a:Claim {claim_id: r.CLAIM_ID_A})
                    MATCH (b:Claim {claim_id: r.CLAIM_ID_B})
                    MERGE (a)-[o:OVERLAPS_WITH {overlap_id: r.OVERLAP_ID}]->(b)
                    SET o.overlap_type = r.OVERLAP_TYPE_CD, o.days_apart = r.DAYS_APART,
                        o.score = r.OVERLAP_SCORE, o.collision = (r.COLLISION_FLG = 'Y'),
                        o.period_overlap_days = r.PERIOD_OVERLAP_DAYS,
                        o.amount_diff_pct = r.AMT_DIFF_PCT, o.reason = r.OVERLAP_REASON
                """, rows=overlaps)
                print(f"[graph_ingest] OVERLAPS_WITH edges: {len(overlaps)}")

            if similar:
                session.run("""
                    UNWIND $rows AS r
                    MATCH (a:Claim {claim_id: r.CLAIM_ID})
                    MATCH (b:Claim {claim_id: r.SIMILAR_CLAIM_ID})
                    MERGE (a)-[s:SIMILAR_TO {similarity_id: r.SIMILARITY_ID}]->(b)
                    SET s.rank = r.SIMILARITY_RANK, s.score = r.SIMILARITY_SCORE,
                        s.basis = r.MATCH_BASIS_TXT
                """, rows=similar)
                print(f"[graph_ingest] SIMILAR_TO edges: {len(similar)}")

            if events:
                session.run("""
                    UNWIND $rows AS r
                    MERGE (e:ClaimEvent {event_id: r.EVENT_ID})
                    SET e.claim_id = r.CLAIM_ID, e.seq = r.EVENT_SEQ, e.date = r.EVENT_DT,
                        e.type = r.EVENT_TYPE_CD, e.from_status = r.FROM_STAT_CD,
                        e.to_status = r.TO_STAT_CD, e.incurred_at_event = r.INCURRED_AMT_AT_EVENT,
                        e.reserve_at_event = r.RESERVE_AMT_AT_EVENT, e.adjuster_id = r.ADJUSTER_ID,
                        e.note = r.EVENT_NOTE
                """, rows=events)
                session.run("""
                    UNWIND $rows AS r
                    MATCH (c:Claim {claim_id: r.CLAIM_ID})
                    MATCH (e:ClaimEvent {event_id: r.EVENT_ID})
                    MERGE (c)-[:HAS_EVENT]->(e)
                """, rows=events)
                by_claim: Dict[Any, list] = {}
                for e in events:
                    by_claim.setdefault(e["CLAIM_ID"], []).append(e)
                nxt = []
                for lst in by_claim.values():
                    lst.sort(key=lambda x: x["EVENT_SEQ"])
                    nxt.extend({"a": x["EVENT_ID"], "b": y["EVENT_ID"]} for x, y in zip(lst, lst[1:]))
                session.run("""
                    UNWIND $rows AS r
                    MATCH (a:ClaimEvent {event_id: r.a}), (b:ClaimEvent {event_id: r.b})
                    MERGE (a)-[:NEXT]->(b)
                """, rows=nxt)
                print(f"[graph_ingest] ClaimEvent nodes: {len(events)}, NEXT edges: {len(nxt)}")

            # Version + dev marker (see utils/safety.py)
            session.run(
                "MERGE (m:GraphMeta {name: 'business_graph_version'}) SET m.value = $v", v=GRAPH_VERSION)
            stamp_dev_marker(session)

        print("[graph_ingest] Business Knowledge Graph build complete.")
    finally:
        src.close()
        driver.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    db = sys.argv[1] if len(sys.argv) > 1 else "database/insurance_dev.db"
    if not Path(db).exists():
        print(f"Database not found at {db}. Run: python database/build_dev_db.py first.")
        sys.exit(1)
    ingest(db)