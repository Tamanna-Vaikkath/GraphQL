"""
kg/graph_ingest.py — Builds the *business data* Knowledge Graph in Neo4j.

IMPORTANT DISTINCTION
----------------------
kg/build_graph.py builds a *schema metadata* graph (:Table / :Column nodes +
:REFERENCES edges) that is used purely to help the SQL lane discover joins.

This module builds a completely separate *business data* graph — real
Claimant / Policy / Claim / Payment nodes carrying the actual row data,
connected by real relationships:

    (:Claimant)-[:FILED]->(:Claim)-[:COVERED_BY]->(:Policy)
    (:Claim)-[:HAS_PAYMENT]->(:Payment)

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

logger = logging.getLogger(__name__)

# Node label used as a "has the business graph been built?" marker.
_MARKER_LABEL = "Claim"


def _rows(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(f"SELECT * FROM {table}")
    return [dict(r) for r in cur.fetchall()]


def graph_is_built(cfg: Config) -> bool:
    """Cheap check: does the business graph already have Claim nodes?"""
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    try:
        with driver.session(database=cfg.neo4j_database) as session:
            rec = session.run(f"MATCH (n:{_MARKER_LABEL}) RETURN count(n) AS c").single()
            return bool(rec and rec["c"] and rec["c"] > 0)
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
    print("[graph_ingest] Business KG not found — ingesting from SQLite now...")
    ingest(db_path or cfg.sqlite_db_path, cfg)
    return True


def ingest(db_path: str, cfg: Config = None) -> None:
    cfg = cfg or load_config()
    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password))
    conn = sqlite3.connect(db_path)

    try:
        claimants = _rows(conn, "CLAIMANT")
        policies  = _rows(conn, "POLICY")
        claims    = _rows(conn, "CLAIMS")
        payments  = _rows(conn, "PAYMENT")

        with driver.session(database=cfg.neo4j_database) as session:
            # Wipe only the business-graph labels; leave the schema-metadata
            # graph (:Table / :Column) untouched.
            session.run(
                "MATCH (n) WHERE n:Claimant OR n:Policy OR n:Claim OR n:Payment "
                "DETACH DELETE n"
            )
            print("[graph_ingest] Cleared prior business graph nodes.")

            # ── Constraints for fast MERGE / lookups ───────────────────────
            for label, key in [
                ("Claimant", "claimant_id"), ("Policy", "policy_id"),
                ("Claim", "claim_id"), ("Payment", "payment_id"),
            ]:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                    f"REQUIRE n.{key} IS UNIQUE"
                )

            # ── Nodes ───────────────────────────────────────────────────────
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

            # ── Relationships ───────────────────────────────────────────────
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
                  "HAS_PAYMENT, SAME_ADJUSTER_AS, SAME_AGENT_AS")

        print("[graph_ingest] Business Knowledge Graph build complete.")
    finally:
        conn.close()
        driver.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    db = sys.argv[1] if len(sys.argv) > 1 else "database/insurance_demo.db"
    if not Path(db).exists():
        print(f"Database not found at {db}. Run: python database/seed_db.py first.")
        sys.exit(1)
    ingest(db)
