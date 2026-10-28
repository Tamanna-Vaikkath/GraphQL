"""
database/validate_dev_db.py
Integrity checks for database/insurance_dev.db. Exit code 0 = all good.

    python database/validate_dev_db.py [path/to/insurance_dev.db]
"""
import sqlite3
import sys
from pathlib import Path

AS_OF = "2026-09-21"


def main(path: str) -> int:
    c = sqlite3.connect(path)
    q = lambda s: c.execute(s).fetchall()
    problems = []

    def check(name, sql, expect_zero=True):
        n = q(sql)[0][0]
        ok = (n == 0) if expect_zero else (n > 0)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {n}")
        if not ok:
            problems.append(name)

    print("Referential integrity")
    check("CLAIMS -> POLICY", "SELECT COUNT(*) FROM CLAIMS WHERE POLICY_ID NOT IN (SELECT POLICY_ID FROM POLICY)")
    check("CLAIMS -> CLAIMANT", "SELECT COUNT(*) FROM CLAIMS WHERE CLAIMANT_ID NOT IN (SELECT CLAIMANT_ID FROM CLAIMANT)")
    check("PAYMENT -> CLAIMS", "SELECT COUNT(*) FROM PAYMENT WHERE CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIMS)")
    check("LOSS_DETAIL -> CLAIMS", "SELECT COUNT(*) FROM CLAIM_LOSS_DETAIL WHERE CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIMS)")
    check("LOSS_DETAIL -> LOSS_CAUSE", "SELECT COUNT(*) FROM CLAIM_LOSS_DETAIL WHERE LOSS_CAUSE_CD NOT IN (SELECT LOSS_CAUSE_CD FROM LOSS_CAUSE)")
    check("every claim has loss detail", "SELECT COUNT(*) FROM CLAIMS WHERE CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIM_LOSS_DETAIL)")
    check("EVENT -> CLAIMS", "SELECT COUNT(*) FROM CLAIM_EVENT WHERE CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIMS)")
    check("OVERLAP -> CLAIMS", "SELECT COUNT(*) FROM CLAIM_OVERLAP WHERE CLAIM_ID_A NOT IN (SELECT CLAIM_ID FROM CLAIMS) OR CLAIM_ID_B NOT IN (SELECT CLAIM_ID FROM CLAIMS)")
    check("CLUSTER_MEMBER -> CLUSTER/CLAIMS", "SELECT COUNT(*) FROM CLAIM_CLUSTER_MEMBER WHERE CLUSTER_ID NOT IN (SELECT CLUSTER_ID FROM CLAIM_CLUSTER) OR CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIMS)")
    check("SIMILARITY -> CLAIMS", "SELECT COUNT(*) FROM CLAIM_SIMILARITY WHERE CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIMS) OR SIMILAR_CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIMS)")

    print("Date sanity (dev copy)")
    check("close before report", "SELECT COUNT(*) FROM CLAIMS WHERE CLOSE_DT IS NOT NULL AND CLOSE_DT < REPORT_DT")
    check("closed/denied without close date", f"SELECT COUNT(*) FROM CLAIMS WHERE CLM_STAT_CD IN ('C','D') AND CLOSE_DT IS NULL")
    check("open/pending WITH close date", "SELECT COUNT(*) FROM CLAIMS WHERE CLM_STAT_CD IN ('O','P') AND CLOSE_DT IS NOT NULL")
    check("dates after AS_OF", f"SELECT COUNT(*) FROM CLAIMS WHERE REPORT_DT > '{AS_OF}' OR CLOSE_DT > '{AS_OF}'")
    check("payment before report", "SELECT COUNT(*) FROM PAYMENT p JOIN CLAIMS c USING(CLAIM_ID) WHERE p.PMT_DT < c.REPORT_DT")
    check("payment after close", "SELECT COUNT(*) FROM PAYMENT p JOIN CLAIMS c USING(CLAIM_ID) WHERE c.CLOSE_DT IS NOT NULL AND p.PMT_DT > c.CLOSE_DT")
    check("payment after AS_OF", f"SELECT COUNT(*) FROM PAYMENT WHERE PMT_DT > '{AS_OF}'")

    print("Temporal events")
    check("claims with no events", "SELECT COUNT(*) FROM CLAIMS WHERE CLAIM_ID NOT IN (SELECT CLAIM_ID FROM CLAIM_EVENT)")
    check("first event is not REPORTED on REPORT_DT",
          "SELECT COUNT(*) FROM CLAIMS c JOIN CLAIM_EVENT e ON e.CLAIM_ID=c.CLAIM_ID AND e.EVENT_SEQ=1 "
          "WHERE e.EVENT_TYPE_CD<>'REPORTED' OR e.EVENT_DT<>c.REPORT_DT")
    check("dates going backwards within a claim",
          "SELECT COUNT(*) FROM CLAIM_EVENT a JOIN CLAIM_EVENT b ON a.CLAIM_ID=b.CLAIM_ID AND b.EVENT_SEQ=a.EVENT_SEQ+1 "
          "WHERE b.EVENT_DT < a.EVENT_DT")
    check("event before report date", "SELECT COUNT(*) FROM CLAIM_EVENT e JOIN CLAIMS c USING(CLAIM_ID) WHERE e.EVENT_DT < c.REPORT_DT")
    check("event after AS_OF", f"SELECT COUNT(*) FROM CLAIM_EVENT WHERE EVENT_DT > '{AS_OF}'")
    last = ("(SELECT e.* FROM CLAIM_EVENT e WHERE e.EVENT_SEQ = "
            "(SELECT MAX(EVENT_SEQ) FROM CLAIM_EVENT x WHERE x.CLAIM_ID=e.CLAIM_ID))")
    check("last event status != claim status", f"SELECT COUNT(*) FROM CLAIMS c JOIN {last} l ON l.CLAIM_ID=c.CLAIM_ID WHERE l.TO_STAT_CD <> c.CLM_STAT_CD")
    check("closed claims: last event date != CLOSE_DT", f"SELECT COUNT(*) FROM CLAIMS c JOIN {last} l ON l.CLAIM_ID=c.CLAIM_ID WHERE c.CLOSE_DT IS NOT NULL AND l.EVENT_DT <> c.CLOSE_DT")
    check("terminal incurred != CLAIMS.INCURRED_AMT (closed/denied)",
          f"SELECT COUNT(*) FROM CLAIMS c JOIN {last} l ON l.CLAIM_ID=c.CLAIM_ID WHERE c.CLM_STAT_CD IN ('C','D') AND ABS(l.INCURRED_AMT_AT_EVENT - c.INCURRED_AMT) > 0.01")
    check("open/pending: latest reserve != CLAIMS.RESERVE_AMT",
          "SELECT COUNT(*) FROM CLAIMS c WHERE c.CLM_STAT_CD IN ('O','P') AND ABS(("
          "SELECT RESERVE_AMT_AT_EVENT FROM CLAIM_EVENT e WHERE e.CLAIM_ID=c.CLAIM_ID AND e.EVENT_TYPE_CD='RESERVE_CHANGE' "
          "ORDER BY EVENT_SEQ DESC LIMIT 1) - c.RESERVE_AMT) > 0.01")
    check("payment rows without a PAYMENT_ISSUED event",
          "SELECT COUNT(*) FROM PAYMENT p WHERE NOT EXISTS (SELECT 1 FROM CLAIM_EVENT e WHERE e.CLAIM_ID=p.CLAIM_ID "
          "AND e.EVENT_TYPE_CD='PAYMENT_ISSUED' AND e.EVENT_NOTE LIKE 'Payment ' || p.PAYMENT_ID || ':%')")

    print("Overlaps / clusters / similarity")
    check("overlap pairs not ordered A<B", "SELECT COUNT(*) FROM CLAIM_OVERLAP WHERE CLAIM_ID_A >= CLAIM_ID_B")
    check("overlap pairs share nothing", "SELECT COUNT(*) FROM CLAIM_OVERLAP WHERE SHARED_POLICY_ID IS NULL AND SHARED_CLAIMANT_ID IS NULL")
    check("overlaps exist", "SELECT COUNT(*) FROM CLAIM_OVERLAP", expect_zero=False)
    check("collision-flagged pairs exist", "SELECT COUNT(*) FROM CLAIM_OVERLAP WHERE COLLISION_FLG='Y'", expect_zero=False)
    check("claim in more than one cluster", "SELECT COUNT(*) FROM (SELECT CLAIM_ID FROM CLAIM_CLUSTER_MEMBER GROUP BY 1 HAVING COUNT(*)>1)")
    check("cluster size mismatch", "SELECT COUNT(*) FROM CLAIM_CLUSTER k WHERE k.CLAIM_CNT <> (SELECT COUNT(*) FROM CLAIM_CLUSTER_MEMBER m WHERE m.CLUSTER_ID=k.CLUSTER_ID)")
    check("cluster with < 2 claims", "SELECT COUNT(*) FROM CLAIM_CLUSTER WHERE CLAIM_CNT < 2")
    check("cluster without exactly one anchor", "SELECT COUNT(*) FROM CLAIM_CLUSTER k WHERE (SELECT COUNT(*) FROM CLAIM_CLUSTER_MEMBER m WHERE m.CLUSTER_ID=k.CLUSTER_ID AND m.ANCHOR_FLG='Y') <> 1")
    check("similarity: self-match", "SELECT COUNT(*) FROM CLAIM_SIMILARITY WHERE CLAIM_ID = SIMILAR_CLAIM_ID")
    check("similarity: more than 5 per claim", "SELECT COUNT(*) FROM (SELECT CLAIM_ID FROM CLAIM_SIMILARITY GROUP BY 1 HAVING COUNT(*)>5)")

    print("\nCluster mix:", q("SELECT CLUSTER_TYPE_CD, COUNT(*), MIN(CLAIM_CNT), MAX(CLAIM_CNT) FROM CLAIM_CLUSTER GROUP BY 1"))
    print("Overlap types:", q("SELECT OVERLAP_TYPE_CD, COUNT(*), SUM(COLLISION_FLG='Y') FROM CLAIM_OVERLAP GROUP BY 1"))
    print("Event types:", q("SELECT EVENT_TYPE_CD, COUNT(*) FROM CLAIM_EVENT GROUP BY 1 ORDER BY 2 DESC"))
    print("\n" + ("ALL CHECKS PASSED" if not problems else f"{len(problems)} CHECK(S) FAILED: {problems}"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("insurance_dev.db"))))
