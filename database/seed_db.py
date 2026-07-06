"""
database/seed_db.py — Create and seed the SQLite demo database.
Run once: python database/seed_db.py
Produces: database/insurance_demo.db with realistic P&C sample data.
"""
import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "insurance_demo.db"

STATES = ["TX", "CA", "FL", "NY", "OH", "PA", "IL", "GA", "NC", "AZ"]
LINES = ["PERSONAL_AUTO", "HOMEOWNERS", "COMMERCIAL", "WC"]
LOSS_TYPES = ["AUTO", "PROP", "LIAB", "WC", "MARINE"]
POL_STAT = ["AC", "CN", "EX"]
CLM_STAT = ["O", "C", "P", "D"]
PMT_STAT = ["IS", "CL", "VD", "PD"]
PMT_TYPES = ["INDEM", "MED", "EXP"]
GENDERS = ["M", "F", "U"]


def rand_date(start_year=2021, end_year=2025) -> str:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def create_schema(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS CLAIMANT (
        CLAIMANT_ID     INTEGER PRIMARY KEY,
        CLAIMANT_NM     TEXT NOT NULL,
        DOB             TEXT,
        GENDER_CD       TEXT,
        ADDRESS_LINE1   TEXT,
        STATE_CD        TEXT,
        CONTACT_PHONE   TEXT,
        ATTY_REP_FLG    TEXT DEFAULT 'N',
        CLAIM_COUNT     INTEGER DEFAULT 0,
        FRAUD_RISK_SCRE REAL DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS POLICY (
        POLICY_ID       INTEGER PRIMARY KEY,
        POLICY_NBR      TEXT UNIQUE,
        INSURED_NM      TEXT,
        POL_EFF_DT      TEXT,
        POL_EXP_DT      TEXT,
        LINE_OF_BUSNSS  TEXT,
        STATE_CD        TEXT,
        PREMIUM_AMT     REAL,
        DEDUCTIBLE_AMT  REAL,
        AGENT_ID        INTEGER,
        POL_STAT_CD     TEXT
    );

    CREATE TABLE IF NOT EXISTS CLAIMS (
        CLAIM_ID        INTEGER PRIMARY KEY,
        POLICY_ID       INTEGER REFERENCES POLICY(POLICY_ID),
        CLAIMANT_ID     INTEGER REFERENCES CLAIMANT(CLAIMANT_ID),
        CLM_STAT_CD     TEXT,
        LOSS_DT         TEXT,
        REPORT_DT       TEXT,
        LOSS_TYPE_CD    TEXT,
        INCURRED_AMT    REAL,
        RESERVE_AMT     REAL,
        ADJUSTER_ID     INTEGER,
        CLOSE_DT        TEXT,
        LITIGATION_FLG  TEXT DEFAULT 'N'
    );

    CREATE TABLE IF NOT EXISTS PAYMENT (
        PAYMENT_ID      INTEGER PRIMARY KEY,
        CLAIM_ID        INTEGER REFERENCES CLAIMS(CLAIM_ID),
        PMT_DT          TEXT,
        PMT_AMT_GROSS   REAL,
        PMT_AMT_NET     REAL,
        PMT_STAT_CD     TEXT,
        PMT_TYPE_CD     TEXT,
        PAYEE_NM        TEXT,
        CHK_NBR         TEXT,
        VOID_RSN_CD     TEXT
    );
    """)


def seed(conn: sqlite3.Connection, n_claimants=200, n_policies=300, n_claims=500, n_payments=700):
    random.seed(42)
    first_names = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
                   "William","Barbara","David","Elizabeth","Richard","Susan","Joseph","Jessica",
                   "Thomas","Sarah","Charles","Karen"]
    last_names = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
                  "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
                  "Thomas","Taylor","Moore","Jackson","Martin"]

    # Claimants
    claimants = []
    for i in range(1, n_claimants + 1):
        nm = f"{random.choice(first_names)} {random.choice(last_names)}"
        claimants.append((
            i, nm, rand_date(1955, 2000), random.choice(GENDERS),
            f"{random.randint(100,9999)} Main St", random.choice(STATES),
            f"555-{random.randint(1000,9999)}", random.choice(["Y","N","N","N"]),
            random.randint(0, 8), round(random.uniform(0, 100), 2)
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO CLAIMANT VALUES(?,?,?,?,?,?,?,?,?,?)", claimants)

    # Policies
    policies = []
    for i in range(1, n_policies + 1):
        eff = rand_date(2020, 2023)
        exp_dt = (date.fromisoformat(eff) + timedelta(days=365)).isoformat()
        status = "AC" if date.fromisoformat(exp_dt) > date.today() else random.choice(["CN","EX"])
        policies.append((
            i, f"PL-{2020+i//100}-{i:05d}",
            f"{random.choice(first_names)} {random.choice(last_names)}",
            eff, exp_dt, random.choice(LINES), random.choice(STATES),
            round(random.uniform(800, 12000), 2), random.choice([500, 1000, 2500, 5000]),
            random.randint(1, 50), status
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO POLICY VALUES(?,?,?,?,?,?,?,?,?,?,?)", policies)

    # Claims
    claims = []
    for i in range(1, n_claims + 1):
        pol_id = random.randint(1, n_policies)
        clmt_id = random.randint(1, n_claimants)
        stat = random.choices(CLM_STAT, weights=[30, 40, 15, 15])[0]
        loss_dt = rand_date(2022, 2025)
        report_dt = (date.fromisoformat(loss_dt) + timedelta(days=random.randint(1, 30))).isoformat()
        close_dt = None if stat in ("O", "P") else rand_date(2023, 2026)
        incurred = round(random.uniform(1000, 500000), 2)
        reserve = round(incurred * random.uniform(0.1, 0.6), 2) if stat in ("O","P") else 0.0
        claims.append((
            i, pol_id, clmt_id, stat, loss_dt, report_dt,
            random.choice(LOSS_TYPES), incurred, reserve,
            random.randint(1, 30), close_dt, random.choice(["Y","N","N","N","N"])
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO CLAIMS VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", claims)

    # Payments
    payments = []
    claim_ids_with_payments = random.sample(range(1, n_claims + 1), min(n_payments, n_claims))
    for i, cid in enumerate(claim_ids_with_payments, 1):
        gross = round(random.uniform(500, 100000), 2)
        net = round(gross * 0.9, 2)
        stat = random.choices(PMT_STAT, weights=[40, 35, 10, 15])[0]
        void_rsn = random.choice(["DUPE", "ERROR", None]) if stat == "VD" else None
        payments.append((
            i, cid, rand_date(2022, 2026), gross, net,
            stat, random.choice(PMT_TYPES),
            f"{random.choice(first_names)} {random.choice(last_names)}",
            f"CHK-{random.randint(100000,999999)}", void_rsn
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO PAYMENT VALUES(?,?,?,?,?,?,?,?,?,?)", payments)

    conn.commit()
    print(f"Seeded: {n_claimants} claimants, {n_policies} policies, "
          f"{n_claims} claims, {len(payments)} payments")


if __name__ == "__main__":
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    seed(conn)
    conn.close()
    print(f"Database ready at: {DB_PATH}")


    
