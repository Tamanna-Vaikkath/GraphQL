"""
database/build_dev_db.py
========================
Builds the DEV database (database/insurance_dev.db) from the untouched base
demo database (database/insurance_demo.db).

The base file is NEVER modified — it is copied first, and every change below
is made on the copy.

What this adds on top of the four original tables
-------------------------------------------------
1. TEMPORAL DATA          CLAIM_EVENT
     One row per lifecycle event of a claim (reported, assigned, reserve
     changes, payments, litigation, pending, denied/closed, reopened...),
     so you can see how a claim EVOLVES over time. The last event of every
     claim always matches the claim's current state in CLAIMS.

2. COLLISION / OVERLAP    CLAIM_OVERLAP
     Pairs of claims that overlap: same policy and/or same claimant, with
     loss dates inside a time window. Each pair gets an OVERLAP_SCORE and a
     plain-English OVERLAP_REASON. COLLISION_FLG='Y' marks the strong ones.

3. CLUSTERS               CLAIM_CLUSTER, CLAIM_CLUSTER_MEMBER
     Groups of claims that are connected by strong links (overlaps, same
     catastrophe event, or a shared-payee ring). Computed as connected
     components, so every cluster can be explained link by link.

4. LOSS CAUSE             LOSS_CAUSE, CLAIM_LOSS_DETAIL
     A lookup of causes of loss (e.g. WATER_DAMAGE, REAR_END) and, per
     claim, its cause, a free-text loss narrative, loss state and severity.

5. "CLAIMS SIMILAR TO..." CLAIM_SIMILARITY
     For every claim, its top-5 most similar claims (same cause, narrative
     text similarity, severity, amount, LOB, state, timing) with a score and
     a plain-English MATCH_BASIS_TXT.

Synthetic data notes (read this — it is deliberate)
---------------------------------------------------
* The base demo data is fully random, so it contains almost no realistic
  overlaps, storms or fraud-style rings. This script therefore PLANTS a
  small number of synthetic scenarios as NEW claims (IDs 501+): duplicate
  submissions, repeat claimants, multi-loss policies, 5 catastrophe storms
  and 3 shared-payee rings. Detection (overlaps, clusters, similarity) is
  then computed by real algorithms over ALL claims — the planted claims are
  not labelled anywhere; they are simply found.
* The base data also has impossible dates (closed before reported, payments
  before the claim existed, dates in the future). A timeline built on those
  would be nonsense, so DATES ONLY are repaired in the dev copy
  (see align_dates()). Amounts/status/keys of base rows are not touched.

Run:
    python database/build_dev_db.py
    python database/build_dev_db.py --base database/insurance_demo.db --out database/insurance_dev.db
"""
from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).parent
DEFAULT_BASE = HERE / "insurance_demo.db"
DEFAULT_OUT = HERE / "insurance_dev.db"

# The "as of" date of the dataset. Nothing in the dev DB is dated after it.
AS_OF = date(2026, 9, 21)
SEED = 20260921

# Detection thresholds (documented in the README as well)
OVERLAP_WINDOW_DAYS = 30        # loss dates this close (same policy/claimant) => overlap
DUPLICATE_WINDOW_DAYS = 7       # ...and this close with same policy+claimant+type => duplicate suspect
COLLISION_SCORE_MIN = 0.60      # COLLISION_FLG = 'Y' at or above this
CLUSTER_LINK_SCORE_MIN = 0.50   # an overlap pair only links claims into a cluster at/above this
CAT_WINDOW_DAYS = 3             # same state + weather peril + this close => same catastrophe event
PAYEE_WINDOW_DAYS = 90          # same payee within this many days (plus other conditions) => ring link
SIMILARITY_TOP_K = 5
SIMILARITY_MIN_SCORE = 0.45

# ─────────────────────────────────────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────────────────────────────────────
# (cause code, description, parent LOSS_TYPE_CD, peril group)
LOSS_CAUSES: List[Tuple[str, str, str, str]] = [
    ("AUTO_REAR_END",     "Rear-end vehicle collision",           "AUTO",   "ACCIDENT"),
    ("AUTO_INTERSECTION", "Intersection / side-impact collision", "AUTO",   "ACCIDENT"),
    ("AUTO_THEFT",        "Vehicle theft",                        "AUTO",   "THEFT"),
    ("AUTO_HAIL",         "Hail damage to vehicle",               "AUTO",   "WEATHER"),
    ("PROP_FIRE",         "Fire / smoke damage",                  "PROP",   "FIRE"),
    ("PROP_WATER",        "Water damage / burst pipe",            "PROP",   "WATER"),
    ("PROP_WIND_HAIL",    "Wind or hail storm damage",            "PROP",   "WEATHER"),
    ("PROP_BURGLARY",     "Burglary / theft from property",       "PROP",   "THEFT"),
    ("LIAB_SLIP_FALL",    "Slip and fall on premises",            "LIAB",   "LIABILITY"),
    ("LIAB_PRODUCT",      "Product liability defect",             "LIAB",   "LIABILITY"),
    ("LIAB_PROP_DMG",     "Third-party property damage",          "LIAB",   "LIABILITY"),
    ("WC_STRAIN",         "Overexertion / strain injury",         "WC",     "INJURY"),
    ("WC_FALL",           "Fall from height or same level",       "WC",     "INJURY"),
    ("WC_STRUCK_BY",      "Struck by object",                     "WC",     "INJURY"),
    ("MAR_CARGO",         "Cargo damage in transit",              "MARINE", "TRANSIT"),
    ("MAR_STORM",         "Heavy weather / storm at sea",         "MARINE", "WEATHER"),
    ("MAR_VESSEL_COLL",   "Vessel-to-vessel collision",           "MARINE", "ACCIDENT"),
]
CAUSES_BY_TYPE: Dict[str, List[str]] = defaultdict(list)
PERIL_OF: Dict[str, str] = {}
for _c, _d, _t, _p in LOSS_CAUSES:
    CAUSES_BY_TYPE[_t].append(_c)
    PERIL_OF[_c] = _p

# Narrative building blocks per cause: (what happened, damage / outcome).
# Claims with the same cause share vocabulary, which is what makes the
# narrative-similarity score meaningful.
NARRATIVES: Dict[str, Tuple[List[str], List[str]]] = {
    "AUTO_REAR_END": (
        ["Insured vehicle was rear-ended while stopped at a red light",
         "Insured vehicle was struck from behind in heavy highway traffic",
         "Vehicle rear-ended at a stop sign by a following driver"],
        ["rear bumper and trunk damage", "taillights and bumper crushed, neck strain reported",
         "trunk and frame damage, other driver cited"]),
    "AUTO_INTERSECTION": (
        ["Side-impact collision at an intersection after a driver ran a red light",
         "Vehicle hit while turning left at a busy intersection",
         "T-bone collision at a controlled intersection"],
        ["driver door and quarter panel damage", "airbag deployed, vehicle towed",
         "front-end and side damage, police report filed"]),
    "AUTO_THEFT": (
        ["Vehicle stolen from a parking lot overnight", "Car reported stolen from residential street",
         "Theft of vehicle from a shopping center garage"],
        ["vehicle not recovered", "vehicle recovered with stripped parts",
         "police report filed, keys still with owner"]),
    "AUTO_HAIL": (
        ["Vehicle parked outdoors during a severe hailstorm",
         "Hail damage to vehicle during a regional storm",
         "Car exposed to golf-ball sized hail while parked"],
        ["dented hood and roof, cracked windshield", "multiple body dents and broken glass",
         "roof and hood dents, paintless repair not possible"]),
    "PROP_FIRE": (
        ["Kitchen fire spread to adjoining rooms", "Electrical fire started in the basement",
         "Fire damaged the roof and upper floor"],
        ["smoke and fire damage, temporary housing required", "structural damage and smoke contamination",
         "extensive fire and water damage from suppression"]),
    "PROP_WATER": (
        ["Burst pipe flooded the ground floor", "Water heater failure caused water damage",
         "Frozen pipe burst and leaked into the ceiling below"],
        ["flooring, drywall and carpet damage", "mold remediation and drywall replacement needed",
         "water damage to ceilings and contents"]),
    "PROP_WIND_HAIL": (
        ["Severe wind and hail storm damaged the roof", "Storm winds tore shingles from the roof",
         "Hail storm damaged roof, siding and windows"],
        ["roof replacement required", "shingle and gutter damage, interior water intrusion",
         "siding and window damage across the property"]),
    "PROP_BURGLARY": (
        ["Burglary while the occupants were away", "Forced entry through rear door, items stolen",
         "Break-in at the property overnight"],
        ["electronics and jewelry stolen, door frame damaged", "forced-entry damage and stolen contents",
         "police report filed, stolen property list submitted"]),
    "LIAB_SLIP_FALL": (
        ["Customer slipped on a wet floor in the entrance", "Visitor fell on an icy walkway outside the premises",
         "Patron tripped on a torn carpet edge"],
        ["fractured wrist and medical treatment", "back injury and physical therapy",
         "knee injury, attorney letter received"]),
    "LIAB_PRODUCT": (
        ["Customer injured by a defective product", "Product malfunction caused injury during normal use",
         "Allegation of a manufacturing defect in a sold product"],
        ["burn injuries and medical expenses", "recall notice issued, injury claim filed",
         "property damage and injury alleged"]),
    "LIAB_PROP_DMG": (
        ["Contractor damaged a neighbouring property during work", "Third-party property damaged by insured operations",
         "Insured's equipment damaged an adjacent building"],
        ["repair estimate submitted by the third party", "structural repair and cleanup needed",
         "third-party demands full repair cost"]),
    "WC_STRAIN": (
        ["Employee strained back lifting heavy boxes", "Overexertion injury while moving equipment",
         "Shoulder strain from repetitive lifting"],
        ["physical therapy and light duty", "lost-time injury with medical treatment",
         "back strain, doctor recommended rest"]),
    "WC_FALL": (
        ["Employee fell from a ladder at the worksite", "Worker slipped on a wet warehouse floor",
         "Fall from scaffolding during construction"],
        ["fractured ankle and lost work days", "head and back injuries, hospital treatment",
         "wrist fracture, time off work"]),
    "WC_STRUCK_BY": (
        ["Worker struck by a falling object at the site", "Employee hit by a moving forklift",
         "Struck by a swinging load during unloading"],
        ["contusions and fractured arm", "head injury and emergency treatment",
         "leg injury, extended recovery"]),
    "MAR_CARGO": (
        ["Cargo container damaged during ocean transit", "Shipment water-damaged in the vessel hold",
         "Goods spoiled after refrigeration failure in transit"],
        ["partial loss of cargo, survey report attached", "total loss of perishable goods",
         "container damage and cargo shortage"]),
    "MAR_STORM": (
        ["Vessel encountered heavy weather and lost deck cargo", "Storm at sea damaged the vessel and cargo",
         "Severe weather caused hull damage during voyage"],
        ["deck cargo lost overboard, hull damage", "cargo damage and delay costs",
         "salvage operation required"]),
    "MAR_VESSEL_COLL": (
        ["Collision between two vessels in the harbor", "Vessel struck a moored ship during docking",
         "Collision in a shipping lane in poor visibility"],
        ["hull damage and repair dry-dock", "bow damage, both vessels inspected",
         "damage to hull and navigation equipment"]),
}

EVENT_TYPES = [
    ("REPORTED", "First notice of loss received"),
    ("ASSIGNED", "Claim assigned to an adjuster"),
    ("RESERVE_CHANGE", "Reserve / incurred estimate revised"),
    ("PAYMENT_ISSUED", "A payment was issued"),
    ("PAYMENT_VOIDED", "A payment was voided"),
    ("LITIGATION_FILED", "Claim moved into litigation"),
    ("STATUS_CHANGE", "Status changed (e.g. to Pending)"),
    ("DENIED", "Claim denied"),
    ("CLOSED", "Claim closed"),
    ("REOPENED", "Closed claim reopened"),
]
RESERVE_REASONS = ["medical review", "repair estimate received", "liability assessment",
                   "expert report", "litigation counsel update", "updated damage survey"]

FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph",
               "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
RING_PAYEES = ["Apex Recovery LLC", "Metro Medical Group", "Rapid Care Clinics"]
STOP_WORDS = set("""a an the and or of to in on at by for with was were is are as it its from that this
after while during into over under be been has have had not no than then their his her""".split())


def d(s: str) -> date:
    return date.fromisoformat(s)


def iso(x: date) -> str:
    return x.isoformat()


def clamp_date(x: date, lo: date, hi: date) -> date:
    return max(lo, min(hi, x))


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
DDL = """
CREATE TABLE LOSS_CAUSE (
    LOSS_CAUSE_CD    TEXT PRIMARY KEY,
    LOSS_CAUSE_DESC  TEXT,
    LOSS_TYPE_CD     TEXT,
    PERIL_GROUP_CD   TEXT
);

CREATE TABLE CLAIM_LOSS_DETAIL (
    CLAIM_ID         INTEGER PRIMARY KEY REFERENCES CLAIMS(CLAIM_ID),
    LOSS_CAUSE_CD    TEXT REFERENCES LOSS_CAUSE(LOSS_CAUSE_CD),
    LOSS_DESC        TEXT,
    LOSS_STATE_CD    TEXT,
    SEVERITY_CD      TEXT
);

CREATE TABLE CLAIM_EVENT (
    EVENT_ID              INTEGER PRIMARY KEY,
    CLAIM_ID              INTEGER REFERENCES CLAIMS(CLAIM_ID),
    EVENT_SEQ             INTEGER,
    EVENT_DT              TEXT,
    EVENT_TYPE_CD         TEXT,
    FROM_STAT_CD          TEXT,
    TO_STAT_CD            TEXT,
    INCURRED_AMT_AT_EVENT REAL,
    RESERVE_AMT_AT_EVENT  REAL,
    ADJUSTER_ID           INTEGER,
    EVENT_NOTE            TEXT
);

CREATE TABLE CLAIM_OVERLAP (
    OVERLAP_ID           INTEGER PRIMARY KEY,
    CLAIM_ID_A           INTEGER REFERENCES CLAIMS(CLAIM_ID),
    CLAIM_ID_B           INTEGER REFERENCES CLAIMS(CLAIM_ID),
    OVERLAP_TYPE_CD      TEXT,
    DAYS_APART           INTEGER,
    SHARED_POLICY_ID     INTEGER REFERENCES POLICY(POLICY_ID),
    SHARED_CLAIMANT_ID   INTEGER REFERENCES CLAIMANT(CLAIMANT_ID),
    SAME_LOSS_TYPE_FLG   TEXT,
    PERIOD_OVERLAP_DAYS  INTEGER,
    AMT_DIFF_PCT         REAL,
    OVERLAP_SCORE        REAL,
    COLLISION_FLG        TEXT,
    OVERLAP_REASON       TEXT
);

CREATE TABLE CLAIM_CLUSTER (
    CLUSTER_ID          INTEGER PRIMARY KEY,
    CLUSTER_TYPE_CD     TEXT,
    CLUSTER_LABEL       TEXT,
    CLAIM_CNT           INTEGER,
    TOTAL_INCURRED_AMT  REAL,
    FIRST_LOSS_DT       TEXT,
    LAST_LOSS_DT        TEXT,
    SPAN_DAYS           INTEGER,
    RISK_LEVEL_CD       TEXT,
    LINK_BASIS_TXT      TEXT
);

CREATE TABLE CLAIM_CLUSTER_MEMBER (
    MEMBER_ID        INTEGER PRIMARY KEY,
    CLUSTER_ID       INTEGER REFERENCES CLAIM_CLUSTER(CLUSTER_ID),
    CLAIM_ID         INTEGER REFERENCES CLAIMS(CLAIM_ID),
    LINK_DEGREE      INTEGER,
    ANCHOR_FLG       TEXT
);

CREATE TABLE CLAIM_SIMILARITY (
    SIMILARITY_ID     INTEGER PRIMARY KEY,
    CLAIM_ID          INTEGER REFERENCES CLAIMS(CLAIM_ID),
    SIMILAR_CLAIM_ID  INTEGER REFERENCES CLAIMS(CLAIM_ID),
    SIMILARITY_RANK   INTEGER,
    SIMILARITY_SCORE  REAL,
    MATCH_BASIS_TXT   TEXT
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — repair impossible dates in the dev copy
# ─────────────────────────────────────────────────────────────────────────────
def align_dates(conn: sqlite3.Connection, rng: random.Random) -> Dict[str, int]:
    """Make dates logically consistent. Only date columns of CLAIMS/PAYMENT are
    touched, and only rows that are impossible:
      * CLOSE_DT before REPORT_DT+3 or after AS_OF
      * PMT_DT before the claim was reported, after it closed, or after AS_OF
    """
    fixed = {"close_dt": 0, "pmt_dt": 0}
    claims = conn.execute(
        "SELECT CLAIM_ID, CLM_STAT_CD, REPORT_DT, CLOSE_DT FROM CLAIMS ORDER BY CLAIM_ID").fetchall()
    close_of: Dict[int, Optional[date]] = {}
    report_of: Dict[int, date] = {}
    for cid, stat, rep, close in claims:
        rep_d = d(rep)
        report_of[cid] = rep_d
        new_close = d(close) if close else None
        if stat in ("C", "D"):
            if new_close is None or new_close < rep_d + timedelta(days=3) or new_close > AS_OF:
                new_close = min(AS_OF, rep_d + timedelta(days=rng.randint(30, 540)))
                fixed["close_dt"] += 1
                conn.execute("UPDATE CLAIMS SET CLOSE_DT=? WHERE CLAIM_ID=?", (iso(new_close), cid))
        close_of[cid] = new_close

    pays = conn.execute("SELECT PAYMENT_ID, CLAIM_ID, PMT_DT FROM PAYMENT ORDER BY PAYMENT_ID").fetchall()
    for pid, cid, pdt in pays:
        lo = report_of[cid] + timedelta(days=1)
        hi = close_of[cid] or AS_OF
        hi = min(hi, AS_OF)
        if hi < lo:
            hi = lo
        cur = d(pdt)
        if cur < lo or cur > hi:
            new = lo + timedelta(days=rng.randint(0, (hi - lo).days))
            conn.execute("UPDATE PAYMENT SET PMT_DT=? WHERE PAYMENT_ID=?", (iso(new), pid))
            fixed["pmt_dt"] += 1
    conn.commit()
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — loss cause / detail
# ─────────────────────────────────────────────────────────────────────────────
def severity_of(incurred: float) -> str:
    if incurred < 50_000:
        return "MINOR"
    if incurred < 150_000:
        return "MODERATE"
    if incurred < 300_000:
        return "MAJOR"
    return "SEVERE"


def make_narrative(cause: str, rng: random.Random) -> str:
    whats, dmgs = NARRATIVES[cause]
    return f"{rng.choice(whats)}; {rng.choice(dmgs)}."


def insert_loss_detail(conn, rng, claim_id, loss_type, incurred, state, cause=None):
    cause = cause or rng.choice(CAUSES_BY_TYPE[loss_type])
    conn.execute(
        "INSERT INTO CLAIM_LOSS_DETAIL VALUES (?,?,?,?,?)",
        (claim_id, cause, make_narrative(cause, rng), state, severity_of(incurred)))
    return cause


def build_loss_detail_for_base(conn, rng) -> None:
    conn.executemany("INSERT INTO LOSS_CAUSE VALUES (?,?,?,?)", LOSS_CAUSES)
    rows = conn.execute(
        "SELECT c.CLAIM_ID, c.LOSS_TYPE_CD, c.INCURRED_AMT, p.STATE_CD "
        "FROM CLAIMS c JOIN POLICY p ON p.POLICY_ID = c.POLICY_ID ORDER BY c.CLAIM_ID").fetchall()
    for cid, ltype, inc, st in rows:
        insert_loss_detail(conn, rng, cid, ltype, inc, st)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — plant synthetic scenarios as NEW claims
# ─────────────────────────────────────────────────────────────────────────────
class Planter:
    def __init__(self, conn: sqlite3.Connection, rng: random.Random):
        self.conn = conn
        self.rng = rng
        self.next_claim = conn.execute("SELECT MAX(CLAIM_ID) FROM CLAIMS").fetchone()[0] + 1
        self.next_pmt = conn.execute("SELECT MAX(PAYMENT_ID) FROM PAYMENT").fetchone()[0] + 1
        self.counts: Counter = Counter()

    # -- helpers ------------------------------------------------------------
    def _policy_state(self, pid: int) -> str:
        return self.conn.execute("SELECT STATE_CD FROM POLICY WHERE POLICY_ID=?", (pid,)).fetchone()[0]

    def add_claim(self, scenario: str, policy_id: int, claimant_id: int, loss_type: str,
                  loss_dt: date, cause: str, incurred: float, adjuster: int,
                  litigation: str = "N", state: Optional[str] = None,
                  status: Optional[str] = None, report_lag: Optional[int] = None) -> int:
        rng = self.rng
        cid = self.next_claim
        self.next_claim += 1
        loss_dt = min(loss_dt, AS_OF - timedelta(days=60))
        rep = clamp_date(loss_dt + timedelta(days=report_lag if report_lag is not None else rng.randint(1, 10)),
                         loss_dt, AS_OF - timedelta(days=30))
        stat = status or rng.choices(["O", "C", "P", "D"], weights=[30, 40, 15, 15])[0]
        close = None
        if stat in ("C", "D"):
            close = min(AS_OF, rep + timedelta(days=rng.randint(30, 300)))
        incurred = round(incurred, 2)
        reserve = round(incurred * rng.uniform(0.1, 0.6), 2) if stat in ("O", "P") else 0.0
        self.conn.execute(
            "INSERT INTO CLAIMS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, policy_id, claimant_id, stat, iso(loss_dt), iso(rep), loss_type,
             incurred, reserve, adjuster, iso(close) if close else None, litigation))
        insert_loss_detail(self.conn, rng, cid, loss_type, incurred,
                           state or self._policy_state(policy_id), cause)
        self.counts[scenario] += 1
        return cid

    def add_payment(self, claim_id: int, payee: Optional[str] = None,
                    on: Optional[date] = None) -> None:
        rng = self.rng
        rep, close, stat = self.conn.execute(
            "SELECT REPORT_DT, CLOSE_DT, CLM_STAT_CD FROM CLAIMS WHERE CLAIM_ID=?", (claim_id,)).fetchone()
        lo = d(rep) + timedelta(days=2)
        hi = min(d(close) if close else AS_OF, AS_OF)
        if hi < lo:
            hi = lo
        when = on if on and lo <= on <= hi else lo + timedelta(days=rng.randint(0, (hi - lo).days))
        gross = round(rng.uniform(500, 60000), 2)
        pstat = rng.choices(["IS", "CL", "VD", "PD"], weights=[40, 35, 10, 15])[0]
        void = rng.choice(["DUPE", "ERROR"]) if pstat == "VD" else None
        payee = payee or f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        self.conn.execute(
            "INSERT INTO PAYMENT VALUES (?,?,?,?,?,?,?,?,?,?)",
            (self.next_pmt, claim_id, iso(when), gross, round(gross * 0.9, 2), pstat,
             rng.choice(["INDEM", "MED", "EXP"]), payee, f"CHK-{rng.randint(100000, 999999)}", void))
        self.next_pmt += 1

    # -- scenarios ----------------------------------------------------------
    def duplicates(self, n: int) -> None:
        """Same policy + claimant + loss type, loss dates 0-6 days apart."""
        rng = self.rng
        pool = self.conn.execute(
            "SELECT c.CLAIM_ID, c.POLICY_ID, c.CLAIMANT_ID, c.LOSS_TYPE_CD, c.LOSS_DT, c.INCURRED_AMT, "
            "c.ADJUSTER_ID, l.LOSS_CAUSE_CD FROM CLAIMS c JOIN CLAIM_LOSS_DETAIL l USING(CLAIM_ID) "
            "WHERE c.LOSS_DT BETWEEN '2022-03-01' AND '2025-06-30' AND c.CLAIM_ID <= 500 "
            "ORDER BY c.CLAIM_ID").fetchall()
        for cid, pol, clmt, ltype, ldt, inc, adj, cause in rng.sample(pool, n):
            adj2 = adj if rng.random() < 0.5 else rng.randint(1, 30)
            new = self.add_claim("duplicate", pol, clmt, ltype, d(ldt) + timedelta(days=rng.randint(0, 6)),
                                 cause, inc * rng.uniform(0.85, 1.15), adj2)
            if rng.random() < 0.6:
                self.add_payment(new)

    def repeat_claimants(self, n: int) -> None:
        """Same claimant, DIFFERENT policy, loss 8-25 days after an existing claim."""
        rng = self.rng
        pool = self.conn.execute(
            "SELECT CLAIM_ID, POLICY_ID, CLAIMANT_ID, LOSS_TYPE_CD, LOSS_DT FROM CLAIMS "
            "WHERE LOSS_DT BETWEEN '2022-03-01' AND '2025-06-30' AND CLAIM_ID <= 500 ORDER BY CLAIM_ID").fetchall()
        n_pol = self.conn.execute("SELECT MAX(POLICY_ID) FROM POLICY").fetchone()[0]
        for cid, pol, clmt, ltype, ldt in rng.sample(pool, n):
            other = rng.choice([p for p in (rng.randint(1, n_pol) for _ in range(5)) if p != pol])
            cause = rng.choice(CAUSES_BY_TYPE[ltype])
            new = self.add_claim("repeat_claimant", other, clmt, ltype,
                                 d(ldt) + timedelta(days=rng.randint(8, 25)), cause,
                                 rng.uniform(5000, 250000), rng.randint(1, 30))
            if rng.random() < 0.5:
                self.add_payment(new)

    def multi_loss_policies(self, n: int) -> None:
        """Same policy, different claimant, loss 8-28 days after an existing claim."""
        rng = self.rng
        pool = self.conn.execute(
            "SELECT CLAIM_ID, POLICY_ID, CLAIMANT_ID, LOSS_TYPE_CD, LOSS_DT FROM CLAIMS "
            "WHERE LOSS_DT BETWEEN '2022-03-01' AND '2025-06-30' AND CLAIM_ID <= 500 ORDER BY CLAIM_ID").fetchall()
        n_clmt = self.conn.execute("SELECT MAX(CLAIMANT_ID) FROM CLAIMANT").fetchone()[0]
        for cid, pol, clmt, ltype, ldt in rng.sample(pool, n):
            other = rng.choice([c for c in (rng.randint(1, n_clmt) for _ in range(5)) if c != clmt])
            ltype2 = ltype if rng.random() < 0.6 else rng.choice(list(CAUSES_BY_TYPE))
            cause = rng.choice(CAUSES_BY_TYPE[ltype2])
            new = self.add_claim("multi_loss_policy", pol, other, ltype2,
                                 d(ldt) + timedelta(days=rng.randint(8, 28)), cause,
                                 rng.uniform(5000, 300000), rng.randint(1, 30))
            if rng.random() < 0.5:
                self.add_payment(new)

    def storms(self, n_events: int, per_event: Tuple[int, int]) -> None:
        """Catastrophe events: many weather claims in one state within ~3 days."""
        rng = self.rng
        states = rng.sample(["TX", "FL", "GA", "NC", "OH", "AZ", "IL"], n_events)
        dates = sorted(rng.sample(
            [date(2023, 3, 1) + timedelta(days=i) for i in range(0, 1000, 7)], n_events))
        for st, day in zip(states, dates):
            pols = [r[0] for r in self.conn.execute(
                "SELECT POLICY_ID FROM POLICY WHERE STATE_CD=? AND LINE_OF_BUSNSS IN "
                "('HOMEOWNERS','COMMERCIAL') ORDER BY POLICY_ID", (st,)).fetchall()]
            if not pols:
                pols = [r[0] for r in self.conn.execute(
                    "SELECT POLICY_ID FROM POLICY WHERE STATE_CD=? ORDER BY POLICY_ID", (st,)).fetchall()]
            team = rng.sample(range(1, 31), 3)          # the catastrophe adjuster team
            for _ in range(rng.randint(*per_event)):
                pol = rng.choice(pols)
                clmt = rng.randint(1, 200)
                new = self.add_claim(
                    "storm", pol, clmt, "PROP", day + timedelta(days=rng.randint(0, 2)),
                    "PROP_WIND_HAIL", rng.uniform(15000, 220000), rng.choice(team),
                    state=st, report_lag=rng.randint(1, 12))
                if rng.random() < 0.7:
                    self.add_payment(new)

    def payee_rings(self, n_rings: int, size: Tuple[int, int]) -> None:
        """Attorney-represented claimants + one adjuster + the same payee."""
        rng = self.rng
        atty = [r[0] for r in self.conn.execute(
            "SELECT CLAIMANT_ID FROM CLAIMANT WHERE ATTY_REP_FLG='Y' ORDER BY CLAIMANT_ID").fetchall()]
        n_pol = self.conn.execute("SELECT MAX(POLICY_ID) FROM POLICY").fetchone()[0]
        starts = [date(2023, 2, 1), date(2024, 1, 15), date(2024, 10, 1)]
        for ring in range(n_rings):
            members = rng.sample(atty, 4)
            adj = rng.randint(1, 30)
            payee = RING_PAYEES[ring % len(RING_PAYEES)]
            base = starts[ring % len(starts)]
            for k in range(rng.randint(*size)):
                loss = base + timedelta(days=rng.randint(0, 55))
                cause = rng.choice(["AUTO_REAR_END", "AUTO_INTERSECTION"])
                new = self.add_claim(
                    "payee_ring", rng.randint(1, n_pol), rng.choice(members), "AUTO", loss, cause,
                    rng.uniform(20000, 140000), adj, litigation="Y" if rng.random() < 0.6 else "N")
                # every ring claim pays the same payee, inside the 90-day window
                self.add_payment(new, payee=payee, on=loss + timedelta(days=rng.randint(20, 70)))
                if rng.random() < 0.4:
                    self.add_payment(new, payee=payee, on=loss + timedelta(days=rng.randint(30, 80)))


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — temporal events
# ─────────────────────────────────────────────────────────────────────────────
_PRIORITY = {"REPORTED": 0, "ASSIGNED": 1, "RESERVE_CHANGE": 2, "LITIGATION_FILED": 3,
             "PAYMENT_ISSUED": 4, "PAYMENT_VOIDED": 5, "STATUS_CHANGE": 6, "REOPENED": 7,
             "DENIED": 8, "CLOSED": 8}


def build_events(conn: sqlite3.Connection, rng: random.Random) -> int:
    claims = conn.execute(
        "SELECT CLAIM_ID, CLM_STAT_CD, LOSS_DT, REPORT_DT, INCURRED_AMT, RESERVE_AMT, ADJUSTER_ID, "
        "CLOSE_DT, LITIGATION_FLG FROM CLAIMS ORDER BY CLAIM_ID").fetchall()
    pays_by_claim: Dict[int, list] = defaultdict(list)
    for r in conn.execute("SELECT PAYMENT_ID, CLAIM_ID, PMT_DT, PMT_AMT_GROSS, PMT_STAT_CD, PMT_TYPE_CD, "
                          "PAYEE_NM, VOID_RSN_CD FROM PAYMENT ORDER BY PMT_DT, PAYMENT_ID"):
        pays_by_claim[r[1]].append(r)
    pmt_status = {"IS": "issued", "CL": "cleared", "VD": "voided", "PD": "paid"}
    pmt_type = {"INDEM": "indemnity", "MED": "medical", "EXP": "expense"}

    event_id = 1
    total = 0
    for (cid, stat, loss_dt, rep_dt, inc, res, adj, close_dt, lit) in claims:
        rep = d(rep_dt)
        end = d(close_dt) if close_dt else AS_OF
        end = max(end, rep + timedelta(days=2))
        inc0 = round(inc * rng.uniform(0.15, 0.45), 2)
        res0 = round(inc0 * rng.uniform(0.7, 1.0), 2)
        res_pre_close = round(inc * rng.uniform(0.05, 0.25), 2)

        # ---- 1. build the un-valued middle events (date + type + payload)
        mid: List[dict] = []
        assign_dt = min(rep + timedelta(days=rng.randint(0, 3)), end)
        lo = assign_dt + timedelta(days=1)
        hi = max(lo, end - timedelta(days=1)) if close_dt else end
        span = max(0, (hi - lo).days)

        def rand_day() -> date:
            return lo + timedelta(days=rng.randint(0, span))

        for _ in range(rng.randint(1, 3)):
            mid.append({"type": "RESERVE_CHANGE", "dt": rand_day()})
        if lit == "Y":
            mid.append({"type": "LITIGATION_FILED", "dt": rand_day()})
        for (pid, _c, pdt, gross, pst, ptype, payee, vrsn) in pays_by_claim.get(cid, []):
            pdt_d = clamp_date(d(pdt), lo, end)
            mid.append({"type": "PAYMENT_ISSUED", "dt": pdt_d,
                        "note": f"Payment {pid}: {pmt_type.get(ptype, ptype)} ${gross:,.2f} to {payee} "
                                f"({pmt_status.get(pst, pst)})"})
            if pst == "VD":
                mid.append({"type": "PAYMENT_VOIDED", "dt": min(end, pdt_d + timedelta(days=rng.randint(1, 10))),
                            "note": f"Payment {pid} voided, reason {vrsn or 'UNSPECIFIED'}"})
        goes_pending = stat == "P" or (stat in ("C", "D") and rng.random() < 0.35)
        if goes_pending:
            # pending always happens after the last reserve change so the final state is coherent
            mid.append({"type": "STATUS_CHANGE", "dt": rand_day(), "to": "P"})

        mid.sort(key=lambda e: (e["dt"], _PRIORITY[e["type"]]))
        if goes_pending:  # keep STATUS_CHANGE last among mid events for a clean state story
            sc = [e for e in mid if e["type"] == "STATUS_CHANGE"][0]
            mid.remove(sc)
            sc["dt"] = max(sc["dt"], mid[-1]["dt"]) if mid else sc["dt"]
            mid.append(sc)
        # guarantee the last reserve change exists and carries the final numbers
        reserve_idx = [i for i, e in enumerate(mid) if e["type"] == "RESERVE_CHANGE"]

        events: List[dict] = []
        cur_inc, cur_res, cur_stat = inc0, res0, "O"
        events.append(dict(dt=rep, type="REPORTED", frm=None, to="O", inc=cur_inc, res=cur_res, adj=None,
                           note="First notice of loss received"))
        events.append(dict(dt=assign_dt, type="ASSIGNED", frm="O", to="O", inc=cur_inc, res=cur_res, adj=adj,
                           note=f"Claim assigned to adjuster {adj}"))
        final_res_open = res if stat in ("O", "P") else res_pre_close
        for i, e in enumerate(mid):
            typ = e["type"]
            frm = cur_stat
            if typ == "RESERVE_CHANGE":
                k = reserve_idx.index(i)
                frac = 1.0 if k == len(reserve_idx) - 1 else round(0.35 + 0.6 * (k + 1) / (len(reserve_idx)), 3)
                prev_res = cur_res
                cur_inc = round(inc0 + (inc - inc0) * frac, 2) if frac < 1.0 else inc
                cur_res = round(res0 + (final_res_open - res0) * frac, 2) if frac < 1.0 else final_res_open
                note = (f"Reserve revised from ${prev_res:,.2f} to ${cur_res:,.2f} after "
                        f"{rng.choice(RESERVE_REASONS)}; incurred now ${cur_inc:,.2f}")
            elif typ == "STATUS_CHANGE":
                cur_stat = e["to"]
                note = "Claim moved to Pending (awaiting documents / decision)"
            elif typ == "LITIGATION_FILED":
                note = "Attorney involvement / litigation flagged on the claim"
            else:
                note = e.get("note", "")
            events.append(dict(dt=e["dt"], type=typ, frm=frm, to=cur_stat, inc=cur_inc, res=cur_res,
                               adj=None, note=note))

        # ---- 2. terminal events so the last state == CLAIMS current state
        if not reserve_idx:
            raise RuntimeError("every claim must have at least one reserve change")
        if stat in ("C", "D"):
            close_d = d(close_dt)
            reopen = stat == "C" and rng.random() < 0.08 and (close_d - events[-1]["dt"]).days > 25
            if reopen:
                first_close = close_d - timedelta(days=rng.randint(15, 24))
                first_close = max(first_close, events[-1]["dt"])
                events.append(dict(dt=first_close, type="CLOSED", frm=cur_stat, to="C", inc=inc, res=0.0,
                                   adj=None, note="Claim closed"))
                events.append(dict(dt=first_close + timedelta(days=rng.randint(2, 6)), type="REOPENED",
                                   frm="C", to="O", inc=inc, res=round(inc * 0.05, 2), adj=None,
                                   note="Claim reopened after new documentation was received"))
                cur_stat = "O"
            t = "DENIED" if stat == "D" else "CLOSED"
            note = "Claim denied - coverage not applicable" if stat == "D" else "Claim closed after final payment / settlement"
            events.append(dict(dt=max(close_d, events[-1]["dt"]), type=t, frm=cur_stat, to=stat, inc=inc,
                               res=0.0, adj=None, note=note))
        # ---- 3. order, number, insert (stable by date then priority)
        events.sort(key=lambda e: e["dt"])  # stable: keeps constructed order for equal dates
        for seq, e in enumerate(events, start=1):
            conn.execute(
                "INSERT INTO CLAIM_EVENT VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, cid, seq, iso(e["dt"]), e["type"], e["frm"], e["to"], e["inc"], e["res"],
                 e["adj"], e["note"]))
            event_id += 1
            total += 1
    conn.commit()
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — collision / overlap detection
# ─────────────────────────────────────────────────────────────────────────────
def build_overlaps(conn: sqlite3.Connection) -> int:
    claims = {r[0]: r for r in conn.execute(
        "SELECT c.CLAIM_ID, c.POLICY_ID, c.CLAIMANT_ID, c.LOSS_TYPE_CD, c.LOSS_DT, c.CLOSE_DT, c.INCURRED_AMT, "
        "l.LOSS_CAUSE_CD FROM CLAIMS c JOIN CLAIM_LOSS_DETAIL l USING(CLAIM_ID)")}
    by_policy: Dict[int, List[int]] = defaultdict(list)
    by_claimant: Dict[int, List[int]] = defaultdict(list)
    for cid, r in claims.items():
        by_policy[r[1]].append(cid)
        by_claimant[r[2]].append(cid)

    pairs = set()
    for grp in list(by_policy.values()) + list(by_claimant.values()):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = sorted((grp[i], grp[j]))
                pairs.add((a, b))

    rows = []
    for a, b in sorted(pairs):
        ra, rb = claims[a], claims[b]
        days = abs((d(ra[4]) - d(rb[4])).days)
        if days > OVERLAP_WINDOW_DAYS:
            continue
        same_pol = ra[1] == rb[1]
        same_clm = ra[2] == rb[2]
        same_type = ra[3] == rb[3]
        same_cause = ra[7] == rb[7]
        end_a = d(ra[5]) if ra[5] else AS_OF
        end_b = d(rb[5]) if rb[5] else AS_OF
        period_overlap = max(0, (min(end_a, end_b) - max(d(ra[4]), d(rb[4]))).days)
        diff = abs(ra[6] - rb[6]) / max(ra[6], rb[6], 1.0)
        score = (0.30 * same_pol + 0.30 * same_clm
                 + 0.20 * (1 - min(days, OVERLAP_WINDOW_DAYS) / OVERLAP_WINDOW_DAYS)
                 + 0.10 * same_type + 0.05 * same_cause + 0.05 * (1 - min(diff, 1.0)))
        score = round(score, 3)
        if same_pol and same_clm and same_type and days <= DUPLICATE_WINDOW_DAYS:
            typ = "DUPLICATE_SUSPECT"
        elif same_pol and same_clm:
            typ = "SAME_POLICY_CLAIMANT"
        elif same_pol:
            typ = "SAME_POLICY"
        else:
            typ = "SAME_CLAIMANT"
        shared = []
        if same_pol:
            shared.append(f"policy {ra[1]}")
        if same_clm:
            shared.append(f"claimant {ra[2]}")
        reason = (f"Claims {a} and {b} share {' and '.join(shared)}; losses {days} day(s) apart"
                  f"{', same loss type ' + ra[3] if same_type else ''}"
                  f"{', same cause ' + ra[7] if same_cause else ''}; "
                  f"incurred amounts differ by {diff:.0%}; open periods overlap {period_overlap} day(s).")
        rows.append((a, b, typ, days, ra[1] if same_pol else None, ra[2] if same_clm else None,
                     "Y" if same_type else "N", period_overlap, round(diff, 3), score,
                     "Y" if score >= COLLISION_SCORE_MIN else "N", reason))
    for i, r in enumerate(rows, start=1):
        conn.execute("INSERT INTO CLAIM_OVERLAP VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (i,) + r)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — cluster identification (connected components over strong links)
# ─────────────────────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self):
        self.p: Dict[int, int] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_clusters(conn: sqlite3.Connection) -> int:
    claims = {r[0]: r for r in conn.execute(
        "SELECT c.CLAIM_ID, c.LOSS_TYPE_CD, c.LOSS_DT, c.INCURRED_AMT, c.ADJUSTER_ID, c.LITIGATION_FLG, "
        "l.LOSS_CAUSE_CD, l.LOSS_STATE_CD, cl.ATTY_REP_FLG "
        "FROM CLAIMS c JOIN CLAIM_LOSS_DETAIL l USING(CLAIM_ID) "
        "JOIN CLAIMANT cl ON cl.CLAIMANT_ID = c.CLAIMANT_ID")}
    links: List[Tuple[int, int, str]] = []

    # Link A — strong overlap pairs
    for a, b in conn.execute(
            "SELECT CLAIM_ID_A, CLAIM_ID_B FROM CLAIM_OVERLAP WHERE OVERLAP_SCORE >= ?",
            (CLUSTER_LINK_SCORE_MIN,)):
        links.append((a, b, "OVERLAP"))

    # Link B — same catastrophe event: same state + weather peril + within CAT_WINDOW_DAYS
    weather = [c for c, r in claims.items() if PERIL_OF.get(r[6]) == "WEATHER"]
    for i in range(len(weather)):
        for j in range(i + 1, len(weather)):
            a, b = weather[i], weather[j]
            if (claims[a][7] == claims[b][7]
                    and abs((d(claims[a][2]) - d(claims[b][2])).days) <= CAT_WINDOW_DAYS):
                links.append((min(a, b), max(a, b), "CAT_EVENT"))

    # Link C — payee ring: same payee, same loss type, payments within window, and
    # (same adjuster OR both claimants attorney-represented)
    pays: Dict[str, List[Tuple[int, date]]] = defaultdict(list)
    for cid, payee, pdt in conn.execute("SELECT CLAIM_ID, PAYEE_NM, PMT_DT FROM PAYMENT WHERE PAYEE_NM IS NOT NULL"):
        pays[payee.strip().lower()].append((cid, d(pdt)))
    seen = set()
    for payee, lst in pays.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (a, da), (b, db) = lst[i], lst[j]
                if a == b or (min(a, b), max(a, b)) in seen:
                    continue
                ca, cb = claims[a], claims[b]
                if (ca[1] == cb[1] and abs((da - db).days) <= PAYEE_WINDOW_DAYS
                        and (ca[4] == cb[4] or (ca[8] == "Y" and cb[8] == "Y"))):
                    seen.add((min(a, b), max(a, b)))
                    links.append((min(a, b), max(a, b), "PAYEE_RING"))

    uf = UnionFind()
    for a, b, _ in links:
        uf.union(a, b)
    comps: Dict[int, set] = defaultdict(set)
    for a, b, _ in links:
        comps[uf.find(a)].update((a, b))
    link_types: Dict[int, Counter] = defaultdict(Counter)
    degree: Dict[int, Counter] = defaultdict(Counter)
    for a, b, t in links:
        r = uf.find(a)
        link_types[r][t] += 1
        degree[r][a] += 1
        degree[r][b] += 1

    clusters = []
    for root, members in comps.items():
        lt = link_types[root]
        if lt.get("CAT_EVENT", 0) and not lt.get("OVERLAP") and not lt.get("PAYEE_RING") and len(members) < 3:
            continue                       # two weather claims are a coincidence, not an event
        ids = sorted(members)
        if lt.get("PAYEE_RING", 0) >= 3:
            ctype = "PAYEE_RING"
        elif lt.get("CAT_EVENT", 0) >= max(lt.get("OVERLAP", 0), 1) and len(members) >= 3:
            ctype = "CAT_EVENT"
        else:
            ctype = "OVERLAP_GROUP"
        first = min(d(claims[c][2]) for c in ids)
        last = max(d(claims[c][2]) for c in ids)
        total = sum(claims[c][3] for c in ids)
        lit_share = sum(1 for c in ids if claims[c][5] == "Y") / len(ids)
        if ctype == "PAYEE_RING":
            risk = "HIGH"
        elif ctype == "CAT_EVENT":
            risk = "HIGH" if len(ids) >= 8 else "MEDIUM"
        else:
            risk = "MEDIUM" if (len(ids) >= 3 or lt.get("OVERLAP", 0) and lit_share >= 0.5) else "LOW"
        if total >= 1_000_000 and risk == "LOW":
            risk = "MEDIUM"
        states = Counter(claims[c][7] for c in ids).most_common(1)[0][0]
        causes = Counter(claims[c][6] for c in ids).most_common(1)[0][0]
        label = {
            "CAT_EVENT": f"Catastrophe event: {causes} in {states} around {first.isoformat()}",
            "PAYEE_RING": f"Shared-payee ring ({len(ids)} claims), first loss {first.isoformat()}",
            "OVERLAP_GROUP": f"Overlapping claims group ({len(ids)} claims), first loss {first.isoformat()}",
        }[ctype]
        basis = "; ".join(f"{k.lower()}={v}" for k, v in sorted(lt.items()))
        clusters.append((ctype, label, ids, round(total, 2), first, last, risk, basis,
                         {c: degree[root][c] for c in ids}))

    clusters.sort(key=lambda c: (c[4], c[2][0]))
    member_id = 1
    for cluster_id, (ctype, label, ids, total, first, last, risk, basis, deg) in enumerate(clusters, start=1):
        conn.execute("INSERT INTO CLAIM_CLUSTER VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (cluster_id, ctype, label, len(ids), total, iso(first), iso(last),
                      (last - first).days, risk, basis))
        anchor = max(ids, key=lambda c: (deg[c], -c))
        for c in ids:
            conn.execute("INSERT INTO CLAIM_CLUSTER_MEMBER VALUES (?,?,?,?,?)",
                         (member_id, cluster_id, c, deg[c], "Y" if c == anchor else "N"))
            member_id += 1
    conn.commit()
    return len(clusters)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — "claims similar to ..."
# ─────────────────────────────────────────────────────────────────────────────
def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP_WORDS]


def build_similarity(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT c.CLAIM_ID, c.LOSS_TYPE_CD, c.LOSS_DT, c.INCURRED_AMT, c.LITIGATION_FLG, "
        "l.LOSS_CAUSE_CD, l.LOSS_DESC, l.LOSS_STATE_CD, l.SEVERITY_CD, p.LINE_OF_BUSNSS "
        "FROM CLAIMS c JOIN CLAIM_LOSS_DETAIL l USING(CLAIM_ID) JOIN POLICY p ON p.POLICY_ID=c.POLICY_ID "
        "ORDER BY c.CLAIM_ID").fetchall()
    n = len(rows)
    toks = {r[0]: Counter(_tokens(r[6])) for r in rows}
    df = Counter()
    for t in toks.values():
        df.update(t.keys())
    idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
    vec = {}
    for cid, t in toks.items():
        v = {w: tf * idf[w] for w, tf in t.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vec[cid] = {w: x / norm for w, x in v.items()}

    def cos(a: int, b: int) -> float:
        va, vb = vec[a], vec[b]
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(x * vb.get(w, 0.0) for w, x in va.items())

    sev_ord = {"MINOR": 0, "MODERATE": 1, "MAJOR": 2, "SEVERE": 3}
    by_type: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r[1]].append(r)

    out = []
    for ltype, grp in by_type.items():
        for ra in grp:
            scored = []
            for rb in grp:
                if ra[0] == rb[0]:
                    continue
                f_cause = 1.0 if ra[5] == rb[5] else 0.0
                f_text = cos(ra[0], rb[0])
                f_sev = 1 - abs(sev_ord[ra[8]] - sev_ord[rb[8]]) / 3
                f_amt = 1 - min(1.0, abs(math.log(max(ra[3], 1) / max(rb[3], 1))) / math.log(5))
                f_lob = 1.0 if ra[9] == rb[9] else 0.0
                f_state = 1.0 if ra[7] == rb[7] else 0.0
                f_lit = 1.0 if ra[4] == rb[4] else 0.0
                f_time = 1 - min(abs((d(ra[2]) - d(rb[2])).days), 365) / 365
                score = (0.30 * f_cause + 0.30 * f_text + 0.10 * f_sev + 0.10 * f_amt
                         + 0.05 * f_lob + 0.05 * f_state + 0.03 * f_lit + 0.07 * f_time)
                if score < SIMILARITY_MIN_SCORE:
                    continue
                basis = []
                if f_cause:
                    basis.append(f"same cause ({ra[5]})")
                if f_text >= 0.35:
                    basis.append(f"similar loss narrative ({f_text:.2f})")
                if f_sev == 1.0:
                    basis.append(f"same severity ({ra[8]})")
                if f_amt >= 0.8:
                    basis.append("similar incurred amount")
                if f_state:
                    basis.append(f"same state ({ra[7]})")
                if f_lob:
                    basis.append(f"same line of business ({ra[9]})")
                if f_time >= 0.9:
                    basis.append("losses close in time")
                scored.append((score, rb[0], "; ".join(basis) or "weak overall similarity"))
            scored.sort(key=lambda x: (-x[0], x[1]))
            for rank, (score, other, basis) in enumerate(scored[:SIMILARITY_TOP_K], start=1):
                out.append((ra[0], other, rank, round(score, 3), basis))
    for i, r in enumerate(out, start=1):
        conn.execute("INSERT INTO CLAIM_SIMILARITY VALUES (?,?,?,?,?,?)", (i,) + r)
    conn.commit()
    return len(out)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def build(base: Path, out: Path) -> Dict[str, int]:
    if not base.exists():
        raise FileNotFoundError(f"Base database not found: {base}")
    if out.resolve() == base.resolve():
        raise ValueError("Refusing to write over the base database. Choose a different --out path.")
    if out.exists():
        out.unlink()
    shutil.copyfile(base, out)               # the base file is never opened for writing

    rng = random.Random(SEED)
    conn = sqlite3.connect(out)
    conn.execute("PRAGMA foreign_keys=OFF")
    stats: Dict[str, int] = {}
    try:
        conn.executescript(DDL)
        fixed = align_dates(conn, rng)
        stats["dates_fixed_close_dt"] = fixed["close_dt"]
        stats["dates_fixed_pmt_dt"] = fixed["pmt_dt"]
        build_loss_detail_for_base(conn, rng)

        pl = Planter(conn, rng)
        pl.duplicates(24)
        pl.repeat_claimants(10)
        pl.multi_loss_policies(10)
        pl.storms(5, (8, 11))
        pl.payee_rings(3, (5, 7))
        conn.commit()
        stats.update({f"planted_{k}": v for k, v in pl.counts.items()})

        stats["events"] = build_events(conn, rng)
        stats["overlaps"] = build_overlaps(conn)
        stats["clusters"] = build_clusters(conn)
        stats["similarity_rows"] = build_similarity(conn)
        for t in ("CLAIMANT", "POLICY", "CLAIMS", "PAYMENT", "LOSS_CAUSE", "CLAIM_LOSS_DETAIL", "CLAIM_EVENT",
                  "CLAIM_OVERLAP", "CLAIM_CLUSTER", "CLAIM_CLUSTER_MEMBER", "CLAIM_SIMILARITY"):
            stats[f"rows_{t}"] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        conn.execute("VACUUM")
    finally:
        conn.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the DEV database from the base demo database.")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    stats = build(args.base, args.out)
    print(f"\nDev database written to: {args.out}\n")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
