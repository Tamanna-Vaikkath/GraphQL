"""
data/generate_gl_data.py  — deterministic synthetic Commercial GL claims data.

Writes:   database/gl_claims.db            (SQLite; the migration notebook copies these tables to Delta)
          database/gl_ground_truth.json    (expected answers to CQ1-CQ7, computed with plain SQL,
                                            INDEPENDENTLY of the graph - used by tests/run_cq_tests.py)

Planted scenarios (each has DECOYS just outside the threshold so the tests prove the logic, not luck):
  CQ1  ATT-0001 has 14 open suits (qualifies); ATT-0002 has exactly 10 (does not: needs > 10)
  CQ2  MP-0001: 9 soft-tissue claims at 9 insureds, in window (qualifies)
       MP-0002: 5 claims (not > 5) | MP-0003: 7 claims at ONE insured | MP-0004: 8 claims BEFORE the 24-month window
  CQ3  10 claims mention a pre-existing condition: 8 with confidence >= 0.7, 2 low-confidence decoys
  CQ4  Hero feature FEA-1001-BI: reserve 25k -> 60k -> 185k, the 60k revision was BACKDATED (recorded after its validFrom)
  CQ5  LOC-001 (4 occurrences), LOC-002 (3), LOC-003 (3) qualify; every other location has <= 2
  CQ6  every payment and extracted fact carries an ingestion run
  CQ7  6 occurrences have BOTH a TPA-administered claim and a carrier-handled claim

Run:  python data/generate_gl_data.py
"""
import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "database" / "gl_claims.db"
GT = ROOT / "database" / "gl_ground_truth.json"
R = random.Random(20261008)
AS_OF = date(2026, 9, 30)
SINCE_24M = date(2024, 9, 30)          # "last 24 months" window start for CQ2

T = {k: [] for k in ["POLICY", "BROKER", "INSURED", "INSURED_LOCATION", "POLICY_LOCATION", "COVERAGE_PART", "CLAIM",
                     "OCCURRENCE", "LITIGATION", "FEATURE", "CLAIMANT", "ATTORNEY", "MEDICAL_PROVIDER", "ADJUSTER",
                     "TPA", "INJURY", "RESERVE_STATE", "PAYMENT", "RECOVERY", "DOCUMENT", "TEXT_CHUNK",
                     "EXTRACTED_FACT", "INGESTION_RUN"]}
iso = lambda d: d.isoformat()
counter = {}


def nid(prefix, width=4):
    counter[prefix] = counter.get(prefix, 0) + 1
    return f"{prefix}-{counter[prefix]:0{width}d}"


def rdate(a, b):
    return a + timedelta(days=R.randint(0, max(0, (b - a).days)))


# ---------------------------------------------------------------- reference data
FIRST = ["Dana", "Marcus", "Priya", "Luis", "Hannah", "Omar", "Grace", "Tyrone", "Mei", "Connor", "Sofia", "Andre", "Nora",
         "Victor", "Alicia", "Ethan", "Rosa", "Kenji", "Leah", "Samir", "Tessa", "Jamal", "Ingrid", "Paulo", "Wendy"]
LAST = ["Whitfield", "Okafor", "Nair", "Ramirez", "Brooks", "Haddad", "Chen", "Walker", "Ito", "Doyle", "Alvarez", "Price",
        "Lindqvist", "Moreau", "Santos", "Kovacs", "Patel", "Reed", "Sato", "Bennett", "Cole", "Nguyen", "Fischer", "Diaz"]
BIZ_A = ["Maple Grove", "Riverside", "Summit", "Harbor", "Cedar Point", "Lakeview", "Ironwood", "Brightside", "Pinecrest",
         "Union Square", "Sunrise", "Oakmont", "Bayfront", "Granite", "Willow Creek"]
BIZ_B = ["Diner", "Fitness Club", "Hardware", "Plaza Hotel", "Cinema", "Market", "Auto Care", "Bowling Lanes", "Garden Centre",
         "Medical Plaza", "Bakery", "Storage", "Mall", "Pizzeria", "Furniture"]
CITIES = [("Columbus", "OH"), ("Austin", "TX"), ("Tampa", "FL"), ("Phoenix", "AZ"), ("Denver", "CO"), ("Atlanta", "GA"),
          ("Chicago", "IL"), ("Newark", "NJ"), ("Seattle", "WA"), ("Charlotte", "NC")]
OFFICES = ["Chicago", "Dallas", "Atlanta", "Newark", "Phoenix"]
CAUSES = {
    "slip_and_fall": "slipped on a wet floor near the entrance",
    "trip_hazard": "tripped over an uneven threshold",
    "falling_object": "was struck by a falling display item",
    "product_liability": "was injured by a defective product sold on site",
    "food_borne": "reported illness after eating at the premises",
    "parking_lot": "fell in the parking lot after a pothole went unrepaired",
    "dog_bite": "was bitten by a customer's dog on the premises",
    "assault": "was injured in an altercation on the premises",
}
SOFT = [("S33.5XXA", "lumbar spine", "Sprain of ligaments of lumbar spine"), ("S13.4XXA", "cervical spine", "Cervical sprain"),
        ("S39.012A", "lower back", "Strain of muscle of lower back"), ("S46.011A", "shoulder", "Rotator cuff strain")]
OTHER = [("S52.501A", "wrist", "fracture", "Fracture of lower end of radius"), ("S82.201A", "tibia", "fracture", "Fracture of tibia"),
         ("S61.411A", "hand", "laceration", "Laceration of hand"), ("S06.0X0A", "head", "head_injury", "Concussion")]
STAGES_OPEN = ["filed", "discovery", "trial"]
COURTS = [("Franklin County Common Pleas", "Franklin County, OH"), ("Travis County District Court", "Travis County, TX"),
          ("Hillsborough Circuit Court", "Hillsborough County, FL"), ("Cook County Circuit Court", "Cook County, IL")]

# ---------------------------------------------------------------- ingestion runs (monthly source loads + 1 extraction)
d = date(2024, 1, 1)
RUNS = {}
while d <= AS_OF:
    rid = f"RUN-CLM-{d:%Y%m}"
    RUNS[d.strftime("%Y-%m")] = rid
    T["INGESTION_RUN"].append(dict(RUN_ID=rid, SOURCE_SYSTEM="ClaimsCore", RUN_TYPE="source_load",
                                   STARTED_AT=iso(d + timedelta(days=27)), ONTOLOGY_VERSION="0.2.0"))
    d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
EXT_RUN = "RUN-EXT-001"
T["INGESTION_RUN"].append(dict(RUN_ID=EXT_RUN, SOURCE_SYSTEM="DocIntelligence", RUN_TYPE="text_extraction",
                               STARTED_AT="2026-10-01", ONTOLOGY_VERSION="0.2.0"))


def run_for(dt):
    return RUNS[dt.strftime("%Y-%m")] if dt.strftime("%Y-%m") in RUNS else RUNS["2026-09"]


# ---------------------------------------------------------------- parties & contract domain
for i in range(1, 9):
    T["BROKER"].append(dict(BROKER_ID=f"BRK-{i:02d}", NAME=f"{R.choice(LAST)} {R.choice(['Risk Partners', 'Insurance Group', 'Brokerage'])}"))
for i in range(1, 13):
    T["ADJUSTER"].append(dict(ADJUSTER_ID=f"ADJ-{i:03d}", NAME=f"{R.choice(FIRST)} {R.choice(LAST)}", OFFICE=R.choice(OFFICES)))
for i, n in enumerate(["Acme Claims Administration", "Northstar TPA", "Pinnacle Claim Services"], 1):
    T["TPA"].append(dict(TPA_ID=f"TPA-{i:03d}", NAME=n, TPA_CODE=f"TPA{i:02d}"))

ATT_NAMES = {1: ("Marcus Hargrove", "Hargrove & Pike LLP"), 2: ("Elena Voss", "Voss Injury Law")}
for i in range(1, 31):
    nm, firm = ATT_NAMES.get(i, (f"{R.choice(FIRST)} {R.choice(LAST)}", f"{R.choice(LAST)} {R.choice(['Law Group', 'Associates', 'LLP'])}"))
    T["ATTORNEY"].append(dict(ATTORNEY_ID=f"ATT-{i:04d}", NAME=nm, FIRM=firm, BAR_ID=f"BAR{R.randint(100000, 999999)}"))
MP_NAMES = {1: "Lakeshore Orthopedic & Spine", 2: "Metro Pain Relief Clinic", 3: "Eastside Chiropractic Centre", 4: "Valley Wellness & Rehab"}
for i in range(1, 26):
    T["MEDICAL_PROVIDER"].append(dict(PROVIDER_ID=f"MP-{i:04d}", NAME=MP_NAMES.get(i, f"{R.choice(LAST)} {R.choice(['Family Clinic', 'Physical Therapy', 'Urgent Care', 'Orthopedics'])}"),
                                      NPI=str(R.randint(1000000000, 1999999999)), SPECIALTY=R.choice(["orthopedics", "chiropractic", "physical therapy", "pain management", "urgent care"])))

N_INS = 60
insured_loc, insured_pol = {}, {}
for i in range(1, N_INS + 1):
    iid = f"INS-{i:03d}"
    T["INSURED"].append(dict(INSURED_ID=iid, NAME=f"{R.choice(BIZ_A)} {R.choice(BIZ_B)}"))
    insured_loc[iid], insured_pol[iid] = [], []
    for k in range(R.choice([1, 1, 2])):
        lid = f"LOC-{len(T['INSURED_LOCATION']) + 1:03d}"
        city, st = R.choice(CITIES)
        T["INSURED_LOCATION"].append(dict(LOCATION_ID=lid, INSURED_ID=iid, ADDRESS=f"{R.randint(10, 9800)} {R.choice(['Main', 'Oak', 'Market', 'Lake', 'Park'])} St",
                                          CITY=city, STATE=st, LAT=round(R.uniform(26, 47), 4), LONG=round(R.uniform(-122, -74), 4),
                                          OCCUPANCY_CLASS=R.choice(["restaurant", "retail", "fitness", "hotel", "warehouse", "office"]),
                                          GEO_RISK_SCORE=round(R.uniform(1, 10), 1)))
        insured_loc[iid].append(lid)
    years = R.choice([[2024, 2025, 2026], [2024, 2025], [2025, 2026], [2024], [2025], [2026]]) if i > 12 else [2024, 2025, 2026]
    for y in years:
        pid = f"POL-{len(T['POLICY']) + 1:04d}"
        T["POLICY"].append(dict(POLICY_ID=pid, POLICY_NUMBER=f"CGL{y}{len(T['POLICY']) + 1:05d}", FORM="claims_made" if R.random() < 0.1 else "occurrence",
                                EFFECTIVE_DATE=f"{y}-01-01", EXPIRY_DATE=f"{y + 1}-01-01", LIMIT_PER_OCCURRENCE=R.choice([1000000, 2000000]),
                                RETENTION=R.choice([5000, 10000, 25000]), BROKER_ID=f"BRK-{R.randint(1, 8):02d}", INSURED_ID=iid))
        insured_pol[iid].append((pid, y))
        for lid in insured_loc[iid]:
            T["POLICY_LOCATION"].append(dict(POLICY_ID=pid, LOCATION_ID=lid))
        for code, label, lim, ded in [("A", "Bodily Injury & Property Damage", 1000000, 5000), ("B", "Personal & Advertising Injury", 1000000, 0), ("C", "Medical Payments", 5000, 0)]:
            T["COVERAGE_PART"].append(dict(COVERAGE_ID=f"COV-{pid}-{code}", POLICY_ID=pid, CODE=code, LABEL=label, LIMIT_AMT=lim, DEDUCTIBLE=ded))


def policy_for(insured, dt):
    for pid, y in insured_pol[insured]:
        if y == dt.year:
            return pid
    return None


def insureds_with_policy_in(year):
    return [i for i, p in insured_pol.items() if any(y == year for _, y in p)]


# ---------------------------------------------------------------- claim factory
state = {"att_open": {}, "claimants": 0}
claim_meta = {}          # claim_id -> dict(insured, location, attorney, provider, hero etc.)


def new_claimant(attorney=None, provider=None):
    cid = f"CLT-{len(T['CLAIMANT']) + 1:04d}"
    T["CLAIMANT"].append(dict(CLAIMANT_ID=cid, NAME=f"{R.choice(FIRST)} {R.choice(LAST)}", STATE=R.choice(CITIES)[1],
                              RESOLUTION_CONFIDENCE=round(R.uniform(0.85, 1.0), 2), ATTORNEY_ID=attorney, PROVIDER_ID=provider))
    return cid


HOT = {"LOC-001": 4, "LOC-002": 3, "LOC-003": 3}
occ_count = {}


def pick_location(insured):
    """Default location for an insured: never a 'hot' CQ5 location; at most 2 occurrences per location."""
    cands = [l for l in insured_loc[insured] if l not in HOT and occ_count.get(l, 0) < 2]
    if cands:
        return R.choice(cands)
    lid = f"LOC-{len(T['INSURED_LOCATION']) + 1:03d}"            # insured needs another premises
    city, st = R.choice(CITIES)
    T["INSURED_LOCATION"].append(dict(LOCATION_ID=lid, INSURED_ID=insured, ADDRESS=f"{R.randint(10, 9800)} {R.choice(['Main', 'Oak', 'Market', 'Lake', 'Park'])} St",
                                      CITY=city, STATE=st, LAT=round(R.uniform(26, 47), 4), LONG=round(R.uniform(-122, -74), 4),
                                      OCCUPANCY_CLASS=R.choice(["restaurant", "retail", "fitness", "hotel", "warehouse", "office"]), GEO_RISK_SCORE=round(R.uniform(1, 10), 1)))
    insured_loc[insured].append(lid)
    for pid, _y in insured_pol[insured]:
        T["POLICY_LOCATION"].append(dict(POLICY_ID=pid, LOCATION_ID=lid))
    return lid


def new_occurrence(insured, loc, loss_dt, cause=None, desc=None):
    occ_count[loc] = occ_count.get(loc, 0) + 1
    oid = f"OCC-{len(T['OCCURRENCE']) + 1:04d}"
    cause = cause or R.choice(list(CAUSES))
    T["OCCURRENCE"].append(dict(OCCURRENCE_ID=oid, LOSS_DATE=iso(loss_dt), CAUSE_OF_LOSS=cause, JURISDICTION=R.choice(CITIES)[1],
                                LOSS_DESCRIPTION=desc or f"Claimant {CAUSES[cause]}.", LOCATION_ID=loc))
    return oid


def add_reserves(fid, loss_dt, plan=None, closed=False):
    """plan: list of (amount, valid_from, recorded_at). Default: 1-3 revisions."""
    if plan is None:
        base = R.choice([5000, 10000, 15000, 25000, 40000])
        plan, d0, amt = [], loss_dt + timedelta(days=R.randint(1, 10)), base
        for _ in range(R.choice([1, 2, 2, 3])):
            if d0 > AS_OF:
                break
            plan.append((amt, d0, d0))
            d0 += timedelta(days=R.randint(25, 120))
            amt = int(amt * R.uniform(1.3, 2.4) / 500) * 500
    plan = sorted(plan, key=lambda p: p[1])
    for k, (amt, vf, rec) in enumerate(plan):
        vt = plan[k + 1][1] if k + 1 < len(plan) else None
        T["RESERVE_STATE"].append(dict(RESERVE_ID=f"RSV-{len(T['RESERVE_STATE']) + 1:05d}", FEATURE_ID=fid, RESERVE_TYPE="indemnity", AMOUNT=float(amt),
                                       VALID_FROM=iso(vf), VALID_TO=iso(vt) if vt else None, RECORDED_AT=iso(rec),
                                       IS_CURRENT=1 if vt is None else 0, SOURCE_TXN=f"TXN-{R.randint(100000, 999999)}", RUN_ID=run_for(rec)))
    return plan


def add_payments(fid, loss_dt, payees, n):
    for _ in range(n):
        pd_ = loss_dt + timedelta(days=R.randint(20, 300))
        if pd_ > AS_OF:
            continue
        T["PAYMENT"].append(dict(PAYMENT_ID=f"PAY-{len(T['PAYMENT']) + 1:05d}", FEATURE_ID=fid, PAYEE_ID=R.choice(payees),
                                 AMOUNT=float(R.choice([1500, 2400, 3800, 5200, 7500, 12000, 18000])), CHECK_NUMBER=f"{R.randint(100000, 999999)}",
                                 PAYMENT_DATE=iso(pd_), COST_CATEGORY=R.choice(["indemnity", "indemnity", "medical", "alae"]), RUN_ID=run_for(pd_)))


def make_claim(insured, loss_dt, *, location=None, occurrence=None, attorney=None, provider=None, suit=None, injury="soft", cause=None,
               tpa=False, claim_id=None, bi_status="open", feature_types=("BI",), reserve_plan=None, desc=None, claimant=None):
    """Create claim + occurrence(optional) + features + claimant + injury + reserves + payments (+ litigation)."""
    pol = policy_for(insured, loss_dt)
    assert pol, (insured, loss_dt)
    loc = location or pick_location(insured)
    occ = occurrence or new_occurrence(insured, loc, loss_dt, cause, desc)
    cid = claim_id or f"CLM-GL-{2000 + len(T['CLAIM'])}"
    rep = loss_dt + timedelta(days=R.randint(1, 20))
    clt = claimant or new_claimant(attorney, provider)
    adj, tpa_id = (None, R.choice(["TPA-001", "TPA-002", "TPA-003"])) if tpa else (R.choice(T["ADJUSTER"])["ADJUSTER_ID"], None)
    T["CLAIM"].append(dict(CLAIM_ID=cid, CLAIM_NUMBER=f"GL-{cid[-4:]}-{rep.year % 100}", STATUS="open" if bi_status == "open" else "closed", LOB="CGL",
                           REPORTED_DATE=iso(rep), HANDLING_OFFICE=R.choice(OFFICES), SUMMARY=(desc or CAUSES.get(T['OCCURRENCE'][-1]['CAUSE_OF_LOSS'], "")),
                           POLICY_ID=pol, OCCURRENCE_ID=occ, ADJUSTER_ID=adj, TPA_ID=tpa_id))
    # claimant sustained injury
    if injury:
        if injury == "soft":
            icd, part, ttl = R.choice(SOFT)
            cls_ = "soft_tissue"
        else:
            icd, part, cls_, ttl = R.choice(OTHER)
        T["INJURY"].append(dict(INJURY_ID=f"INJ-{len(T['INJURY']) + 1:04d}", CLAIMANT_ID=clt, ICD10=icd, BODY_PART=part, INJURY_CLASS=cls_,
                                SEVERITY_BAND=R.choice(["minor", "moderate", "moderate", "severe"])))
    fids = []
    for ft in feature_types:
        fid = f"FEA-{cid[-4:]}-{ft}"
        cov = "C" if ft == "MedPay" else "A"
        T["FEATURE"].append(dict(FEATURE_ID=fid, CLAIM_ID=cid, CLAIMANT_ID=clt, COVERAGE_ID=f"COV-{pol}-{cov}", FEATURE_TYPE=ft,
                                 STATUS=bi_status, SEVERITY_SCORE=round(R.uniform(1, 10), 1)))
        add_reserves(fid, loss_dt, reserve_plan if ft == "BI" else None)
        payees = [clt] + ([attorney] if attorney else []) + ([provider] if provider else [])
        add_payments(fid, loss_dt, payees, R.choice([0, 1, 1, 2, 3]))
        fids.append(fid)
    if suit:
        stage, filed = suit
        court, venue = R.choice(COURTS)
        T["LITIGATION"].append(dict(LITIGATION_ID=f"LIT-{len(T['LITIGATION']) + 1:04d}", CLAIM_ID=cid, ATTORNEY_ID=attorney,
                                    SUIT_NUMBER=f"{filed.year}-CV-{R.randint(1000, 99999)}", VENUE=venue, COURT=court, STAGE=stage,
                                    FILED_DATE=iso(filed), TRIAL_DATE=iso(filed + timedelta(days=R.randint(300, 600))) if stage != "closed" else None))
        if attorney:
            state["att_open"][attorney] = state["att_open"].get(attorney, 0) + (stage != "closed")
    claim_meta[cid] = dict(insured=insured, loc=loc, occ=occ, claimant=clt, attorney=attorney, provider=provider, loss=loss_dt, features=fids, tpa=tpa)
    return cid


def pick_insured(year_lo, year_hi, exclude=()):
    pool = [i for y in range(year_lo, year_hi + 1) for i in insureds_with_policy_in(y) if i not in exclude]
    return R.choice(pool)


def loss_in(lo, hi, insured):
    for _ in range(50):
        dt = rdate(lo, hi)
        if policy_for(insured, dt):
            return dt
    raise RuntimeError("no policy-covered date")


# PLANTED SCENARIOS
# ---- CQ4 / story: the hero claim (also in CQ1 set, CQ2 set, CQ3 set)
hero_ins = "INS-010"
hero_loss = date(2025, 2, 14)
hero_loc = pick_location(hero_ins)
hero = make_claim(hero_ins, hero_loss, location=hero_loc, attorney="ATT-0001", provider="MP-0001", suit=("discovery", date(2025, 4, 2)), claim_id="CLM-GL-1001",
                  cause="slip_and_fall", desc="Patron slipped on a wet floor near the pool entrance and reported lower-back pain.",
                  reserve_plan=[(25000, date(2025, 3, 1), date(2025, 3, 1)),
                                (60000, date(2025, 6, 15), date(2025, 7, 10)),      # BACKDATED: effective 06-15, only recorded 07-10
                                (185000, date(2025, 10, 20), date(2025, 10, 22))])
# make the hero feature id predictable
for r in T["FEATURE"]:
    if r["CLAIM_ID"] == hero:
        old = r["FEATURE_ID"]
        r["FEATURE_ID"] = "FEA-1001-BI"
        for t in ("RESERVE_STATE", "PAYMENT"):
            for x in T[t]:
                if x["FEATURE_ID"] == old:
                    x["FEATURE_ID"] = "FEA-1001-BI"

hero_clt = claim_meta[hero]["claimant"]
for t_ in ("PAYMENT",):
    T[t_] = [x for x in T[t_] if x["FEATURE_ID"] != "FEA-1001-BI"]
add_payments("FEA-1001-BI", hero_loss, [hero_clt, "ATT-0001", "MP-0001"], 4)
claim_meta[hero]["features"] = ["FEA-1001-BI"]

# ---- CQ1: ATT-0001 = 14 open suits (hero + 13), + 3 closed suits (not counted). ATT-0002 = exactly 10 open (decoy)
att1_claims = [hero]
used_ins_att1 = {hero_ins}
for _ in range(13):
    ins = pick_insured(2024, 2026)
    dt = loss_in(date(2024, 10, 1), date(2026, 6, 30), ins)
    att1_claims.append(make_claim(ins, dt, attorney="ATT-0001", suit=(R.choice(STAGES_OPEN), dt + timedelta(days=R.randint(40, 120))),
                                  provider=None, injury=R.choice(["soft", "other"])))
for _ in range(3):
    ins = pick_insured(2024, 2025)
    dt = loss_in(date(2024, 1, 15), date(2025, 3, 31), ins)
    make_claim(ins, dt, attorney="ATT-0001", suit=("closed", dt + timedelta(days=60)), bi_status="closed", provider=None)
for _ in range(10):
    ins = pick_insured(2024, 2026)
    dt = loss_in(date(2024, 10, 1), date(2026, 6, 30), ins)
    make_claim(ins, dt, attorney="ATT-0002", suit=(R.choice(STAGES_OPEN), dt + timedelta(days=R.randint(40, 120))), provider=None)

# ---- CQ2: provider networks. MP-0001 already has the hero; give 3 ATT-0001 claims + 5 fresh claims => 9 claims at 9 distinct insureds
mp1 = [hero]
by_ins = {claim_meta[c]["insured"] for c in mp1}
for c in att1_claims[1:]:
    if len(mp1) >= 4:
        break
    m = claim_meta[c]
    if m["insured"] in by_ins:
        continue
    clt = next(x for x in T["CLAIMANT"] if x["CLAIMANT_ID"] == m["claimant"])
    clt["PROVIDER_ID"] = "MP-0001"
    inj = next((x for x in T["INJURY"] if x["CLAIMANT_ID"] == m["claimant"]), None)
    if inj is None or inj["INJURY_CLASS"] != "soft_tissue":
        icd, part, _ = R.choice(SOFT)
        if inj:
            inj.update(ICD10=icd, BODY_PART=part, INJURY_CLASS="soft_tissue")
        else:
            T["INJURY"].append(dict(INJURY_ID=f"INJ-{len(T['INJURY']) + 1:04d}", CLAIMANT_ID=m["claimant"], ICD10=icd, BODY_PART=part, INJURY_CLASS="soft_tissue", SEVERITY_BAND="moderate"))
    claim_meta[c]["provider"] = "MP-0001"
    mp1.append(c)
    by_ins.add(m["insured"])
while len(mp1) < 9:
    ins = pick_insured(2025, 2026, exclude=by_ins)
    dt = loss_in(date(2025, 2, 1), date(2026, 6, 30), ins)
    c = make_claim(ins, dt, provider="MP-0001", injury="soft", attorney=R.choice(["ATT-0005", "ATT-0006", None]))
    mp1.append(c)
    by_ins.add(ins)


def provider_block(pid, n, distinct, lo, hi):
    used = set()
    one = pick_insured(lo.year, hi.year)
    for _ in range(n):
        ins = one if not distinct else pick_insured(lo.year, hi.year, exclude=used)
        used.add(ins)
        dt = loss_in(lo, hi, ins)
        make_claim(ins, dt, provider=pid, injury="soft", attorney=None)


provider_block("MP-0002", 5, True, date(2025, 2, 1), date(2026, 6, 30))     # 5 claims: not > 5
provider_block("MP-0003", 7, False, date(2025, 2, 1), date(2026, 6, 30))    # 7 claims, ONE insured
provider_block("MP-0004", 8, True, date(2024, 2, 1), date(2024, 8, 31))     # before the window

# ---- CQ5: three hot locations (4/3/3 occurrences). Use insureds that hold policies in 2025 and 2026.
for loc, n in HOT.items():
    ins = next(r["INSURED_ID"] for r in T["INSURED_LOCATION"] if r["LOCATION_ID"] == loc)
    for k in range(n):
        yr = 2025 if k % 2 == 0 else 2026
        if not policy_for(ins, date(yr, 3, 1)):
            yr = 2024
        dt = loss_in(date(yr, 1, 5), date(yr, 8, 30) if yr == 2026 else date(yr, 11, 30), ins)
        make_claim(ins, dt, location=loc, cause=R.choice(["slip_and_fall", "trip_hazard", "falling_object"]),
                   feature_types=R.choice([("BI",), ("BI", "MedPay"), ("BI", "PD")]), injury=R.choice(["soft", "other"]))

# ---- CQ7: six occurrences with a TPA-administered claim AND a carrier-handled claim
cq7 = []
for _ in range(6):
    ins = pick_insured(2025, 2026)
    dt = loss_in(date(2025, 3, 1), date(2026, 6, 30), ins)
    carrier = make_claim(ins, dt, injury="soft")
    occ = claim_meta[carrier]["occ"]
    tpa_c = make_claim(ins, dt + timedelta(days=R.randint(1, 5)), occurrence=occ, tpa=True, injury="soft")
    cq7.append((tpa_c, carrier, occ))

# ---- bulk filler
n_bulk = 70
for _ in range(n_bulk):
    ins = pick_insured(2024, 2026)
    dt = loss_in(date(2024, 1, 10), date(2026, 7, 31), ins)
    att = None
    if R.random() < 0.35:
        cand = [f"ATT-{i:04d}" for i in range(4, 31) if state["att_open"].get(f"ATT-{i:04d}", 0) < 5]
        att = R.choice(cand) if cand else None
    prov = R.choice([f"MP-{i:04d}" for i in range(5, 26)]) if R.random() < 0.45 else None
    suit = (R.choice(STAGES_OPEN + ["closed"]), dt + timedelta(days=R.randint(40, 120))) if att and R.random() < 0.6 else None
    status = "closed" if (suit and suit[0] == "closed") or R.random() < 0.3 else "open"
    # keep bulk soft-tissue claims off the four planted providers' arithmetic (they only use MP-0005+)
    make_claim(ins, dt, attorney=att, provider=prov, suit=suit, injury=R.choice(["soft", "soft", "other", None]), bi_status=status,
               feature_types=R.choice([("BI",), ("BI",), ("BI", "PD"), ("PD",), ("BI", "MedPay")]), tpa=R.random() < 0.08)

# ---------------------------------------------------------------- documents, chunks, extracted facts
PRE = ["Medical record notes a pre-existing degenerative disc condition at L4-L5 predating the loss.",
       "Adjuster note: claimant has a documented prior back injury (pre-existing condition) from 2022.",
       "Defence IME suggests the complaints reflect a pre-existing condition rather than the incident.",
       "Demand letter acknowledges prior treatment for chronic lumbar pain before the date of loss."]
CLEAN = ["No prior injury to the affected area was reported by the claimant.", "Claimant denies any previous treatment for this body part.",
         "Scene photos show the wet-floor sign was not displayed.", "Surveillance footage was requested from the insured.",
         "Insured reports the floor was mopped 20 minutes earlier.", "Claimant was transported by ambulance and released the same day."]
DOCTYPES = ["FNOL", "adjuster_note", "medical_record", "demand_letter"]

pre_claims = [hero] + R.sample([c for c in claim_meta if c != hero and not claim_meta[c]["tpa"]], 9)
pre_conf = {c: (0.55 if k >= 8 else round(R.uniform(0.82, 0.97), 2)) for k, c in enumerate(pre_claims)}   # last 2 = low-confidence decoys
for c, m in claim_meta.items():
    for dtp in R.sample(DOCTYPES, R.choice([1, 2, 2, 3])):
        did = f"DOC-{len(T['DOCUMENT']) + 1:05d}"
        rec = m["loss"] + timedelta(days=R.randint(2, 200))
        T["DOCUMENT"].append(dict(DOCUMENT_ID=did, CLAIM_ID=c, DOC_TYPE=dtp, RECEIVED_DATE=iso(min(rec, AS_OF)), SOURCE_URI=f"abfss://claims-docs/{c}/{did}.pdf"))
        for ci in range(R.choice([1, 2])):
            txt = R.choice(CLEAN)
            T["TEXT_CHUNK"].append(dict(CHUNK_ID=f"CHK-{len(T['TEXT_CHUNK']) + 1:05d}", DOCUMENT_ID=did, CHUNK_INDEX=ci, TEXT=txt,
                                        EMBEDDING_MODEL=None, MENTIONED_CLAIMANT_ID=m["claimant"] if R.random() < 0.5 else None))
    if c in pre_claims:                                         # guarantee a pre-existing-condition chunk on an adjuster note / medical record
        did = f"DOC-{len(T['DOCUMENT']) + 1:05d}"
        T["DOCUMENT"].append(dict(DOCUMENT_ID=did, CLAIM_ID=c, DOC_TYPE=R.choice(["adjuster_note", "medical_record"]), RECEIVED_DATE=iso(min(m["loss"] + timedelta(days=45), AS_OF)),
                                  SOURCE_URI=f"abfss://claims-docs/{c}/{did}.pdf"))
        T["TEXT_CHUNK"].append(dict(CHUNK_ID=f"CHK-{len(T['TEXT_CHUNK']) + 1:05d}", DOCUMENT_ID=did, CHUNK_INDEX=0, TEXT=R.choice(PRE),
                                    EMBEDDING_MODEL=None, MENTIONED_CLAIMANT_ID=m["claimant"]))
        pre_chunk = T["TEXT_CHUNK"][-1]["CHUNK_ID"]
        T["EXTRACTED_FACT"].append(dict(FACT_ID=f"FCT-{len(T['EXTRACTED_FACT']) + 1:05d}", CHUNK_ID=pre_chunk, FACT_TYPE="pre_existing_condition",
                                        VALUE="degenerative / prior injury", CONFIDENCE=pre_conf[c], MODEL_VERSION="extractor-v1.3", EXTRACTED_AT="2026-10-01",
                                        RUN_ID=EXT_RUN, INJURY_ID=None))
# other facts for ordinary chunks
for ch in list(T["TEXT_CHUNK"]):
    if "pre-existing" in ch["TEXT"] or "prior" in ch["TEXT"].lower() and "Demand" in ch["TEXT"]:
        continue
    if R.random() < 0.35:
        ft = R.choice(["witness_present", "surveillance_requested", "ambulance_transport", "prior_injury_denied"])
        T["EXTRACTED_FACT"].append(dict(FACT_ID=f"FCT-{len(T['EXTRACTED_FACT']) + 1:05d}", CHUNK_ID=ch["CHUNK_ID"], FACT_TYPE=ft, VALUE="true",
                                        CONFIDENCE=round(R.uniform(0.6, 0.98), 2), MODEL_VERSION=R.choice(["extractor-v1.2", "extractor-v1.3"]),
                                        EXTRACTED_AT="2026-10-01", RUN_ID=EXT_RUN, INJURY_ID=None))

# a few recoveries
for c in R.sample(list(claim_meta), 12):
    T["RECOVERY"].append(dict(RECOVERY_ID=f"RCV-{len(T['RECOVERY']) + 1:04d}", CLAIM_ID=c, AMOUNT=float(R.choice([2500, 5000, 12000])), RECOVERY_TYPE=R.choice(["subrogation", "salvage"])))

# ---------------------------------------------------------------- write SQLite
if DB.exists():
    DB.unlink()
con = sqlite3.connect(DB)
for tbl, rows in T.items():
    cols = list(rows[0].keys())
    con.execute(f"CREATE TABLE {tbl} ({', '.join(c + ' TEXT' if c.endswith('_ID') or c in ('NAME',) else c for c in cols)})")
    con.executemany(f"INSERT INTO {tbl} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [tuple(r[c] for c in cols) for r in rows])
con.commit()


def q(sql, *a):
    return con.execute(sql, a).fetchall()


# ---------------------------------------------------------------- ground truth via SQL (independent of the graph)
gt = {"as_of": iso(AS_OF), "since_24m": iso(SINCE_24M)}
gt["CQ1"] = sorted(r[0] for r in q("""
    SELECT f.FEATURE_ID FROM FEATURE f JOIN CLAIM c ON c.CLAIM_ID=f.CLAIM_ID JOIN CLAIMANT t ON t.CLAIMANT_ID=f.CLAIMANT_ID
    WHERE c.LOB='CGL' AND f.FEATURE_TYPE='BI' AND f.STATUS='open' AND t.ATTORNEY_ID IN
      (SELECT ATTORNEY_ID FROM LITIGATION WHERE STAGE<>'closed' GROUP BY ATTORNEY_ID HAVING COUNT(DISTINCT LITIGATION_ID) > 10)"""))
gt["CQ2"] = sorted(r[0] for r in q("""
    SELECT t.PROVIDER_ID FROM CLAIMANT t JOIN INJURY i ON i.CLAIMANT_ID=t.CLAIMANT_ID AND i.INJURY_CLASS='soft_tissue'
    JOIN FEATURE f ON f.CLAIMANT_ID=t.CLAIMANT_ID JOIN CLAIM c ON c.CLAIM_ID=f.CLAIM_ID JOIN POLICY p ON p.POLICY_ID=c.POLICY_ID
    WHERE t.PROVIDER_ID IS NOT NULL AND c.REPORTED_DATE >= ?
    GROUP BY t.PROVIDER_ID HAVING COUNT(DISTINCT c.CLAIM_ID) > 5 AND COUNT(DISTINCT p.INSURED_ID) > 1""", iso(SINCE_24M)))
gt["CQ3"] = sorted(r[0] for r in q("""
    SELECT DISTINCT d.CLAIM_ID FROM EXTRACTED_FACT x JOIN TEXT_CHUNK k ON k.CHUNK_ID=x.CHUNK_ID JOIN DOCUMENT d ON d.DOCUMENT_ID=k.DOCUMENT_ID
    WHERE x.FACT_TYPE='pre_existing_condition' AND x.CONFIDENCE >= 0.7"""))
gt["CQ3_development"] = {}
for cid in gt["CQ3"]:
    for fid, in q("SELECT FEATURE_ID FROM FEATURE WHERE CLAIM_ID=?", cid):
        rs = q("SELECT AMOUNT FROM RESERVE_STATE WHERE FEATURE_ID=? AND RESERVE_TYPE='indemnity' ORDER BY VALID_FROM", fid)
        if rs:
            gt["CQ3_development"][fid] = [rs[0][0], rs[-1][0]]
gt["CQ4"] = {"feature": "FEA-1001-BI",
             "known_2025-06-30": q("""SELECT AMOUNT FROM RESERVE_STATE WHERE FEATURE_ID='FEA-1001-BI' AND RESERVE_TYPE='indemnity' AND RECORDED_AT<='2025-06-30'
                                      AND VALID_FROM<='2025-06-30' ORDER BY VALID_FROM DESC, RECORDED_AT DESC LIMIT 1""")[0][0],
             "known_2025-12-31_asof_2025-06-30": q("""SELECT AMOUNT FROM RESERVE_STATE WHERE FEATURE_ID='FEA-1001-BI' AND RESERVE_TYPE='indemnity' AND RECORDED_AT<='2025-12-31'
                                      AND VALID_FROM<='2025-06-30' ORDER BY VALID_FROM DESC, RECORDED_AT DESC LIMIT 1""")[0][0],
             "current": q("SELECT AMOUNT FROM RESERVE_STATE WHERE FEATURE_ID='FEA-1001-BI' AND IS_CURRENT=1")[0][0]}
gt["CQ5"] = {r[0]: r[1] for r in q("""SELECT LOCATION_ID, COUNT(DISTINCT OCCURRENCE_ID) n FROM OCCURRENCE GROUP BY LOCATION_ID HAVING n>=3 ORDER BY LOCATION_ID""")}
gt["CQ7"] = sorted([list(x) for x in q("""
    SELECT t.CLAIM_ID, c.CLAIM_ID, t.OCCURRENCE_ID FROM CLAIM t JOIN CLAIM c ON c.OCCURRENCE_ID=t.OCCURRENCE_ID AND c.CLAIM_ID<>t.CLAIM_ID
    WHERE t.TPA_ID IS NOT NULL AND c.TPA_ID IS NULL AND c.ADJUSTER_ID IS NOT NULL""")], key=lambda x: x[2])
gt["CQ6_sample_payment"] = (q("SELECT PAYMENT_ID FROM PAYMENT WHERE FEATURE_ID='FEA-1001-BI' ORDER BY PAYMENT_ID LIMIT 1") or [[None]])[0][0]
gt["CQ6_expected_facts"] = sorted(r[0] for r in q("""
    SELECT x.FACT_ID FROM PAYMENT p JOIN FEATURE f ON f.FEATURE_ID=p.FEATURE_ID JOIN DOCUMENT d ON d.CLAIM_ID=f.CLAIM_ID
    JOIN TEXT_CHUNK k ON k.DOCUMENT_ID=d.DOCUMENT_ID JOIN EXTRACTED_FACT x ON x.CHUNK_ID=k.CHUNK_ID WHERE p.PAYMENT_ID=?""", gt["CQ6_sample_payment"]))
GT.write_text(json.dumps(gt, indent=2), encoding="utf-8")

print(f"Wrote {DB.name}: " + ", ".join(f"{t}={len(r)}" for t, r in T.items()))
print("Ground truth sizes:", {k: (len(v) if hasattr(v, '__len__') else v) for k, v in gt.items() if k.startswith('CQ')})
