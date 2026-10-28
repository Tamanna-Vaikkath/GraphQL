"""
Negative tests: prove the ontology is ENFORCED, not just documented.  Run:  python tests/test_ontology_enforcement.py
Each case feeds deliberately bad data / mapping and asserts the ontology layer rejects it with a clear message.
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kg.ontology import Ontology, load_ontology            # noqa: E402
from kg.graph_ingest_gl import build_model, validate_model  # noqa: E402


class FakeSrc:
    label = "fake"

    def __init__(self, tables):
        self.t = tables

    def rows(self, name):
        return self.t.get(name, [])


def base_tables():
    return {
        "POLICY": [{"POLICY_ID": "P1", "POLICY_NUMBER": "N1", "FORM": "occurrence", "EFFECTIVE_DATE": "2025-01-01", "EXPIRY_DATE": "2026-01-01",
                    "LIMIT_PER_OCCURRENCE": 1e6, "RETENTION": 5000, "BROKER_ID": None, "INSURED_ID": "I1"}],
        "INSURED": [{"INSURED_ID": "I1", "NAME": "Acme"}],
        "ATTORNEY": [{"ATTORNEY_ID": "A1", "NAME": "X", "FIRM": "F", "BAR_ID": "B"}],
        "CLAIMANT": [{"CLAIMANT_ID": "C1", "NAME": "Y", "STATE": "OH", "RESOLUTION_CONFIDENCE": 1.0, "ATTORNEY_ID": "A1", "PROVIDER_ID": None}],
        "FEATURE": [{"FEATURE_ID": "F1", "CLAIM_ID": None, "CLAIMANT_ID": "C1", "COVERAGE_ID": None, "FEATURE_TYPE": "BI", "STATUS": "open", "SEVERITY_SCORE": 5}],
        "PAYMENT": [{"PAYMENT_ID": "PAY1", "FEATURE_ID": "F1", "PAYEE_ID": "A1", "AMOUNT": 100, "CHECK_NUMBER": "1", "PAYMENT_DATE": "2025-02-01", "COST_CATEGORY": "indemnity", "RUN_ID": None}],
    }


onto = load_ontology()
passed = 0


def check(name, cond):
    global passed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    passed += bool(cond)
    return cond


# 0. sanity: the clean fixture is accepted
m = build_model(onto, FakeSrc(base_tables()))
check("clean data is accepted", validate_model(onto, m) == [])

# 1. payee may be ANY Party subclass (range = Party)
check("PAYEE -> Attorney allowed (subclass of Party)", onto.edge_allowed("PAYEE", "Payment", "Attorney"))

# 2. an edge the ontology does not declare is rejected
t = base_tables(); t["PAYMENT"][0]["PAYEE_ID"] = "P1"          # payee that is a Policy, not a Party
m = build_model(onto, FakeSrc(t))
check("payee that is not a Party is rejected", any("PAYEE" in p for p in validate_model(onto, m)))

# 3. dangling reference
t = base_tables(); t["PAYMENT"][0]["FEATURE_ID"] = "F-MISSING"
check("dangling reference is flagged", any("dangling" in p for p in validate_model(onto, build_model(onto, FakeSrc(t)))))

# 4. enum violation
t = base_tables(); t["FEATURE"][0]["FEATURE_TYPE"] = "WorkersComp"
check("value outside the enum is flagged", any("featureType" in p for p in validate_model(onto, build_model(onto, FakeSrc(t)))))

# 5. same id used by two different Party roles
t = base_tables(); t["CLAIMANT"][0]["CLAIMANT_ID"] = "A1"
check("id shared by two Party roles is flagged", any("two Party roles" in p for p in validate_model(onto, build_model(onto, FakeSrc(t)))))

# 6. mapping mentions a property the ontology does not declare
bad = copy.deepcopy(onto.mapping); bad["nodes"]["Attorney"]["props"]["favouriteColour"] = "BAR_ID"
check("undeclared property in mapping is flagged", any("favouriteColour" in e for e in Ontology(onto.ont, bad).validate_mapping()))

# 7. mapping declares a relationship the ontology does not
bad = copy.deepcopy(onto.mapping); bad["relationships"].append({"rel": "OWNS", "from": "Attorney", "to": "Policy", "table": "X", "from_id": "A", "to_id": "B"})
check("undeclared relationship in mapping is flagged", any("OWNS" in e for e in Ontology(onto.ont, bad).validate_mapping()))

# 8. a new ontology version changes the graph version (=> rebuild)
changed = copy.deepcopy(onto.ont); changed["classes"]["Claim"]["props"].append("newField")
check("ontology change changes graph version", Ontology(changed, onto.mapping).graph_version() != onto.graph_version())

print(f"\n{passed}/9 passed")
sys.exit(0 if passed == 9 else 1)
