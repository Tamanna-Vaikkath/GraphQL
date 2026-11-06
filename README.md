# NLQ Intelligence 

A natural-language query system over a Commercial General Liability (CGL) claims dataset that routes
every question down one of two swim lanes — **SQL** for filters, aggregations and fixed-depth joins, or
**Cypher / Knowledge Graph** for relationship-centric, multi-hop, unknown-depth questions — and answers
it against real data, with the full reasoning trail visible end to end.

The point of this repo isn't "an LLM that writes SQL." It's a demonstration that a **governed ontology**
can sit in front of an LLM and make its output provably correct: every node, edge and property the graph
is allowed to contain is declared once, in YAML, and every generated query is validated against that
declaration before it runs.

---

## What this demonstrates

- **Two swim lanes, one router.** `pipeline/query_router.py` decides, from the business intent of the
  question (not just keyword matching), whether it belongs in the SQL lane or the Cypher lane — with a
  fast deterministic heuristic first, and an LLM classifier only as a fallback.
- **An ontology that's actually enforced, not just documentation.** `ontology/gl_claims.yaml` is the single
  source of truth for every class, relationship, property and allowed value in the graph. Ingest refuses
  anything not declared; a relationship's domain/range is checked (`Payment -[:PAYEE]-> Attorney` is fine,
  `Payment -[:PAYEE]-> Policy` is rejected); and changing the ontology changes the graph's version, which
  triggers a rebuild.
- **Competency questions as executable tests.** Seven target questions (CQ1–CQ7) are both the ontology's
  stated scope and an automated test suite (`tests/run_cq_tests.py`) — each one's Cypher result is checked
  against a SQL ground truth computed completely independently, so passing the suite proves the graph
  traversal agrees with the raw data, not just with itself.
- **Full query traceability.** Every answer comes with an expandable trace: HyDE query expansion, the
  exact columns/schema retrieved, the generated SQL or Cypher (with any auto-repair attempts), and a live
  "traversal graph" visualization of the specific tables/labels and joins/relationships that query actually
  touched — not a static schema diagram.
- **Provenance and bitemporal history built into the data model.** Every extracted fact and payment
  carries the ingestion run that produced it (`IngestionRun`), and `Reserve` is modeled as versioned state
  (`validFrom`/`validTo`/`recordedAt`) so "what was the reserve as of quarter-end close" is answerable
  without ever overwriting history.

---

## Architecture

```
                                    ┌─────────────────────────┐
  User question  ───────────────▶  │   Query Router           │
                                    │ (keyword heuristic, LLM  │
                                    │  fallback)                │
                                    └───────────┬──────────────┘
                         ┌────────────────────┬─┴───────────────────────┐
                         ▼                                              ▼
            ┌────────────────────────────┐                ┌──────────────────────────────┐
            │          SQL LANE           │                │        CYPHER / KG LANE       │
            │                             │                │                                │
            │ Schema Scoping              │                │ Relationship Detection         │
            │   → HyDE Expansion          │                │   → Graph Schema Grounding     │
            │   → KG Column Retrieval     │                │   → Cypher Generation          │
            │   → Join Discovery          │                │   → Neo4j Traversal            │
            │   → SQL Generation          │                │   → Result Binding             │
            │   → SQLite/Databricks Exec  │                │                                │
            └────────────────────────────┘                └──────────────────────────────┘
                         │                                              │
                         └────────────────────┬─────────────────────────┘
                                               ▼
                                   Result DataFrame + AI Summary
                                   + full Query Trace (expandable)
```

Both lanes are grounded in the **same ontology** (`ontology/gl_claims.yaml` + `ontology/gl_mapping.yaml`):
the SQL lane's schema metadata and the Cypher lane's graph schema block are both generated from it, so the
two lanes can never silently drift apart.

Some graph questions — claim clusters/networks tied together through a shared adjuster, agent or payee —
are answered by a deterministic, Python-side connected-components traversal
(`pipeline/cypher_generator.find_claim_clusters`) rather than by asking the LLM to free-hand a single
Cypher statement for something that's really a graph algorithm. The router recognizes this pattern and
routes to it directly.

---

## Data model

The ontology (`ontology/gl_claims.yaml`) organizes 23 classes into five domains:

| Domain | Example classes |
|---|---|
| **Contract** | `Policy`, `Broker`, `InsuredLocation`, `CoveragePart` |
| **Loss & Claim** | `Claim`, `Occurrence`, `Litigation`, `Feature`, `Injury` |
| **Financial** | `Reserve` (versioned state), `Payment`, `Recovery` |
| **Party** | `Party` (abstract), `Insured`, `Adjuster`, `TPA`, `Claimant`, `Attorney`, `MedicalProvider` |
| **Evidence & Provenance** | `Document`, `Chunk`, `ExtractedFact`, `IngestionRun` |

`Feature` is the key modeling idea: it represents one claimant × one coverage, and is the unit that
reserves, payments and coverage decisions actually attach to — not the claim header itself, which can
have several claimants and several coverages in play at once.

### Competency questions (the ontology's scope, and its test suite)

| ID | Question | Capability it proves |
|---|---|---|
| CQ1 | Which open CGL bodily-injury features have a claimant represented by counsel with more than 10 open suits against us? | 4-hop traversal + aggregation |
| CQ2 | Which medical providers appear on more than N soft-tissue claims across different insureds in the last 24 months? | Network centrality + time filter |
| CQ3 | Which claims mention a pre-existing condition, and what is their reserve development? | Extracted facts + temporal state |
| CQ4 | What was the indemnity reserve on feature X as known on the quarter-end close date? | Bitemporal (as-known-at) query |
| CQ5 | Which insured locations have had 3+ occurrences, and under which coverage parts were they paid? | Location rollup + coverage linkage |
| CQ6 | For a given payment, which source system, ingestion run and model produced each supporting fact? | Provenance / lineage |
| CQ7 | Which TPA-reported claims map to the same occurrence as a carrier-handled claim? | Shared-occurrence resolution |

---

## Repository structure

```
app.py                      Streamlit UI — the only entry point for the demo
config.py                   Typed config loaded from .env (OpenAI, Neo4j, SQLite/Databricks)
requirements.txt

ontology/
  gl_claims.yaml             SOURCE OF TRUTH — classes, relationships, competency questions, query rules
  gl_mapping.yaml            How ontology classes/relationships map onto the relational tables
  fewshot.yaml                Few-shot Cypher examples used in the generation prompt
  cq_tests.yaml                Executable form of the competency questions + ground-truth checks

kg/
  ontology.py                 Runtime loader/validator for the two YAML files above
  graph_ingest_gl.py           Ontology-driven ingest: SQL tables → GraphModel → validate → Neo4j
  schema_registry.py           Dynamic schema metadata (tables/columns/FKs), derived at runtime
  schema_retriever.py          Stage 0 — scoped schema retrieval for the SQL lane
  retriever.py                 KG-side retrieval for the Cypher lane
  build_graph.py               Builds/embeds the schema knowledge graph the SQL lane searches

pipeline/
  query_router.py              Decides SQL lane vs Cypher lane
  hyde_expander.py             HyDE query expansion for semantic schema retrieval
  sql_generator.py             SQL generation (Stage 4 of the SQL lane)
  cypher_generator.py          Cypher generation + the deterministic claim-cluster traversal
  query_complexity_analyzer.py Plans multi-pass SQL for nested/comparative questions
  executor.py / cypher_executor.py   Execution + auto-repair for SQL / Cypher respectively
  summarizer.py                 Optional AI Summary of results
  orchestrator.py               Ties every stage together end to end

data/
  generate_gl_data.py           Deterministic synthetic CGL claims data + independent SQL ground truth

database/
  gl_ground_truth.json          Ground truth for CQ1–CQ7, computed independently of the graph
  seed_db.py / build_dev_db.py  Legacy/alternate seed scripts for an earlier simple demo schema

tools/
  build_ontology_artifacts.py   Regenerates Neo4j constraints, the LLM schema block, glossary, and the
                                 interactive ontology visual — all from the ontology, nothing hand-maintained
  neo4j_reset_for_gl.py         Read-only inspection (and guarded wipe) of the target Neo4j database

tests/
  run_cq_tests.py               Runs CQ1–CQ7 (and few-shot examples) against Neo4j or an embedded engine
  test_ontology_enforcement.py  Proves the ontology actually rejects out-of-model data
  test_cypher_validator.py      Proves the Cypher validator accepts good queries, rejects bad ones
  test_neo4j_guard.py / test_health.py

utils/
  llm_client.py                 Azure OpenAI client wrapper
  sql_traversal.py / cypher_traversal.py   Build the live, per-query "traversal graph" shown in the UI
  safety.py                     Guard rails that keep this dev copy from ever touching a production schema/graph
  health.py                     Backing logic for the in-app Health Check page
```

---

## Getting started

### 1. Prerequisites

- Python 3.10+
- A Neo4j instance (AuraDB or self-hosted) — the Cypher lane always talks to Neo4j directly
- Azure OpenAI access (chat completions + embeddings)

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```ini
# Azure OpenAI — chat
OPENAI_DEPLOYMENT_ENDPOINT=https://<your-resource>.openai.azure.com/
OPENAI_API_KEY=<key>
OPENAI_API_VERSION=2025-04-01-preview
OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
OPENAI_MODEL_NAME=gpt-4o-mini

# Azure OpenAI — embeddings
OPENAI_EMBEDDING_ENDPOINT=https://<your-resource>.openai.azure.com/
OPENAI_EMBEDDING_API_KEY=<key>
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Neo4j
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j

# Data backend — "sqlite" (default, local file) or "databricks"
DATA_BACKEND=sqlite
SQLITE_DB_PATH=database/gl_claims.db

# Only needed if DATA_BACKEND=databricks
# DATABRICKS_CATALOG=...
# DATABRICKS_SCHEMA=...
# DATABRICKS_SERVER_HOSTNAME=...
# DATABRICKS_HTTP_PATH=...
# DATABRICKS_TOKEN=...          # not needed inside a Databricks App — auth is automatic there
```

### 4. Build the data and the graph

```bash
# Generates database/gl_claims.db + database/gl_ground_truth.json
python data/generate_gl_data.py

# Regenerates generated/* (Neo4j constraints, LLM schema block, glossary, ontology visual)
python tools/build_ontology_artifacts.py
```

The graph itself is built the first time the app runs (`kg.graph_ingest_gl.ensure_graph_ingested`), or
on demand from the in-app **Health Check** page ("Build business graph now"). Before pointing this at a
shared Neo4j instance, run the read-only check:

```bash
python tools/neo4j_reset_for_gl.py
```

This prints what's already in that database and whether it's safe for the GL build to use.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Using the app

- **Type a Query** — the main chat interface. Ask a question, see the routed lane, the result table, an
  optional AI-generated plain-English summary, and (toggle-able) the full query trace: HyDE expansion,
  retrieved schema, generated SQL/Cypher, and the live traversal graph of exactly what the query touched.
  Sample questions for both lanes are in the sidebar.
- **Graph Visualization** — write or generate Cypher directly and see it rendered as an interactive graph
  against the real business graph.
- **Health Check** — what the app can currently see in SQLite/Databricks and Neo4j, with one-click buttons
  to (re)build the business graph and the schema knowledge graph, guarded so they refuse to run against
  a Neo4j instance that already holds a different demo's data.

---

## Testing

```bash
# Full competency-question suite against an embedded openCypher engine (no Neo4j needed)
pip install falkordblite
python tests/run_cq_tests.py --embedded --fewshot

# Against your configured Neo4j instance instead (graph must already be ingested)
python tests/run_cq_tests.py

# Ontology enforcement — proves bad data is actually rejected, not just discouraged
python tests/test_ontology_enforcement.py

# Cypher validator — accepts good queries, rejects ontology-violating ones
python tests/test_cypher_validator.py

# The reset tool never wipes a working demo's graph
python tests/test_neo4j_guard.py
```

`run_cq_tests.py` exits non-zero if any competency question's graph answer disagrees with its
independently-computed SQL ground truth — it's meant to gate a deployment, not just run informally.

---

## Notes

- `.env`, `*.db`, and `generated/*.html` are gitignored — generate them locally with the commands above.
- `database/seed_db.py` / `database/build_dev_db.py` build an older, simpler demo schema and are not part
  of the active CGL claims ontology; the active data path is `data/generate_gl_data.py` →
  `database/gl_claims.db`.
- `utils/safety.py` guards against this dev copy ever overwriting a production Databricks schema or a
  different Neo4j graph — see its docstring for the two independent protections.
