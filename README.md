# Graph Query Language

P&C Insurance NLQ platform that transforms natural language questions into explainable SQL using Neo4j Knowledge Graphs, semantic retrieval, and LLM-based reasoning.

---

## Overview

This project demonstrates how Knowledge Graphs can improve insurance analytics by enabling:

* Semantic schema understanding
* Dynamic join discovery
* Explainable SQL generation
* Graph-powered reasoning
* AI-generated business insights
* Real-time KG traversal visualization

The platform focuses on the insurance claims lifecycle across:

* Claims
* Claimants
* Policies
* Payments

---

## Key Features

* Neo4j Knowledge Graph integration
* Semantic vector retrieval using embeddings
* HyDE-based query expansion
* Dynamic FK traversal and join discovery
* Explainable SQL generation pipeline
* LLM reranking and grounding
* Real-time KG reasoning visualization
* AI-generated business summaries

---

## Tech Stack

### Backend

* Python
* SQLite
* Neo4j AuraDB
* OpenAI / Azure OpenAI

### AI & Knowledge Graph

* Neo4j Knowledge Graph
* Vector Embeddings
* Semantic Retrieval
* HyDE Query Expansion
* LLM-based SQL Generation

### Frontend

* Streamlit
* Interactive KG Visualization

---

## Project Flow

```text
User Query
   ↓
HyDE Query Expansion
   ↓
Embedding Generation
   ↓
Neo4j KG Semantic Retrieval
   ↓
FK Join Path Discovery
   ↓
LLM SQL Generation
   ↓
SQL Validation & Repair
   ↓
Database Execution
   ↓
AI Business Summary
```

---

## Repository Structure

```text
Claims-Lifecycle-Knowledge-Graph/
│
├── assets/                # UI assets
├── database/              # SQLite DB and setup scripts
├── kg/                    # Neo4j KG construction and retrieval
├── pipeline/              # NLQ-to-SQL orchestration
├── utils/                 # Utility functions and LLM helpers
│
├── app.py                 # Streamlit application
├── config.py              # Environment configuration
├── requirements.txt
└── README.md
```

---

## Example Queries

```text
show open claims from Texas
show pending claims with payment
show denied claims
show unpaid claims
show policies with active claims
```

---

## Knowledge Graph Reasoning

The system dynamically:

* retrieves semantically relevant schema columns,
* traverses FK relationships in Neo4j,
* discovers shortest join paths,
* generates explainable SQL queries,
* visualizes graph reasoning in real time.

Example traversal:

```text
Texas
   ↓
CLAIMANT.STATE_CD
   ↓
CLAIMANT
   ↓
CLAIMS
   ↓
Generated SQL
```

---

## Highlights

* Explainable AI reasoning
* Graph-based SQL generation
* Dynamic schema grounding
* Relationship-aware retrieval
* Real-time traversal visualization
* Claims lifecycle intelligence

---

## Author

Tamanna Vaikkath
