"""
utils/safety.py — guard rails that keep the DEV copy from touching PRODUCTION.

Two independent protections:

1. DELTA / UNITY CATALOG
   assert_safe_schema_write(catalog, schema)
   Any script that creates/overwrites/drops tables refuses to run against a
   protected schema. `graphql.insurance_dev` (the working demo's schema) is protected
   by default; add more with the env var PROTECTED_DATABRICKS_SCHEMAS
   (comma-separated, "catalog.schema").

2. NEO4J
   assert_neo4j_is_dev_or_empty(session)
   Scripts that DELETE or REBUILD graph data first check the target database.
   They only go ahead if the database is EMPTY or carries the DEV marker
   node (:GraphMeta {name:'environment', value:'dev'}) that these same
   scripts stamp on databases they create. A database that already holds
   data without that marker (for example the production graph) is refused,
   with a clear error, before anything is deleted.
   Optional extra: PROTECTED_NEO4J_URIS (comma-separated) is checked too.

Also exposes app_env() and describe_target() for the UI banner.
"""
from __future__ import annotations

import os
from typing import Iterable, List

DEFAULT_PROTECTED_SCHEMAS = ("graphql.insurance_dev",)   # the WORKING demo. graphql.insurance is the GL build's target schema.
DEV_MARKER_NAME = "environment"
DEV_MARKER_VALUE = "gl"   # GL build marker: refuses to wipe the original demo graph (marker "dev")


class ProductionGuardError(RuntimeError):
    """Raised when an operation would touch a protected (production) resource."""


def _csv_env(name: str) -> List[str]:
    return [x.strip().lower() for x in os.getenv(name, "").split(",") if x.strip()]


def app_env() -> str:
    return os.getenv("APP_ENV", "dev").strip().lower()


def protected_schemas() -> List[str]:
    return sorted(set(DEFAULT_PROTECTED_SCHEMAS) | set(_csv_env("PROTECTED_DATABRICKS_SCHEMAS")))


def is_protected_schema(catalog: str, schema: str) -> bool:
    return f"{catalog}.{schema}".strip().lower() in protected_schemas()


def assert_safe_schema_write(catalog: str, schema: str, action: str = "write to") -> None:
    if is_protected_schema(catalog, schema):
        raise ProductionGuardError(
            f"REFUSING to {action} {catalog}.{schema}: it is a PROTECTED (production) schema.\n"
            f"This schema belongs to the working demo; the GL build writes to graphql.insurance."
        )


def _norm_uri(uri: str) -> str:
    u = (uri or "").strip().lower()
    for prefix in ("neo4j+s://", "neo4j+ssc://", "neo4j://", "bolt+s://", "bolt+ssc://", "bolt://"):
        if u.startswith(prefix):
            u = u[len(prefix):]
    return u.rstrip("/")


def assert_neo4j_is_dev_or_empty(session, uri: str = "", action: str = "delete / rebuild data in") -> None:
    """Refuse to continue if this Neo4j database holds data the dev build did not create."""
    if uri and _norm_uri(uri) in {_norm_uri(u) for u in _csv_env("PROTECTED_NEO4J_URIS")}:
        raise ProductionGuardError(
            f"REFUSING to {action} Neo4j at {uri}: it is listed in PROTECTED_NEO4J_URIS."
        )
    total = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    if total == 0:
        return
    marker = session.run(
        "MATCH (m:GraphMeta {name: $n, value: $v}) RETURN count(m) AS c",
        n=DEV_MARKER_NAME, v=DEV_MARKER_VALUE,
    ).single()["c"]
    if marker == 0:
        raise ProductionGuardError(
            f"REFUSING to {action} this Neo4j database: it already contains {total} node(s) "
            "that were NOT created by the DEV build (no dev marker found). "
            "This looks like a production / shared database.\n"
            "Point NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD at a NEW, EMPTY Neo4j instance for dev."
        )


def stamp_dev_marker(session) -> None:
    session.run(
        "MERGE (m:GraphMeta {name: $n}) SET m.value = $v, m.note = "
        "'Created by the DEV build. Scripts may delete/rebuild this database.'",
        n=DEV_MARKER_NAME, v=DEV_MARKER_VALUE,
    )


def _mask(uri: str) -> str:
    u = _norm_uri(uri)
    if len(u) <= 10:
        return u
    return u[:6] + "…" + u[-10:]


def describe_target(cfg) -> dict:
    """Small dict for the UI banner: which data / graph is this app pointed at?"""
    backend = getattr(cfg, "data_backend", "sqlite")
    if backend == "databricks":
        data = f"{cfg.databricks_catalog}.{cfg.databricks_schema}"
    else:
        data = f"SQLite file {cfg.sqlite_db_path}"
    return {
        "env": app_env(),
        "backend": backend,
        "data": data,
        "neo4j": _mask(cfg.neo4j_uri),
        "protected_data": backend == "databricks" and is_protected_schema(cfg.databricks_catalog, cfg.databricks_schema),
    }
