"""
utils/db_connector.py — Single place that knows how to get a DB connection
and run a query, for EITHER backend (sqlite or databricks).

Everywhere in the codebase that used to do:

    import sqlite3
    conn = sqlite3.connect(cfg.sqlite_db_path)
    df = pd.read_sql_query(sql, conn)

now does:

    from utils.db_connector import run_query
    df = run_query(cfg, sql)

This is the ONLY file that needs to know about how the databricks
backend connects. Everything else in the pipeline stays backend-agnostic.

NOTE ON THE DATABRICKS BACKEND — SQL WAREHOUSE
------------------------------------------------
This project talks to Databricks through a **SQL Warehouse**, using the
`databricks-sql-connector` package (the same driver that powers the
Databricks SQL JDBC/ODBC connectors), NOT Spark / Databricks Connect.
There is no SparkSession anywhere in this path — every query is sent as
plain ANSI SQL over a lightweight thrift connection to the warehouse and
comes back as a cursor of rows, which we turn into a pandas DataFrame.

Set the following in your .env (or the Databricks App's environment):

    DATA_BACKEND=databricks
    DATABRICKS_SERVER_HOSTNAME=<workspace-hostname, no https://>
    DATABRICKS_HTTP_PATH=<SQL warehouse HTTP path, e.g. /sql/1.0/warehouses/xxxx>
    DATABRICKS_TOKEN=<personal access token>        (omit if running as a
                                                       Databricks App — see below)
    DATABRICKS_CATALOG=<catalog name, e.g. graphql>
    DATABRICKS_SCHEMA=<schema name, e.g. insurance>

Auth
----
- Locally / notebooks: set DATABRICKS_TOKEN to a personal access token.
- Running as a Databricks App: leave DATABRICKS_TOKEN unset. The connector
  explicitly authenticates via the Databricks SDK's Config object
  (`credentials_provider=lambda: Config().authenticate`), which picks up
  the app's own service-principal credentials (DATABRICKS_CLIENT_ID /
  DATABRICKS_CLIENT_SECRET / DATABRICKS_HOST, already injected by the
  platform) — no token needed. This does NOT fall back to interactive
  browser OAuth.

Set DATA_BACKEND=sqlite (or leave unset) to keep the original local-demo
behavior — this is what lets you keep developing/testing on your laptop
exactly as before while the Databricks side is being stood up.

Requires the `databricks-sql-connector` package (see requirements.txt).
Unlike `databricks-connect`, this does NOT conflict with `pyspark` and
does not need a warehouse-less serverless Spark session — it needs a
running SQL Warehouse (serverless or classic) reachable via HTTP path.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Tuple, TypeVar

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# Cached SQL Warehouse connection so we don't reconnect on every query.
_databricks_conn: Optional[object] = None
_conn_lock = threading.RLock()

T = TypeVar("T")

# Server-side sessions on a SQL Warehouse are closed after a period of
# inactivity (and whenever the warehouse auto-stops / restarts). The cached
# connection object still exists in this process but its SessionHandle is
# dead, so the next statement fails with e.g.
#   INVALID_STATE: Invalid SessionHandle: SessionHandle [...]
# We detect these errors, throw the dead connection away, reconnect and retry.
_STALE_SESSION_MARKERS = (
    "invalid sessionhandle",
    "invalid_state",
    "session is closed",
    "session was closed",
    "session_closed",
    "connection is closed",
    "connection closed",
    "invalid operationhandle",
    "operation is closed",
    "could not find session",
    "session not found",
    "unauthenticated",
    "invalid access token",
    "token has expired",
    "temporarily_unavailable",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "max retries exceeded",
    "bad gateway",
    "service unavailable",
)


def _is_stale_session_error(exc: BaseException) -> bool:
    """True if `exc` looks like a dead/expired warehouse session or connection."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(m in text for m in _STALE_SESSION_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def get_databricks_connection(cfg: Config) -> object:
    """Return a cached databricks-sql-connector connection to the SQL Warehouse.

    Auth priority:
      1. DATABRICKS_TOKEN (personal access token), if set.
      2. Otherwise, explicit service-principal auth via the Databricks SDK's
         Config object (automatic when running as a Databricks App, using
         its own injected service-principal env vars).
    """
    global _databricks_conn

    with _conn_lock:
        if _databricks_conn is not None:
            return _databricks_conn
        _databricks_conn = _open_databricks_connection(cfg)
        return _databricks_conn


def _open_databricks_connection(cfg: Config) -> object:
    """Open a brand-new SQL Warehouse connection (no caching)."""
    from databricks import sql as databricks_sql

    logger.info(
        "Connecting to Databricks SQL Warehouse %s (http_path=%s)",
        cfg.databricks_server_hostname, cfg.databricks_http_path,
    )

    connect_kwargs = {
        "server_hostname": cfg.databricks_server_hostname,
        "http_path": cfg.databricks_http_path,
        "catalog": cfg.databricks_catalog,
        "schema": cfg.databricks_schema,
    }

    if cfg.databricks_token:
        # Explicit PAT auth (local dev / notebooks).
        connect_kwargs["access_token"] = cfg.databricks_token
    else:
        # Running as a Databricks App: explicitly wire in the app's own
        # service-principal credentials via the Databricks SDK's Config
        # object. Just omitting access_token/auth_type is NOT enough —
        # databricks-sql-connector still falls back to the interactive
        # user-to-machine (U2M) OAuth flow by default, which spins up a
        # localhost:8020 browser-login listener and hangs forever inside
        # an App (no browser/interactive session available there).
        # `Config()` auto-picks up DATABRICKS_CLIENT_ID / _CLIENT_SECRET /
        # _HOST, which the platform injects for you.
        from databricks.sdk.core import Config as DatabricksSdkConfig

        sdk_cfg = DatabricksSdkConfig()
        connect_kwargs["credentials_provider"] = lambda: sdk_cfg.authenticate

    return databricks_sql.connect(**connect_kwargs)


def reset_databricks_connection() -> None:
    """Drop the cached SQL Warehouse connection (e.g. between test runs)."""
    global _databricks_conn
    with _conn_lock:
        if _databricks_conn is not None:
            try:
                _databricks_conn.close()
            except Exception:  # noqa: BLE001
                # Expected when the server already dropped the session.
                logger.debug("Error closing Databricks SQL Warehouse connection", exc_info=True)
            _databricks_conn = None


def with_databricks_connection(cfg: Config, fn: Callable[[object], T]) -> T:
    """Run `fn(conn)` on the warehouse connection, self-healing stale sessions.

    If the cached connection's server-side session has expired (idle timeout,
    warehouse restart, token rotation...), drop it, open a fresh connection
    and retry once. Any other error is raised unchanged.
    """
    conn = get_databricks_connection(cfg)
    try:
        return fn(conn)
    except Exception as exc:  # noqa: BLE001
        if not _is_stale_session_error(exc):
            raise
        logger.warning(
            "Databricks SQL session is stale (%s); reconnecting and retrying once.", exc
        )
        with _conn_lock:
            # Only reset if nobody else already replaced the dead connection.
            if _databricks_conn is conn:
                reset_databricks_connection()
        return fn(get_databricks_connection(cfg))


def run_rows(cfg: Config, sql: str) -> List[Tuple]:
    """Run a SQL statement on the warehouse and return rows as plain tuples."""
    def _do(conn: object) -> List[Tuple]:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            return [tuple(r) for r in cur.fetchall()]
        finally:
            try:
                cur.close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing cursor", exc_info=True)

    return with_databricks_connection(cfg, _do)


@contextmanager
def get_connection(cfg: Config) -> Iterator[object]:
    """Yield a live DB-API connection for the sqlite backend.

    Only used for sqlite — the databricks backend runs through the
    cached SQL Warehouse connection (see get_databricks_connection) /
    run_query(cfg, sql) instead.
    """
    if cfg.data_backend == "databricks":
        raise RuntimeError(
            "get_connection() is not supported for the databricks backend; "
            "use get_databricks_connection(cfg) / run_query(cfg, sql) "
            "instead, which run queries via a Databricks SQL Warehouse "
            "connection (databricks-sql-connector)."
        )
    conn = sqlite3.connect(cfg.sqlite_db_path)
    try:
        yield conn
    finally:
        conn.close()


def qualify(cfg: Config, table_name: str) -> str:
    """Return the fully-qualified table name for the active backend.

    - databricks: catalog.schema.TABLE  (Unity Catalog three-level namespace)
    - sqlite:     TABLE                 (no catalog/schema concept)
    """
    if cfg.data_backend == "databricks":
        return f"{cfg.databricks_catalog}.{cfg.databricks_schema}.{table_name}"
    return table_name


def run_query(cfg: Config, sql: str) -> pd.DataFrame:
    """Run a SQL statement against the active backend and return a DataFrame.

    - sqlite:     goes through sqlite3 + pandas.read_sql_query.
    - databricks: goes through a SQL Warehouse connection
                  (databricks-sql-connector cursor.fetchall_arrow()/pandas).
    """
    if cfg.data_backend == "databricks":
        def _do(conn: object) -> pd.DataFrame:
            cur = conn.cursor()
            try:
                cur.execute(sql)
                # fetchall_arrow() is the fastest path for large result sets;
                # fall back to plain fetchall() + description if unavailable.
                try:
                    return cur.fetchall_arrow().to_pandas()
                except AttributeError:
                    columns = [c[0] for c in cur.description]
                    rows = cur.fetchall()
                    return pd.DataFrame(rows, columns=columns)
            finally:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001
                    logger.debug("Error closing cursor", exc_info=True)

        return with_databricks_connection(cfg, _do)

    with get_connection(cfg) as conn:
        return pd.read_sql_query(sql, conn)


def execute(cfg: Config, sql: str) -> None:
    """Run a statement that doesn't return rows (CREATE TABLE, INSERT, etc.)."""
    if cfg.data_backend == "databricks":
        def _do(conn: object) -> None:
            cur = conn.cursor()
            try:
                cur.execute(sql)
            finally:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001
                    logger.debug("Error closing cursor", exc_info=True)

        with_databricks_connection(cfg, _do)
        return

    with get_connection(cfg) as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            conn.commit() if hasattr(conn, "commit") else None
        finally:
            cur.close()