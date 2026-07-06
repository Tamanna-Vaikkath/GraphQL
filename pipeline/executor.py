"""
pipeline/executor.py — Stage 5: SQL Execution + Self-Healing Repair Agent.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)


def _build_repair_system(registry=None) -> str:
    base = (
        "You are a SQL repair agent for a SQLite database.\n"
        "You will receive a failing SQL query and its error. "
        "Fix the SQL and return ONLY the corrected SQL.\n"
        "No explanation, no markdown fences.\n\n"
    )

    if registry is not None:
        schema_lines = ["Schema:"]
        for tname, tmeta in registry.tables.items():
            col_names = ", ".join(tmeta.columns.keys())
            schema_lines.append(f"- {tname}({col_names})")
        schema_block = "\n".join(schema_lines)
    else:
        schema_block = (
            "Schema:\n"
            "- CLAIMS(CLAIM_ID, POLICY_ID, CLAIMANT_ID, CLM_STAT_CD, LOSS_DT, REPORT_DT,\n"
            "          LOSS_TYPE_CD, INCURRED_AMT, RESERVE_AMT, ADJUSTER_ID, CLOSE_DT, LITIGATION_FLG)\n"
            "- POLICY(POLICY_ID, POLICY_NBR, INSURED_NM, POL_EFF_DT, POL_EXP_DT, LINE_OF_BUSNSS,\n"
            "          STATE_CD, PREMIUM_AMT, DEDUCTIBLE_AMT, AGENT_ID, POL_STAT_CD)\n"
            "- PAYMENT(PAYMENT_ID, CLAIM_ID, PMT_DT, PMT_AMT_GROSS, PMT_AMT_NET, PMT_STAT_CD,\n"
            "           PMT_TYPE_CD, PAYEE_NM, CHK_NBR, VOID_RSN_CD)\n"
            "- CLAIMANT(CLAIMANT_ID, CLAIMANT_NM, DOB, GENDER_CD, ADDRESS_LINE1, STATE_CD,\n"
            "            CONTACT_PHONE, ATTY_REP_FLG, CLAIM_COUNT, FRAUD_RISK_SCRE)"
        )

    return (
        base
        + schema_block
        + "\n\nAll dates stored as TEXT in ISO format (YYYY-MM-DD). "
        "Use date() for SQLite date math.\n"
        "Only SQLite-compatible syntax — no Oracle ROWNUM, SYSDATE, or NVL."
    )


@dataclass
class ExecutionResult:
    success: bool
    df: Optional[pd.DataFrame] = None
    sql_executed: str = ""
    error: str = ""
    attempts: int = 1
    repair_history: List[Dict[str, str]] = field(default_factory=list)


class SQLExecutor:
    def __init__(self, cfg: Config, registry=None) -> None:
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)
        self.db_path = cfg.sqlite_db_path
        self._repair_system = _build_repair_system(registry)
        print(f"\n[SQLExecutor INIT] db_path={self.db_path}")
        print(f"  Repair system prompt length: {len(self._repair_system)} chars")

    def execute(
        self, sql: str, question: str, max_retries: int = 5
    ) -> ExecutionResult:
        print(f"\n{'='*60}")
        print(f"[Stage 5 INPUT] SQLExecutor.execute()")
        print(f"  Question   : {question}")
        print(f"  SQL to run :\n{sql}")
        print(f"  max_retries: {max_retries}")
        print(f"{'='*60}")

        current_sql = sql
        repair_history: List[Dict[str, str]] = []

        for attempt in range(1, max_retries + 2):
            print(f"\n[Stage 5] Attempt {attempt}/{max_retries+1}")
            try:
                df = self._run_sql(current_sql)
                print(f"[Stage 5 OUTPUT] SQL succeeded on attempt {attempt}")
                print(f"  Result shape: {df.shape[0]} rows × {df.shape[1]} cols")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Preview:\n{df.head(5).to_string(index=False)}")
                return ExecutionResult(
                    success=True,
                    df=df,
                    sql_executed=current_sql,
                    attempts=attempt,
                    repair_history=repair_history,
                )
            except Exception as e:
                error_msg = str(e)
                logger.warning("SQL attempt %d failed: %s", attempt, error_msg)
                print(f"[Stage 5 ERROR] Attempt {attempt} failed: {error_msg}")
                repair_history.append({"sql": current_sql, "error": error_msg})

                if attempt > max_retries:
                    break

                print(f"[Stage 5] Invoking repair agent...")
                current_sql = self._repair(question, current_sql, error_msg)
                print(f"[Stage 5] Repaired SQL:\n{current_sql}")

        print(f"[Stage 5 FAILED] All {max_retries+1} attempts exhausted.")
        return ExecutionResult(
            success=False,
            sql_executed=current_sql,
            error=repair_history[-1]["error"] if repair_history else "Unknown error",
            attempts=max_retries + 1,
            repair_history=repair_history,
        )

    def _run_sql(self, sql: str) -> pd.DataFrame:
        print(f"  [_run_sql] Connecting to: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(sql, conn)
            print(f"  [_run_sql] Query returned {len(df)} rows")
            return df
        finally:
            conn.close()

    def _repair(self, question: str, failing_sql: str, error: str) -> str:
        print(f"\n[Stage 5 REPAIR INPUT]")
        print(f"  question   : {question}")
        print(f"  failing_sql: {failing_sql}")
        print(f"  error      : {error}")
        user_prompt = (
            f"Original question: {question}\n\n"
            f"Failing SQL:\n{failing_sql}\n\n"
            f"Error: {error}\n\n"
            "Return the corrected SQL only."
        )
        repaired = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            self._repair_system, user_prompt,
            temperature=0.0, max_tokens=800,
        )
        repaired = (
            repaired.strip()
            .removeprefix("```sql").removeprefix("```")
            .removesuffix("```").strip()
        )
        print(f"[Stage 5 REPAIR OUTPUT] Repaired SQL:\n{repaired}")
        logger.info("Repair agent produced:\n%s", repaired)
        return repaired
    

    
