"""
pipeline/executor.py — Stage 5: SQL Execution + Self-Healing Repair Agent.

Executes the generated SQL against the SQLite database.
On failure, a repair agent receives the error and rewrites the SQL (up to 5 retries).

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

REPAIR_SYSTEM = """You are a SQL repair agent for a SQLite P&C insurance database.
You will receive a failing SQL query and its error. Fix the SQL and return ONLY the corrected SQL.
No explanation, no markdown fences.

Schema:
- CLAIMS(CLAIM_ID, POLICY_ID, CLAIMANT_ID, CLM_STAT_CD, LOSS_DT, REPORT_DT,
          LOSS_TYPE_CD, INCURRED_AMT, RESERVE_AMT, ADJUSTER_ID, CLOSE_DT, LITIGATION_FLG)
- POLICY(POLICY_ID, POLICY_NBR, INSURED_NM, POL_EFF_DT, POL_EXP_DT, LINE_OF_BUSNSS,
          STATE_CD, PREMIUM_AMT, DEDUCTIBLE_AMT, AGENT_ID, POL_STAT_CD)
- PAYMENT(PAYMENT_ID, CLAIM_ID, PMT_DT, PMT_AMT_GROSS, PMT_AMT_NET, PMT_STAT_CD,
           PMT_TYPE_CD, PAYEE_NM, CHK_NBR, VOID_RSN_CD)
- CLAIMANT(CLAIMANT_ID, CLAIMANT_NM, DOB, GENDER_CD, ADDRESS_LINE1, STATE_CD,
            CONTACT_PHONE, ATTY_REP_FLG, CLAIM_COUNT, FRAUD_RISK_SCRE)

All dates stored as TEXT in ISO format (YYYY-MM-DD). Use date() for SQLite date math.
Only SQLite-compatible syntax — no Oracle ROWNUM, SYSDATE, or NVL."""


@dataclass
class ExecutionResult:
    success: bool
    df: Optional[pd.DataFrame] = None
    sql_executed: str = ""
    error: str = ""
    attempts: int = 1
    repair_history: List[Dict[str, str]] = field(default_factory=list)


class SQLExecutor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)
        self.db_path = cfg.sqlite_db_path

    def execute(self, sql: str, question: str, max_retries: int = 5) -> ExecutionResult:
        """Execute SQL with self-healing repair on failure.

        Args:
            sql: Generated SQL to execute.
            question: Original user question (for repair context).
            max_retries: Max repair attempts before surfacing error.
        Returns:
            ExecutionResult with DataFrame on success.
        """
        current_sql = sql
        repair_history = []

        for attempt in range(1, max_retries + 2):  # +1 for initial attempt
            try:
                df = self._run_sql(current_sql)
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
                repair_history.append({"sql": current_sql, "error": error_msg})

                if attempt > max_retries:
                    break

                # Invoke repair agent
                current_sql = self._repair(question, current_sql, error_msg)

        return ExecutionResult(
            success=False,
            sql_executed=current_sql,
            error=repair_history[-1]["error"] if repair_history else "Unknown error",
            attempts=max_retries + 1,
            repair_history=repair_history,
        )

    def _run_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return result as DataFrame."""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(sql, conn)
            return df
        finally:
            conn.close()

    def _repair(self, question: str, failing_sql: str, error: str) -> str:
        """Ask the repair agent to fix the SQL."""
        user_prompt = (
            f"Original question: {question}\n\n"
            f"Failing SQL:\n{failing_sql}\n\n"
            f"Error: {error}\n\n"
            "Return the corrected SQL only."
        )
        repaired = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            REPAIR_SYSTEM, user_prompt,
            temperature=0.0, max_tokens=800
        )
        repaired = repaired.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        logger.info("Repair agent produced:\n%s", repaired)
        return repaired
