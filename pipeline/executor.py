"""
pipeline/executor.py — Stage 5: SQL Execution + Self-Healing Repair Agent.

Repair-agent design notes
--------------------------
Every retry cycle is captured as a `RepairAttempt` record containing:
  - the exact SQL that failed and the exact DB error it raised
  - the *complete* system + user prompt sent to the repair LLM for that cycle
    (not just a summary — the literal strings, so a failure can be replayed)
  - the SQL the repair LLM returned
  - a unified diff between the failing SQL and the repaired SQL, so it's
    obvious *what changed* and therefore *why* the repair should fix the error
  - validation results (did the repair pass basic sanity checks, or did it
    trip a termination condition such as "identical to what just failed" or
    "we've already tried this exact SQL before / repair loop detected")

`ExecutionResult.full_trace` is a human-readable, pre-formatted report of
every attempt end-to-end. It is always populated when repairs occurred, and
is printed automatically when the overall execution ultimately fails, so a
failure always comes with the complete repair history attached — prompts,
SQL, errors, diffs, and the reason retries stopped.
"""
from __future__ import annotations

import difflib
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)


def _build_repair_system(registry=None) -> str:
    base = (
        "You are a SQL repair agent for a SQLite database.\n"
        "You will receive a failing SQL query, its error, and (if this is not "
        "the first attempt) the full history of every SQL/error pair already "
        "tried for this question.\n"
        "Fix the SQL and return ONLY the corrected SQL.\n"
        "Do not repeat any SQL string that appears in the history below — if "
        "an approach already failed, try a materially different fix.\n"
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
class RepairAttempt:
    """Full record of a single repair cycle — everything needed to explain,
    reproduce, or debug that cycle after the fact."""

    attempt_number: int                 # which failing execution this repair is fixing (1-indexed)
    failing_sql: str                    # the SQL that just failed
    error: str                          # the exact DB error raised by failing_sql
    repair_system_prompt: str           # exact system prompt sent to the repair LLM
    repair_user_prompt: str             # exact user prompt sent to the repair LLM
    repaired_sql: str = ""              # SQL returned by the repair LLM (post-cleanup)
    sql_diff: str = ""                  # unified diff: failing_sql -> repaired_sql
    validation_passed: bool = True      # did the repaired SQL pass sanity checks?
    validation_notes: List[str] = field(default_factory=list)  # why it passed/failed
    terminated_retries: bool = False    # did this attempt trigger early termination?
    termination_reason: str = ""        # human-readable reason, if terminated_retries

    def render(self) -> str:
        """Render this attempt as a readable block for the full trace report."""
        lines = [
            f"{'-'*70}",
            f"REPAIR CYCLE #{self.attempt_number}",
            f"{'-'*70}",
            "[Failing SQL]",
            self.failing_sql,
            "",
            "[Error]",
            self.error,
            "",
            "[Exact repair prompt — system]",
            self.repair_system_prompt,
            "",
            "[Exact repair prompt — user]",
            self.repair_user_prompt,
            "",
            "[Repaired SQL returned by LLM]",
            self.repaired_sql or "(empty response)",
            "",
            "[What changed — diff of failing SQL -> repaired SQL]",
            self.sql_diff or "(no diff — SQL unchanged)",
            "",
            f"[Validation] passed={self.validation_passed}",
        ]
        for note in self.validation_notes:
            lines.append(f"  - {note}")
        if self.terminated_retries:
            lines.append(f"[RETRY LOOP TERMINATED] {self.termination_reason}")
        return "\n".join(lines)


@dataclass
class ExecutionResult:
    success: bool
    df: Optional[pd.DataFrame] = None
    sql_executed: str = ""
    error: str = ""
    attempts: int = 1
    repair_history: List[Dict[str, str]] = field(default_factory=list)
    repair_attempts: List[RepairAttempt] = field(default_factory=list)
    termination_reason: str = ""
    full_trace: str = ""


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
        repair_attempts: List[RepairAttempt] = []
        # every SQL string we've already tried (failing + repaired), used for
        # loop detection so the repair agent can't get stuck re-proposing the
        # same broken fix over and over.
        tried_sql: List[str] = [current_sql]
        termination_reason = ""

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
                    repair_attempts=repair_attempts,
                    full_trace=self._render_trace(
                        repair_attempts, final_status="SUCCESS", attempts=attempt
                    ) if repair_attempts else "",
                )
            except Exception as e:
                error_msg = str(e)
                logger.warning("SQL attempt %d failed: %s", attempt, error_msg)
                print(f"[Stage 5 ERROR] Attempt {attempt} failed: {error_msg}")
                repair_history.append({"sql": current_sql, "error": error_msg})

                # ---- Retry termination criterion 1: max_retries exhausted ----
                if attempt > max_retries:
                    termination_reason = (
                        f"max_retries ({max_retries}) exhausted after {attempt} "
                        f"total execution attempts."
                    )
                    print(f"[Stage 5] Termination criterion hit: {termination_reason}")
                    break

                print(f"[Stage 5] Invoking repair agent...")
                repair_attempt = self._repair(
                    question=question,
                    failing_sql=current_sql,
                    error=error_msg,
                    attempt_number=attempt,
                    full_history=repair_history,
                )
                repair_attempts.append(repair_attempt)
                print(f"[Stage 5] Repaired SQL:\n{repair_attempt.repaired_sql}")
                print(repair_attempt.render())

                # ---- Retry termination criterion 2: repair returned nothing ----
                if not repair_attempt.repaired_sql.strip():
                    termination_reason = (
                        "Repair agent returned an empty SQL string — cannot "
                        "continue retrying."
                    )
                    repair_attempt.terminated_retries = True
                    repair_attempt.termination_reason = termination_reason
                    print(f"[Stage 5] Termination criterion hit: {termination_reason}")
                    break

                # ---- Retry termination criterion 3: repair loop detected ----
                # The repaired SQL is byte-identical (modulo whitespace/case)
                # to something we've already tried — either the SQL that just
                # failed, or any earlier attempt in this same repair chain.
                # Retrying it would just reproduce the same error forever.
                normalized_repaired = " ".join(repair_attempt.repaired_sql.split()).upper()
                normalized_tried = [" ".join(s.split()).upper() for s in tried_sql]
                if normalized_repaired in normalized_tried:
                    termination_reason = (
                        "Repair agent proposed SQL that is identical to a "
                        "previously attempted query (repair loop detected) — "
                        "stopping to avoid an infinite retry cycle."
                    )
                    repair_attempt.terminated_retries = True
                    repair_attempt.termination_reason = termination_reason
                    repair_attempt.validation_passed = False
                    repair_attempt.validation_notes.append(termination_reason)
                    print(f"[Stage 5] Termination criterion hit: {termination_reason}")
                    break

                current_sql = repair_attempt.repaired_sql
                tried_sql.append(current_sql)

        print(f"[Stage 5 FAILED] Retries stopped: {termination_reason}")
        full_trace = self._render_trace(
            repair_attempts,
            final_status="FAILED",
            attempts=len(repair_history),
            termination_reason=termination_reason,
        )
        print("\n" + full_trace)
        return ExecutionResult(
            success=False,
            sql_executed=current_sql,
            error=repair_history[-1]["error"] if repair_history else "Unknown error",
            attempts=len(repair_history) if repair_history else 1,
            repair_history=repair_history,
            repair_attempts=repair_attempts,
            termination_reason=termination_reason,
            full_trace=full_trace,
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

    def _repair(
        self,
        question: str,
        failing_sql: str,
        error: str,
        attempt_number: int,
        full_history: List[Dict[str, str]],
    ) -> RepairAttempt:
        """Run one repair cycle.

        The user prompt includes the *entire* history of SQL/error pairs
        tried so far for this question (not just the most recent failure),
        so the repair LLM can see what has already been ruled out and is
        pushed toward a materially different fix rather than re-deriving
        (and re-failing) the same approach.
        """
        print(f"\n[Stage 5 REPAIR INPUT] cycle #{attempt_number}")
        print(f"  question   : {question}")
        print(f"  failing_sql: {failing_sql}")
        print(f"  error      : {error}")
        print(f"  prior attempts in history: {len(full_history) - 1}")

        history_block = ""
        prior = full_history[:-1]  # everything before this current failure
        if prior:
            history_lines = ["Previously attempted (and failed) SQL for this question:"]
            for i, h in enumerate(prior, start=1):
                history_lines.append(
                    f"  Attempt {i} SQL:\n{h['sql']}\n  Attempt {i} Error: {h['error']}\n"
                )
            history_block = "\n".join(history_lines) + "\n\n"

        user_prompt = (
            f"Original question: {question}\n\n"
            f"{history_block}"
            f"Failing SQL (most recent attempt):\n{failing_sql}\n\n"
            f"Error: {error}\n\n"
            "Return the corrected SQL only. Do not reuse any SQL shown above "
            "as already-failed."
        )

        repaired_raw = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            self._repair_system, user_prompt,
            temperature=0.0, max_tokens=800,
        )
        repaired = (
            repaired_raw.strip()
            .removeprefix("```sql").removeprefix("```")
            .removesuffix("```").strip()
        )
        print(f"[Stage 5 REPAIR OUTPUT] Repaired SQL:\n{repaired}")
        logger.info("Repair agent produced:\n%s", repaired)

        diff = self._diff_sql(failing_sql, repaired)
        validation_passed, validation_notes = self._validate_repair(
            failing_sql=failing_sql, repaired_sql=repaired
        )

        return RepairAttempt(
            attempt_number=attempt_number,
            failing_sql=failing_sql,
            error=error,
            repair_system_prompt=self._repair_system,
            repair_user_prompt=user_prompt,
            repaired_sql=repaired,
            sql_diff=diff,
            validation_passed=validation_passed,
            validation_notes=validation_notes,
        )

    @staticmethod
    def _diff_sql(old_sql: str, new_sql: str) -> str:
        """Unified diff explaining exactly what the repair changed — this is
        the concrete evidence for *why* the repair should resolve the error
        (e.g. a bad column swapped for a valid one, a missing JOIN added,
        quoting/date-function syntax fixed, etc.)."""
        diff_lines = list(
            difflib.unified_diff(
                old_sql.splitlines(),
                new_sql.splitlines(),
                fromfile="failing_sql",
                tofile="repaired_sql",
                lineterm="",
            )
        )
        return "\n".join(diff_lines)

    @staticmethod
    def _validate_repair(failing_sql: str, repaired_sql: str) -> Tuple[bool, List[str]]:
        """Basic sanity checks run on every repaired SQL before it is retried.
        These don't guarantee the SQL will execute successfully — that's
        confirmed by actually running it next loop iteration — but they catch
        obviously-bad repairs early and are recorded in the trace either way.
        """
        notes: List[str] = []
        passed = True

        if not repaired_sql.strip():
            notes.append("Repaired SQL is empty.")
            return False, notes

        normalized_old = " ".join(failing_sql.split()).upper()
        normalized_new = " ".join(repaired_sql.split()).upper()
        if normalized_old == normalized_new:
            notes.append(
                "Repaired SQL is identical to the failing SQL — no actual "
                "change was made, so the same error is expected to recur."
            )
            passed = False
        else:
            notes.append("Repaired SQL differs from the failing SQL (see diff).")

        if "```" in repaired_sql:
            notes.append("Repaired SQL still contains markdown fences.")
            passed = False

        first_token = repaired_sql.strip().split(None, 1)[0].upper() if repaired_sql.strip() else ""
        if first_token not in ("SELECT", "WITH"):
            notes.append(
                f"Repaired SQL does not start with SELECT/WITH (starts with "
                f"'{first_token}') — unexpected for a read-only query."
            )
            passed = False
        else:
            notes.append(f"Repaired SQL starts with '{first_token}' as expected.")

        return passed, notes

    @staticmethod
    def _render_trace(
        repair_attempts: List[RepairAttempt],
        final_status: str,
        attempts: int,
        termination_reason: str = "",
    ) -> str:
        """Build the full, human-readable trace of every repair cycle. This
        is what gets logged/returned whenever execution ultimately fails, so
        the entire repair history — every prompt, every SQL variant, every
        error, and why retries stopped — is available in one place."""
        header = [
            f"{'='*70}",
            f"SQL EXECUTION + REPAIR TRACE",
            f"{'='*70}",
            f"Final status      : {final_status}",
            f"Total attempts     : {attempts}",
            f"Repair cycles run  : {len(repair_attempts)}",
        ]
        if termination_reason:
            header.append(f"Retry termination  : {termination_reason}")
        header.append("")

        body = [ra.render() for ra in repair_attempts]

        footer = [f"{'='*70}"]

        return "\n".join(header + body + footer)