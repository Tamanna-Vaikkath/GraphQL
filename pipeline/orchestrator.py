"""
pipeline/orchestrator.py — End-to-end NLQ pipeline orchestrator.

Ties all 5 stages together:
  Stage 1 → HyDE expansion
  Stage 2 → KG vector retrieval + LLM re-rank
  Stage 3 → FK join path extraction
  Stage 4 → SQL generation
  Stage 5 → SQL execution + self-healing

Returns a structured QueryResult usable by the Streamlit UI.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import Config, load_config
from pipeline.hyde_expander import LLMHyDEExpander, SchemaGroundingError
from kg.retriever import KGRetriever, RetrievalResult
from pipeline.sql_generator import SQLGenerator, SQLColumnNotFoundError
from pipeline.executor import SQLExecutor, ExecutionResult
from pipeline.summarizer import ResultSummarizer
from utils.llm_client import GPTClient   

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.75  # Re-expansion trigger (Section 4, Stage 2)
MAX_EXPAND_RETRIES = 2


@dataclass
class QueryTrace:
    original_question: str = ""
    expanded_query: str = ""
    max_retrieval_score: float = 0.0
    retrieved_columns: list = field(default_factory=list)
    join_conditions: list = field(default_factory=list)
    generated_sql: str = ""
    sql_repair_attempts: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class QueryResult:
    success: bool
    df: Optional[pd.DataFrame]
    summary: str
    trace: QueryTrace
    error: str = ""
    schema_grounding_error: bool = False   # True → question has no DB column mapping


class NLQPipeline:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()

        # Build a GPTClient adapter first — it exposes the .complete() method
        # that LLMHyDEExpander (and any other prompt-based component) expects.
        # Previously cfg was passed directly, which caused:
        #   AttributeError: 'Config' object has no attribute 'complete'
        llm = GPTClient(self.cfg)

        self.expander = LLMHyDEExpander(llm)         
        self.retriever = KGRetriever(self.cfg)
        self.sql_gen = SQLGenerator(self.cfg)
        self.executor = SQLExecutor(self.cfg)
        self.summarizer = ResultSummarizer(self.cfg)

    def query(self, question: str, generate_summary: bool = True) -> QueryResult:
        """Run the full 5-stage pipeline for a natural language question."""
        t0 = time.time()
        trace = QueryTrace(original_question=question)

        try:
            # ── Stage 1: HyDE Expansion ───────────────────────────────────────
            logger.info("[Stage 1] HyDE expansion...")
            try:
                expanded = self.expander.expand_or_raise(question)
            except SchemaGroundingError as sge:
                logger.warning("[Stage 1] Schema grounding failed: %s", sge)
                trace.elapsed_seconds = round(time.time() - t0, 2)
                return QueryResult(
                    success=False,
                    df=None,
                    summary="",
                    trace=trace,
                    error=str(sge),
                    schema_grounding_error=True,
                )
            trace.expanded_query = expanded

            # ── Stage 2+3: KG Retrieval with retry loop ───────────────────────
            logger.info("[Stage 2+3] KG retrieval...")
            retrieval: RetrievalResult = self.retriever.retrieve(expanded)

            if retrieval.max_score < RELEVANCE_THRESHOLD:
                logger.info(
                    "Score %.3f < threshold %.2f — re-expanding (strict)...",
                    retrieval.max_score, RELEVANCE_THRESHOLD
                )
                for _ in range(MAX_EXPAND_RETRIES):
                    expanded = self.expander.expand(question)
                    retrieval = self.retriever.retrieve(expanded)
                    trace.expanded_query = expanded
                    if retrieval.max_score >= RELEVANCE_THRESHOLD:
                        break

            trace.max_retrieval_score = retrieval.max_score
            trace.retrieved_columns = [
                {"table": c.table, "name": c.name, "score": round(c.score, 3)}
                for c in retrieval.columns
            ]
            trace.join_conditions = retrieval.join_conditions

            # ── Stage 4: SQL Generation ───────────────────────────────────────
            logger.info("[Stage 4] Generating SQL...")
            try:
                sql = self.sql_gen.generate(question, retrieval)
            except SQLColumnNotFoundError as col_err:
                logger.warning("[Stage 4] Column validation failed: %s", col_err)
                trace.elapsed_seconds = round(time.time() - t0, 2)
                return QueryResult(
                    success=False,
                    df=None,
                    summary="",
                    trace=trace,
                    error=str(col_err),
                    schema_grounding_error=True,  # reuse this flag → same UI path
                )
            trace.generated_sql = sql

            # ── Stage 5: Execute + Self-Heal ──────────────────────────────────
            logger.info("[Stage 5] Executing SQL...")
            exec_result: ExecutionResult = self.executor.execute(sql, question)
            trace.sql_repair_attempts = exec_result.attempts - 1
            trace.generated_sql = exec_result.sql_executed  # may be repaired

            trace.elapsed_seconds = round(time.time() - t0, 2)

            if not exec_result.success:
                return QueryResult(
                    success=False, df=None, summary="",
                    trace=trace, error=exec_result.error
                )

            # ── Optional AI Summary ───────────────────────────────────────────
            summary = ""
            if generate_summary and exec_result.df is not None:
                logger.info("[Summary] Generating AI narrative...")
                try:
                    summary = self.summarizer.summarize(question, exec_result.df)
                except Exception as e:
                    logger.warning("Summarizer failed: %s", e)

            return QueryResult(
                success=True,
                df=exec_result.df,
                summary=summary,
                trace=trace,
            )

        except Exception as e:
            logger.exception("Pipeline error: %s", e)
            trace.elapsed_seconds = round(time.time() - t0, 2)
            return QueryResult(
                success=False, df=None, summary="",
                trace=trace, error=str(e)
            )

    def close(self):
        self.retriever.close()