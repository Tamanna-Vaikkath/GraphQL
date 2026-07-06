"""
pipeline/orchestrator.py — End-to-end NLQ pipeline orchestrator.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from config import Config, load_config
from kg.schema_registry import SchemaRegistry
from kg.schema_retriever import SchemaRetriever, ScopedSchema
from pipeline.hyde_expander import (
    LLMHyDEExpander, SchemaGroundingError, PromptStrategy,
    inject_registry,
)
from kg.retriever import KGRetriever, RetrievalResult
from pipeline.sql_generator import SQLGenerator, SQLColumnNotFoundError
from pipeline.executor import SQLExecutor, ExecutionResult
from pipeline.summarizer import ResultSummarizer
from utils.llm_client import GPTClient

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.75
MAX_EXPAND_RETRIES  = 2


@dataclass
class QueryTrace:
    original_question: str = ""
    scoped_tables: List[str] = field(default_factory=list)
    scope_retrieval_method: str = ""
    table_scores: dict = field(default_factory=dict)
    expanded_query: str = ""
    max_retrieval_score: float = 0.0
    retrieved_columns: list = field(default_factory=list)
    join_conditions: list = field(default_factory=list)
    generated_sql: str = ""
    sql_repair_attempts: int = 0
    repair_history: List[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class QueryResult:
    success: bool
    df: Optional[pd.DataFrame]
    summary: str
    trace: QueryTrace
    error: str = ""
    schema_grounding_error: bool = False


class NLQPipeline:
    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or load_config()

        print("\n" + "#"*70)
        print("# [NLQPipeline INIT] Initialising all pipeline components...")
        print("#"*70)

        logger.info("Loading SchemaRegistry...")
        self.registry = SchemaRegistry.load(self.cfg)
        print(f"[NLQPipeline INIT] SchemaRegistry loaded:")
        print(f"  Tables      : {len(self.registry.table_names)} → {self.registry.table_names}")
        print(f"  FK edges    : {len(self.registry.fk_edges)}")
        print(f"  Domain rules: {len(self.registry.domain_rules)}")
        logger.info(
            "Registry ready: %d tables, %d FK edges, %d domain rules",
            len(self.registry.table_names),
            len(self.registry.fk_edges),
            len(self.registry.domain_rules),
        )

        inject_registry(self.registry)

        llm = GPTClient(self.cfg)

        self.schema_retriever = SchemaRetriever(
            cfg=self.cfg,
            registry=self.registry,
            top_k_tables=6,
            embedding_threshold=0.40,
        )
        print("[NLQPipeline INIT] SchemaRetriever ready (top_k=6, threshold=0.40)")

        self.expander = LLMHyDEExpander(
            llm_client=llm,
            registry=self.registry,
            strategy=PromptStrategy.CONCISE,
        )
        print("[NLQPipeline INIT] LLMHyDEExpander ready (strategy=CONCISE)")

        self.retriever = KGRetriever(cfg=self.cfg, registry=self.registry)
        print("[NLQPipeline INIT] KGRetriever ready")

        self.sql_gen = SQLGenerator(
            cfg=self.cfg,
            registry=self.registry,
            scoped_schema=None,
        )
        print("[NLQPipeline INIT] SQLGenerator ready")

        self.executor   = SQLExecutor(self.cfg)
        self.summarizer = ResultSummarizer(self.cfg)
        print("[NLQPipeline INIT] SQLExecutor + ResultSummarizer ready")
        print("#"*70 + "\n")

    def query(self, question: str, generate_summary: bool = True) -> QueryResult:
        print("\n" + "★"*70)
        print(f"[PIPELINE START] question={repr(question)}")
        print(f"                 generate_summary={generate_summary}")
        print("★"*70)

        t0 = time.time()
        trace = QueryTrace(original_question=question)

        try:
            # ── Stage 0 ──────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 0] Scoped schema retrieval")
            print(f"{'─'*60}")
            logger.info("[Stage 0] Scoped schema retrieval...")
            scoped: ScopedSchema = self.schema_retriever.retrieve_scoped_schema(question)
            trace.scoped_tables          = scoped.relevant_tables
            trace.scope_retrieval_method = scoped.retrieval_method
            trace.table_scores           = scoped.table_scores

            print(f"[Stage 0 OUTPUT]")
            print(f"  Retrieval method : {scoped.retrieval_method}")
            print(f"  Relevant tables  : {scoped.relevant_tables}")
            print(f"  Table scores     : { {k: round(v,3) for k,v in scoped.table_scores.items()} }")
            print(f"  Scoped columns   : {len(scoped.scoped_all_columns)}")
            print(f"  FK edges in scope: {len(scoped.scoped_fk_edges)}")

            logger.info(
                "[Stage 0] Scoped to %d tables via %s: %s",
                len(scoped.relevant_tables), scoped.retrieval_method, scoped.relevant_tables
            )

            self.sql_gen.update_scoped_schema(scoped)

            # ── Stage 1 ──────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 1] HyDE expansion")
            print(f"{'─'*60}")
            print(f"[Stage 1 INPUT] question={repr(question)}")
            logger.info("[Stage 1] HyDE expansion with scoped schema...")

            scoped_expander = _make_scoped_expander(self.expander, scoped, self.cfg)

            try:
                expanded = scoped_expander.expand_or_raise(question)
            except SchemaGroundingError as sge:
                logger.warning("[Stage 1] Schema grounding failed: %s", sge)
                print(f"[Stage 1 ERROR] SchemaGroundingError: {sge}")
                trace.elapsed_seconds = round(time.time() - t0, 2)
                return QueryResult(
                    success=False, df=None, summary="", trace=trace,
                    error=str(sge), schema_grounding_error=True,
                )
            trace.expanded_query = expanded
            print(f"[Stage 1 OUTPUT] Expanded query:\n  {expanded}")

            # ── Stage 2+3 ─────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 2+3] KG scoped retrieval")
            print(f"{'─'*60}")
            print(f"[Stage 2+3 INPUT] expanded_query={repr(expanded)}")
            logger.info("[Stage 2+3] KG scoped retrieval...")
            retrieval: RetrievalResult = self.retriever.retrieve_scoped(expanded, scoped)

            if retrieval.max_score < RELEVANCE_THRESHOLD:
                print(f"[Stage 2+3] Score {retrieval.max_score:.3f} < threshold {RELEVANCE_THRESHOLD} — re-expanding...")
                logger.info(
                    "[Stage 2+3] Score %.3f < threshold %.2f — re-expanding...",
                    retrieval.max_score, RELEVANCE_THRESHOLD,
                )
                for retry_i in range(MAX_EXPAND_RETRIES):
                    expanded = scoped_expander.expand(question)
                    retrieval = self.retriever.retrieve_scoped(expanded, scoped)
                    trace.expanded_query = expanded
                    print(f"[Stage 2+3] Retry {retry_i+1}: new score={retrieval.max_score:.3f}, query={repr(expanded)}")
                    if retrieval.max_score >= RELEVANCE_THRESHOLD:
                        break

            trace.max_retrieval_score = retrieval.max_score
            trace.retrieved_columns   = [
                {"table": c.table, "name": c.name, "score": round(c.score, 3)}
                for c in retrieval.columns
            ]
            trace.join_conditions = retrieval.join_conditions

            print(f"[Stage 2+3 OUTPUT]")
            print(f"  Max score      : {retrieval.max_score:.4f}")
            print(f"  Tables involved: {retrieval.tables_involved}")
            print(f"  Columns (top 5):")
            for c in retrieval.columns[:5]:
                print(f"    {c.table}.{c.name} score={c.score:.3f}")
            print(f"  Join conditions: {retrieval.join_conditions}")

            # ── Stage 4 ──────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 4] SQL Generation")
            print(f"{'─'*60}")
            print(f"[Stage 4 INPUT] question={repr(question)}")
            logger.info("[Stage 4] Generating SQL with scoped domain rules...")
            try:
                sql = self.sql_gen.generate(question, retrieval)
            except SQLColumnNotFoundError as col_err:
                logger.warning("[Stage 4] Column validation failed: %s", col_err)
                print(f"[Stage 4 ERROR] SQLColumnNotFoundError: {col_err}")
                trace.elapsed_seconds = round(time.time() - t0, 2)
                return QueryResult(
                    success=False, df=None, summary="", trace=trace,
                    error=str(col_err), schema_grounding_error=True,
                )
            trace.generated_sql = sql
            print(f"[Stage 4 OUTPUT] Generated SQL:\n{sql}")

            # ── Stage 5 ──────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 5] SQL Execution")
            print(f"{'─'*60}")
            logger.info("[Stage 5] Executing SQL...")
            exec_result: ExecutionResult = self.executor.execute(sql, question)
            trace.sql_repair_attempts = exec_result.attempts - 1
            trace.repair_history      = exec_result.repair_history
            trace.generated_sql       = exec_result.sql_executed
            trace.elapsed_seconds     = round(time.time() - t0, 2)

            print(f"[Stage 5 RESULT] success={exec_result.success}, attempts={exec_result.attempts}")
            if exec_result.repair_history:
                print(f"  Repair history ({len(exec_result.repair_history)} repairs):")
                for i, r in enumerate(exec_result.repair_history):
                    print(f"    Repair {i+1}: error={r['error'][:100]}")

            if not exec_result.success:
                print(f"[Stage 5 FAILED] error={exec_result.error}")
                return QueryResult(
                    success=False, df=None, summary="",
                    trace=trace, error=exec_result.error,
                )

            # ── Summary ───────────────────────────────────────────────────────
            summary = ""
            if generate_summary and exec_result.df is not None:
                print(f"\n{'─'*60}")
                print(f"[Summary] Generating AI narrative")
                print(f"{'─'*60}")
                logger.info("[Summary] Generating AI narrative...")
                try:
                    summary = self.summarizer.summarize(question, exec_result.df)
                except Exception as e:
                    logger.warning("Summarizer failed: %s", e)
                    print(f"[Summary ERROR] {e}")

            print(f"\n{'★'*70}")
            print(f"[PIPELINE DONE] success=True | elapsed={trace.elapsed_seconds}s")
            print(f"  Rows returned   : {len(exec_result.df) if exec_result.df is not None else 0}")
            print(f"  SQL repairs     : {trace.sql_repair_attempts}")
            print(f"  Scoped tables   : {trace.scoped_tables}")
            print(f"  Final SQL:\n{trace.generated_sql}")
            print(f"{'★'*70}\n")

            return QueryResult(
                success=True,
                df=exec_result.df,
                summary=summary,
                trace=trace,
            )

        except Exception as e:
            logger.exception("Pipeline error: %s", e)
            trace.elapsed_seconds = round(time.time() - t0, 2)
            print(f"\n[PIPELINE ERROR] Unhandled exception: {e}")
            return QueryResult(
                success=False, df=None, summary="",
                trace=trace, error=str(e),
            )

    def close(self) -> None:
        self.retriever.close()
        self.schema_retriever.close()


def _make_scoped_expander(
    base_expander: LLMHyDEExpander,
    scoped: ScopedSchema,
    cfg: Config,
) -> LLMHyDEExpander:
    print(f"\n[_make_scoped_expander] Building scoped expander for tables: {scoped.relevant_tables}")
    from utils.llm_client import GPTClient
    from pipeline.hyde_expander import PromptStrategy, _PROMPT_TEMPLATES

    llm = base_expander._client
    strategy = base_expander._strategy

    template = _PROMPT_TEMPLATES[strategy]
    scoped_system = template.format(schema_block=scoped.scoped_schema_block)
    print(f"[_make_scoped_expander] Scoped system prompt ({len(scoped_system)} chars)")

    scoped_exp = LLMHyDEExpander.__new__(LLMHyDEExpander)
    scoped_exp._client         = llm
    scoped_exp._max_sentences  = base_expander._max_sentences
    scoped_exp._strategy       = strategy
    scoped_exp._strict         = base_expander._strict
    scoped_exp._system         = scoped_system
    scoped_exp._schema_block   = scoped.scoped_schema_block
    scoped_exp._all_columns    = scoped.scoped_all_columns
    scoped_exp._manifest       = scoped.scoped_manifest
    scoped_exp._table_names    = scoped.relevant_tables
    scoped_exp._registry       = None

    from kg.schema_registry import SchemaRegistry
    from pipeline.hyde_expander import SchemaValidator
    mini_concept_map: dict = {}
    for tname, cols in scoped.scoped_manifest.items():
        tl = tname.lower()
        pk_cols = [c for c in cols if c.endswith("_ID") and not c[:-3].endswith("_")]
        mini_concept_map.setdefault(tl, pk_cols or [list(cols.keys())[0]])
        mini_concept_map.setdefault(tl + "s", mini_concept_map[tl])
        for cname in cols:
            readable = cname.lower().replace("_", " ").replace(" cd", "").replace(" amt", " amount")
            mini_concept_map.setdefault(readable, [cname])

    scoped_exp._concept_map = mini_concept_map
    scoped_exp._validator   = SchemaValidator(scoped.scoped_all_columns)
    print(f"[_make_scoped_expander] Scoped expander ready (concept_map entries={len(mini_concept_map)})")
    return scoped_exp