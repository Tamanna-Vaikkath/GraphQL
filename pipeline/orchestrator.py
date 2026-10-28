"""
pipeline/orchestrator.py — End-to-end NLQ pipeline orchestrator.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
from pipeline.query_complexity_analyzer import QueryComplexityAnalyzer, QueryPlan
from pipeline.executor import SQLExecutor, ExecutionResult, RepairAttempt
from pipeline.summarizer import ResultSummarizer
from pipeline.query_router import QueryRouter, RouteDecision, SQL_LANE, CYPHER_LANE
from pipeline.cypher_generator import CypherGenerator, CypherValidationError, find_claim_clusters
from pipeline.cypher_executor import CypherExecutor, CypherExecutionResult
from kg.graph_ingest_gl import ensure_graph_ingested   # ontology-driven GL ingest (legacy: kg/graph_ingest.py)
from utils.llm_client import GPTClient

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.75
MAX_EXPAND_RETRIES  = 2

# Questions asking for clusters/groups/networks of claims tied together
# through a shared adjuster, agent, or payee are answered by the
# deterministic, Python-side connected-components traversal in
# pipeline.cypher_generator.find_claim_clusters() rather than by asking the
# LLM to free-hand a single Cypher statement for it. A single query that
# unions three different shared-entity join patterns into one clustering
# result is exactly the kind of thing LLM-generated Cypher tends to get
# subtly wrong (e.g. trying to pattern-match against an already-aggregated
# collect() result) — see the module docstring in cypher_generator.py.
import re as _re
_CLAIM_CLUSTER_RE = _re.compile(
    r"\bcluster[s]?\b.*\b(adjuster|agent|payee|payment)\b"
    r"|\b(adjuster|agent|payee|payment)\b.*\bcluster[s]?\b"
    r"|\bgroup[s]?\b.*\bclaims\b.*\b(adjuster|agent|payee)\b"
    r"|\bnetwork[s]?\b.*\bclaims\b.*\b(adjuster|agent|payee)\b",
    _re.IGNORECASE,
)


@dataclass
class QueryTrace:
    original_question: str = ""
    # ── Swim-lane routing ───────────────────────────────────────────────
    lane: str = SQL_LANE                # "SQL" or "CYPHER"
    route_reason: str = ""
    route_method: str = ""
    # ── SQL lane ─────────────────────────────────────────────────────────
    scoped_tables: List[str] = field(default_factory=list)
    scope_retrieval_method: str = ""
    table_scores: dict = field(default_factory=dict)
    expanded_query: str = ""
    max_retrieval_score: float = 0.0
    retrieved_columns: list = field(default_factory=list)
    join_conditions: list = field(default_factory=list)
    # ── Stage 3.5: NLQ complexity analysis / query planning ─────────────
    query_complexity: str = "SIMPLE"
    complexity_method: str = ""
    needs_cte: bool = False
    needs_subquery: bool = False
    needs_window_function: bool = False
    reasoning_steps: List[str] = field(default_factory=list)
    generated_sql: str = ""
    sql_repair_attempts: int = 0
    repair_history: List[dict] = field(default_factory=list)
    # Full repair-cycle detail (before/after SQL, diffs, prompts, validation
    # notes) — one RepairAttempt per retry — so the UI can show exactly how
    # the self-healing repair agent got from a failing query to a working one.
    sql_repair_detail: List[RepairAttempt] = field(default_factory=list)
    # Pre-rendered, human-readable end-to-end trace of the whole repair
    # sequence (same text that's logged/printed on failure), ready to drop
    # straight into the UI.
    sql_repair_full_trace: str = ""

    # ── Cypher / Knowledge Graph lane ───────────────────────────────────
    generated_cypher: str = ""
    cypher_repair_attempts: int = 0
    cypher_repair_history: List[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    # ── Stage-wise actual execution time (seconds), in pipeline order ───
    # keys are human-readable stage labels; insertion order == run order,
    # so the UI can render this straight through without re-sorting.
    stage_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryResult:
    success: bool
    df: Optional[pd.DataFrame]
    summary: str
    trace: QueryTrace
    error: str = ""
    schema_grounding_error: bool = False

    @property
    def lane(self) -> str:
        return self.trace.lane


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

        self.complexity_analyzer = QueryComplexityAnalyzer(self.cfg)
        print("[NLQPipeline INIT] QueryComplexityAnalyzer ready")

        self.executor   = SQLExecutor(self.cfg)
        self.summarizer = ResultSummarizer(self.cfg)
        print("[NLQPipeline INIT] SQLExecutor + ResultSummarizer ready")

        # ── Swim-lane 2: Cypher / Knowledge Graph ──────────────────────────
        self.router = QueryRouter(self.cfg)
        print("[NLQPipeline INIT] QueryRouter ready (SQL vs Cypher lane)")

        self.cypher_gen = CypherGenerator(self.cfg)
        self.cypher_executor = CypherExecutor(self.cfg)
        print("[NLQPipeline INIT] CypherGenerator + CypherExecutor ready")

        try:
            built = ensure_graph_ingested(self.cfg)
            print(f"[NLQPipeline INIT] Business Knowledge Graph {'built now' if built else 'already present'}")
        except Exception as e:
            logger.warning("Business graph ingestion check failed: %s", e)
            print(f"[NLQPipeline INIT] WARNING: business graph ingestion check failed: {e}")

        print("#"*70 + "\n")

    def query(self, question: str, generate_summary: bool = True) -> QueryResult:
        print("\n" + "★"*70)
        print(f"[PIPELINE START] question={repr(question)}")
        print(f"                 generate_summary={generate_summary}")
        print("★"*70)

        t0 = time.time()
        trace = QueryTrace(original_question=question)

        # ── Stage -1: Swim-lane routing (SQL vs Cypher/Knowledge Graph) ────
        print(f"\n{'─'*60}")
        print(f"[Stage -1] Swim-lane routing")
        print(f"{'─'*60}")
        stage_t0 = time.time()
        decision: RouteDecision = self.router.route(question)
        trace.lane          = decision.lane
        trace.route_reason  = decision.reason
        trace.route_method  = decision.method
        trace.stage_timings["Stage -1 — Swim-lane Routing"] = round(time.time() - stage_t0, 3)
        print(f"[Stage -1 OUTPUT] lane={decision.lane} method={decision.method}")
        print(f"  reason: {decision.reason}")

        if decision.lane == CYPHER_LANE:
            try:
                return self._run_cypher_lane(question, trace, t0, generate_summary)
            except Exception as e:
                logger.exception("[Cypher lane] Unhandled error: %s", e)
                trace.elapsed_seconds = round(time.time() - t0, 2)
                return QueryResult(success=False, df=None, summary="", trace=trace, error=str(e))

        try:
            # ── Stage 0 ──────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 0] Scoped schema retrieval")
            print(f"{'─'*60}")
            logger.info("[Stage 0] Scoped schema retrieval...")
            stage_t0 = time.time()
            scoped: ScopedSchema = self.schema_retriever.retrieve_scoped_schema(question)
            trace.scoped_tables          = scoped.relevant_tables
            trace.scope_retrieval_method = scoped.retrieval_method
            trace.table_scores           = scoped.table_scores
            trace.stage_timings["Stage 0 — Scoped Schema Retrieval"] = round(time.time() - stage_t0, 3)

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
            stage_t0 = time.time()

            scoped_expander = _make_scoped_expander(self.expander, scoped, self.cfg, self.registry)

            try:
                expanded = scoped_expander.expand_or_raise(question)
            except SchemaGroundingError as sge:
                logger.warning("[Stage 1] Schema grounding failed: %s", sge)
                print(f"[Stage 1 ERROR] SchemaGroundingError: {sge}")
                trace.stage_timings["Stage 1 — HyDE Expansion"] = round(time.time() - stage_t0, 3)
                trace.elapsed_seconds = round(time.time() - t0, 2)
                return QueryResult(
                    success=False, df=None, summary="", trace=trace,
                    error=str(sge), schema_grounding_error=True,
                )
            trace.expanded_query = expanded
            trace.stage_timings["Stage 1 — HyDE Expansion"] = round(time.time() - stage_t0, 3)
            print(f"[Stage 1 OUTPUT] Expanded query:\n  {expanded}")

            # ── Stage 2+3 ─────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 2+3] KG scoped retrieval")
            print(f"{'─'*60}")
            print(f"[Stage 2+3 INPUT] expanded_query={repr(expanded)}")
            logger.info("[Stage 2+3] KG scoped retrieval...")
            stage_t0 = time.time()
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
            trace.stage_timings["Stage 2+3 — KG Column Retrieval"] = round(time.time() - stage_t0, 3)

            print(f"[Stage 2+3 OUTPUT]")
            print(f"  Max score      : {retrieval.max_score:.4f}")
            print(f"  Tables involved: {retrieval.tables_involved}")
            print(f"  Columns (top 5):")
            for c in retrieval.columns[:5]:
                print(f"    {c.table}.{c.name} score={c.score:.3f}")
            print(f"  Join conditions: {retrieval.join_conditions}")

            # ── Stage 3.5 ────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 3.5] NLQ complexity analysis / query planning")
            print(f"{'─'*60}")
            logger.info("[Stage 3.5] Analyzing query complexity...")
            stage_t0 = time.time()
            query_plan: QueryPlan = self.complexity_analyzer.analyze(question)
            trace.query_complexity  = query_plan.complexity.value
            trace.complexity_method = query_plan.method
            trace.needs_cte             = query_plan.needs_cte
            trace.needs_subquery        = query_plan.needs_subquery
            trace.needs_window_function = query_plan.needs_window_function
            trace.reasoning_steps       = query_plan.reasoning_steps
            trace.stage_timings["Stage 3.5 — Complexity Analysis"] = round(time.time() - stage_t0, 3)

            print(f"[Stage 3.5 OUTPUT]")
            print(f"  Complexity : {query_plan.complexity.value} (method={query_plan.method})")
            print(f"  CTE / Subquery / Window : "
                  f"{query_plan.needs_cte} / {query_plan.needs_subquery} / "
                  f"{query_plan.needs_window_function}")
            if query_plan.reasoning_steps:
                print(f"  Reasoning steps:")
                for i, step in enumerate(query_plan.reasoning_steps, 1):
                    print(f"    {i}. {step}")

            # ── Stage 4 ──────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 4] SQL Generation")
            print(f"{'─'*60}")
            print(f"[Stage 4 INPUT] question={repr(question)}")
            logger.info("[Stage 4] Generating SQL with scoped domain rules...")
            stage_t0 = time.time()
            try:
                sql = self.sql_gen.generate(question, retrieval, query_plan=query_plan)
            except SQLColumnNotFoundError as col_err:
                logger.warning("[Stage 4] Column validation failed: %s", col_err)
                print(f"[Stage 4 ERROR] SQLColumnNotFoundError: {col_err}")
                trace.stage_timings["Stage 4 — SQL Generation"] = round(time.time() - stage_t0, 3)
                trace.elapsed_seconds = round(time.time() - t0, 2)
                return QueryResult(
                    success=False, df=None, summary="", trace=trace,
                    error=str(col_err), schema_grounding_error=True,
                )
            trace.generated_sql = sql
            trace.stage_timings["Stage 4 — SQL Generation"] = round(time.time() - stage_t0, 3)
            print(f"[Stage 4 OUTPUT] Generated SQL:\n{sql}")

            # ── Stage 5 ──────────────────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"[Stage 5] SQL Execution")
            print(f"{'─'*60}")
            logger.info("[Stage 5] Executing SQL...")
            exec_result: ExecutionResult = self.executor.execute(sql, question)
            trace.sql_repair_attempts   = exec_result.attempts - 1
            trace.repair_history        = exec_result.repair_history
            trace.sql_repair_detail     = exec_result.repair_attempts
            trace.sql_repair_full_trace = exec_result.full_trace
            trace.generated_sql         = exec_result.sql_executed
            # Actual, measured wall-clock time for Stage 5 (all execution
            # attempts + all repair cycles), taken from the executor itself
            # rather than re-timed here, so it reflects real DB + repair work.
            trace.stage_timings["Stage 5 — SQL Execution (incl. repairs)"] = exec_result.elapsed_seconds
            trace.elapsed_seconds       = round(time.time() - t0, 2)

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
                stage_t0 = time.time()
                try:
                    summary = self.summarizer.summarize(question, exec_result.df)
                except Exception as e:
                    logger.warning("Summarizer failed: %s", e)
                    print(f"[Summary ERROR] {e}")
                trace.stage_timings["Summary — AI Narrative"] = round(time.time() - stage_t0, 3)
                trace.elapsed_seconds = round(time.time() - t0, 2)

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

    def _run_cypher_lane(
        self, question: str, trace: QueryTrace, t0: float, generate_summary: bool
    ) -> QueryResult:
        """Swim lane 2: Knowledge Graph + Cypher retrieval, with actual
        Neo4j execution against the business data graph."""

        # ── Shared-entity claim clustering: use the deterministic traversal ──
        # ("clusters of claims linked through shared adjusters, agents, or
        # payment payees") instead of asking the LLM to free-hand a single
        # Cypher statement, which is unreliable for a multi-shared-entity
        # union like this (see find_claim_clusters() docstring).
        if _CLAIM_CLUSTER_RE.search(question):
            print(f"\n{'─'*60}")
            print(f"[Cypher lane] Shared-entity cluster question detected — "
                  f"using deterministic find_claim_clusters() instead of LLM Cypher gen")
            print(f"{'─'*60}")
            stage_t0 = time.time()
            try:
                clusters = find_claim_clusters(self.cfg)
                df = pd.DataFrame(clusters)
                trace.generated_cypher = (
                    "-- Deterministic connected-components traversal "
                    "(pipeline.cypher_generator.find_claim_clusters) --\n"
                    "MATCH (c1:Claim)-[:HANDLED_BY]->(a:Adjuster)<-[:HANDLED_BY]-(c2:Claim) ...\n"
                    "MATCH (c1:Claim)-[:COVERED_BY]->(p1:Policy), (c2:Claim)-[:COVERED_BY]->(p2:Policy) "
                    "WHERE p1.agent_id = p2.agent_id ...\n"
                    "MATCH (c1:Claim)-[:HAS_PAYMENT]->(p1:Payment), (c2:Claim)-[:HAS_PAYMENT]->(p2:Payment) "
                    "WHERE p1.payee_name = p2.payee_name ...\n"
                    "-- unioned via union-find into connected components --"
                )
                trace.stage_timings["Cypher Generation"] = 0.0
                trace.stage_timings["Cypher Execution (incl. repairs)"] = round(time.time() - stage_t0, 3)
                trace.elapsed_seconds = round(time.time() - t0, 2)

                summary = ""
                if generate_summary and not df.empty:
                    stage_t0 = time.time()
                    try:
                        summary = self.summarizer.summarize(question, df)
                    except Exception as e:
                        logger.warning("[Cypher lane] Summarizer failed: %s", e)
                    trace.stage_timings["Summary — AI Narrative"] = round(time.time() - stage_t0, 3)
                    trace.elapsed_seconds = round(time.time() - t0, 2)

                print(f"[Cypher lane DONE] success=True (deterministic clustering) | "
                      f"elapsed={trace.elapsed_seconds}s | clusters={len(df)}")
                return QueryResult(success=True, df=df, summary=summary, trace=trace)
            except Exception as e:
                logger.exception("[Cypher lane] find_claim_clusters failed: %s", e)
                print(f"[Cypher lane ERROR] find_claim_clusters failed: {e} — "
                      f"falling back to LLM-generated Cypher")
                # fall through to the normal LLM-generation path below

        print(f"\n{'─'*60}")
        print(f"[Cypher lane] Generation")
        print(f"{'─'*60}")
        stage_t0 = time.time()
        try:
            cypher = self.cypher_gen.generate(question)
        except CypherValidationError as cve:
            logger.warning("[Cypher lane] Validation failed: %s", cve)
            print(f"[Cypher lane ERROR] CypherValidationError: {cve}")
            trace.stage_timings["Cypher Generation"] = round(time.time() - stage_t0, 3)
            trace.elapsed_seconds = round(time.time() - t0, 2)
            return QueryResult(
                success=False, df=None, summary="", trace=trace,
                error=str(cve), schema_grounding_error=True,
            )
        except Exception as e:
            logger.exception("[Cypher lane] Generation error: %s", e)
            trace.stage_timings["Cypher Generation"] = round(time.time() - stage_t0, 3)
            trace.elapsed_seconds = round(time.time() - t0, 2)
            return QueryResult(success=False, df=None, summary="", trace=trace, error=str(e))

        trace.generated_cypher = cypher
        trace.stage_timings["Cypher Generation"] = round(time.time() - stage_t0, 3)

        print(f"\n{'─'*60}")
        print(f"[Cypher lane] Execution")
        print(f"{'─'*60}")
        stage_t0 = time.time()
        exec_result: CypherExecutionResult = self.cypher_executor.execute(cypher, question)
        trace.cypher_repair_attempts = exec_result.attempts - 1
        trace.cypher_repair_history  = exec_result.repair_history
        trace.generated_cypher       = exec_result.cypher_executed
        trace.stage_timings["Cypher Execution (incl. repairs)"] = round(time.time() - stage_t0, 3)
        trace.elapsed_seconds        = round(time.time() - t0, 2)

        if not exec_result.success:
            print(f"[Cypher lane FAILED] error={exec_result.error}")
            return QueryResult(
                success=False, df=None, summary="",
                trace=trace, error=exec_result.error,
            )

        summary = ""
        if generate_summary and exec_result.df is not None:
            stage_t0 = time.time()
            try:
                summary = self.summarizer.summarize(question, exec_result.df)
            except Exception as e:
                logger.warning("[Cypher lane] Summarizer failed: %s", e)
            trace.stage_timings["Summary — AI Narrative"] = round(time.time() - stage_t0, 3)
            trace.elapsed_seconds = round(time.time() - t0, 2)

        print(f"\n[Cypher lane DONE] success=True | elapsed={trace.elapsed_seconds}s")
        print(f"  Rows returned: {len(exec_result.df) if exec_result.df is not None else 0}")

        return QueryResult(
            success=True,
            df=exec_result.df,
            summary=summary,
            trace=trace,
        )

    def close(self) -> None:
        self.retriever.close()
        self.schema_retriever.close()
        try:
            self.cypher_executor.close()
        except Exception:
            pass


def _make_scoped_expander(
    base_expander: LLMHyDEExpander,
    scoped: ScopedSchema,
    cfg: Config,
    registry: SchemaRegistry,
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

    # Merge in the registry's alias index (both auto-generated naming-convention
    # variants AND the hand-curated abbreviations / business synonyms loaded from
    # aliases.json, e.g. "clm" -> CLAIMS, "prem" -> POLICY.PREMIUM_AMT). Without
    # this, the mini concept map built above only recognizes a table/column's own
    # name — so once the pipeline narrows to a scoped expander (this function),
    # abbreviations that work fine at Stage 0 (schema retrieval) stop being
    # recognized at Stage 1 (HyDE), because this local map silently shadows the
    # richer registry-derived one the base expander was built with.
    scoped_table_set = set(scoped.relevant_tables)
    for alias, target in registry.aliases.items():
        if "." in target:
            tname, cname = target.split(".", 1)
            if tname in scoped_table_set and cname in scoped.scoped_manifest.get(tname, {}):
                mini_concept_map.setdefault(alias, [cname])
        else:
            if target in scoped_table_set:
                cols = scoped.scoped_manifest.get(target, {})
                pk_cols = [c for c in cols if c.endswith("_ID") and not c[:-3].endswith("_")]
                fallback = list(cols.keys())[:1] if cols else []
                mini_concept_map.setdefault(alias, pk_cols or fallback)

    scoped_exp._concept_map = mini_concept_map
    scoped_exp._validator   = SchemaValidator(scoped.scoped_all_columns)
    print(f"[_make_scoped_expander] Scoped expander ready (concept_map entries={len(mini_concept_map)})")
    return scoped_exp