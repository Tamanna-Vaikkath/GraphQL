"""
pipeline/query_complexity_analyzer.py — Stage 3.5: NLQ complexity analysis
+ query planning, run just before SQL generation (Stage 4).

WHY
----
Plain single-pass SQL generation works fine for simple filter/aggregate
questions ("claims with status Open in Texas"). It breaks down on nested or
comparative questions that require multiple logical passes over the data —
"top adjuster by incurred amount in each state", "claimants whose claim
count is above the average", "running total of payments per month" — where
the LLM benefits from being told up front which SQL techniques to reach for
(CTEs, subqueries, window functions) and in what order to reason about the
problem.

This module scans the question, classifies its complexity, and — for
moderate/complex questions — asks the LLM to decompose it into an ordered
list of logical reasoning steps. The result is a `QueryPlan` that
`SQLGenerator.generate()` folds into its prompt so the generated SQL mirrors
the plan (e.g. one CTE per step) instead of guessing at structure from the
raw question alone.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"


@dataclass
class QueryPlan:
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    needs_cte: bool = False
    needs_subquery: bool = False
    needs_window_function: bool = False
    reasoning_steps: List[str] = field(default_factory=list)
    notes: str = ""
    method: str = "heuristic"  # "heuristic" | "llm" | "default"

    def is_trivial(self) -> bool:
        return self.complexity == QueryComplexity.SIMPLE and not self.reasoning_steps

    def to_prompt_block(self) -> str:
        """Render this plan as an instructive block for the SQL generator's
        user prompt. Returns "" for trivial/simple queries so the prompt
        stays lean for the common case."""
        if self.is_trivial():
            return ""

        lines = [f"Query complexity: {self.complexity.value}"]

        techniques = []
        if self.needs_cte:
            techniques.append("Common Table Expression(s) (WITH ... AS)")
        if self.needs_subquery:
            techniques.append("a subquery (correlated or scalar, as needed)")
        if self.needs_window_function:
            techniques.append("window function(s) (e.g. RANK(), ROW_NUMBER(), SUM() OVER (...))")
        if techniques:
            lines.append("Recommended SQL techniques: " + "; ".join(techniques) + ".")

        if self.reasoning_steps:
            lines.append(
                "Break the query into these logical steps, and mirror that "
                "structure in the generated SQL (e.g. one CTE per step where "
                "it helps readability and correctness):"
            )
            for i, step in enumerate(self.reasoning_steps, 1):
                lines.append(f"  {i}. {step}")

        if self.notes:
            lines.append(f"Notes: {self.notes}")

        return "\n".join(lines)


# ── Heuristic signals ─────────────────────────────────────────────────────

_WINDOW_PATTERNS = [
    r"\brank(ed|ing)?\b", r"\btop \d+ (per|for|within|by)\b",
    r"\brunning (total|sum|balance)\b", r"\bcumulative\b",
    r"\bpercentile\b", r"\bquartile\b", r"\bmoving average\b",
    r"\bfor each\b.*\btop\b", r"\bfirst\b.*\bper\b", r"\blast\b.*\bper\b",
    r"\bconsecutive\b", r"\byear[- ]over[- ]year\b", r"\bmonth[- ]over[- ]month\b",
]

_SUBQUERY_PATTERNS = [
    r"\bmore than (the |their )?average\b", r"\babove average\b", r"\bbelow average\b",
    r"\bhigher than\b.*\baverage\b", r"\bexceeds?\b.*\baverage\b",
    r"\bwithout (any|a)\b", r"\bthat (do|does) not have\b",
    r"\bnot in\b", r"\bexcept\b", r"\bnot exists\b",
    r"\bwhose\b.*\bis (the )?(highest|lowest|max|min)\b",
]

_CTE_PATTERNS = [
    r"\bfirst\b.*\bthen\b", r"\band then\b", r"\bafter that\b",
    r"\bstep[- ]by[- ]step\b",
    r"\bfor each\b.*\bfind\b.*\bthen\b",
    r"\bcompare[sd]?\b.*\b(to|with|against)\b.*\b(average|overall|total|each other)\b",
]

_MULTI_AGGREGATION_RE = re.compile(
    r"\b(count|sum|avg|average|total|min|max)\b", re.IGNORECASE
)
_GROUP_RE = re.compile(
    r"\b(for each|per|by (state|status|type|adjuster|agent|month|year|claimant|policy))\b",
    re.IGNORECASE,
)

_WINDOW_RE = re.compile("|".join(_WINDOW_PATTERNS), re.IGNORECASE)
_SUBQUERY_RE = re.compile("|".join(_SUBQUERY_PATTERNS), re.IGNORECASE)
_CTE_RE = re.compile("|".join(_CTE_PATTERNS), re.IGNORECASE)


_DECOMPOSE_SYSTEM = """You are a query-planning assistant for a SQL generation \
pipeline over an insurance database (claimants, policies, claims, payments).

Given a natural-language analytics question, decide:
1. Whether answering it correctly needs a Common Table Expression (CTE), a \
subquery, and/or a window function.
2. Break the question into an ordered list of short logical reasoning steps \
(2-6 steps) describing how to compute the answer, e.g. "Step 1: compute total \
incurred amount per adjuster", "Step 2: rank adjusters by that total within \
each state", "Step 3: keep only the top-ranked adjuster per state".

Respond ONLY with a JSON object, no markdown fences, no preamble, in exactly \
this shape:
{"complexity": "SIMPLE|MODERATE|COMPLEX", "needs_cte": true|false, \
"needs_subquery": true|false, "needs_window_function": true|false, \
"steps": ["...", "..."], "notes": "..."}
"""


class QueryComplexityAnalyzer:
    """
    Stage 3.5 of the SQL lane. Runs just before SQL generation to detect
    whether a question is a simple single-pass query or a complex/nested one
    that needs CTEs, subqueries, or window functions — and, for the latter,
    to break it into an ordered query plan the SQL generator can follow.

    Cheap keyword heuristics handle the common cases (and let SIMPLE queries
    skip the extra LLM call entirely); an LLM decomposition step only runs
    when multiple complexity signals fire together.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)

    def analyze(self, question: str) -> QueryPlan:
        print(f"\n[QueryComplexityAnalyzer] Analyzing: {repr(question)}")

        window_hit = _WINDOW_RE.search(question)
        subquery_hit = _SUBQUERY_RE.search(question)
        cte_hit = _CTE_RE.search(question)
        agg_hits = _MULTI_AGGREGATION_RE.findall(question)
        group_hit = _GROUP_RE.search(question)

        needs_window = bool(window_hit) or (bool(group_hit) and "top" in question.lower())
        needs_subquery = bool(subquery_hit)
        needs_cte = bool(cte_hit) or (needs_window and needs_subquery)
        multi_aggregation = len(set(a.lower() for a in agg_hits)) >= 2

        signal_count = sum([needs_window, needs_subquery, needs_cte, multi_aggregation])

        if signal_count == 0:
            print("[QueryComplexityAnalyzer] No complexity signals — SIMPLE, skipping planning.")
            return QueryPlan(complexity=QueryComplexity.SIMPLE, method="heuristic")

        if signal_count == 1 and not (needs_window or needs_subquery):
            print("[QueryComplexityAnalyzer] Single moderate signal — MODERATE "
                  "(heuristic), skipping LLM decomposition.")
            return QueryPlan(
                complexity=QueryComplexity.MODERATE,
                needs_cte=needs_cte, needs_subquery=needs_subquery,
                needs_window_function=needs_window,
                method="heuristic",
            )

        # ── Strong or multiple signals: ask the LLM to decompose into steps ──
        try:
            raw = chat_complete(
                self.gpt, self.cfg.openai_deployment_name,
                _DECOMPOSE_SYSTEM, f"Question: {question}",
                temperature=0.0, max_tokens=500,
            )
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)

            complexity_raw = str(data.get("complexity", "MODERATE")).upper()
            try:
                complexity = QueryComplexity(complexity_raw)
            except ValueError:
                complexity = QueryComplexity.MODERATE

            plan = QueryPlan(
                complexity=complexity,
                needs_cte=bool(data.get("needs_cte", needs_cte)),
                needs_subquery=bool(data.get("needs_subquery", needs_subquery)),
                needs_window_function=bool(data.get("needs_window_function", needs_window)),
                reasoning_steps=[str(s) for s in data.get("steps", [])],
                notes=str(data.get("notes", "")),
                method="llm",
            )
            print(
                f"[QueryComplexityAnalyzer] LLM plan: complexity={plan.complexity.value}, "
                f"cte={plan.needs_cte}, subquery={plan.needs_subquery}, "
                f"window={plan.needs_window_function}, steps={len(plan.reasoning_steps)}"
            )
            return plan
        except Exception as e:
            logger.warning("QueryComplexityAnalyzer LLM decomposition failed: %s", e)
            print(f"[QueryComplexityAnalyzer] LLM decomposition failed ({e}) — "
                  "falling back to heuristic plan.")
            return QueryPlan(
                complexity=QueryComplexity.COMPLEX if signal_count >= 2 else QueryComplexity.MODERATE,
                needs_cte=needs_cte, needs_subquery=needs_subquery,
                needs_window_function=needs_window,
                method="default",
            )