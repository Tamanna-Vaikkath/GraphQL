"""
pipeline/cypher_executor.py — Cypher lane execution + self-healing repair.

Mirrors pipeline/executor.py (the SQL lane's executor) so both lanes have
symmetric behavior: run the query against the live database, and if it
errors, ask the LLM to repair it and retry, up to max_retries times.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from neo4j import GraphDatabase

from config import Config
from pipeline.cypher_generator import GRAPH_SCHEMA_BLOCK, _validate_cypher, CypherValidationError
from utils.llm_client import get_gpt_client, chat_complete

logger = logging.getLogger(__name__)

_REPAIR_SYSTEM = f"""You are a Cypher repair agent for a Neo4j database.
You will receive a failing Cypher query and its error. Fix the Cypher and
return ONLY the corrected Cypher statement. No explanation, no markdown
fences.

{GRAPH_SCHEMA_BLOCK}
"""


@dataclass
class CypherExecutionResult:
    success: bool
    df: Optional[pd.DataFrame] = None
    cypher_executed: str = ""
    error: str = ""
    attempts: int = 1
    repair_history: List[Dict[str, str]] = field(default_factory=list)


class CypherExecutor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)
        self.driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_username, cfg.neo4j_password)
        )
        print(f"\n[CypherExecutor INIT] Connected to Neo4j at {cfg.neo4j_uri}")

    def close(self) -> None:
        self.driver.close()

    def execute(
        self, cypher: str, question: str, max_retries: int = 3
    ) -> CypherExecutionResult:
        print(f"\n{'='*60}")
        print(f"[Cypher lane] CypherExecutor.execute()")
        print(f"  Question       : {question}")
        print(f"  Cypher to run  :\n{cypher}")
        print(f"  max_retries    : {max_retries}")
        print(f"{'='*60}")

        current_cypher = cypher
        repair_history: List[Dict[str, str]] = []

        for attempt in range(1, max_retries + 2):
            print(f"\n[Cypher lane] Attempt {attempt}/{max_retries+1}")
            try:
                df = self._run_cypher(current_cypher)
                print(f"[Cypher lane OUTPUT] Cypher succeeded on attempt {attempt}")
                print(f"  Result shape: {df.shape[0]} rows x {df.shape[1]} cols")
                print(f"  Columns: {list(df.columns)}")
                return CypherExecutionResult(
                    success=True,
                    df=df,
                    cypher_executed=current_cypher,
                    attempts=attempt,
                    repair_history=repair_history,
                )
            except Exception as e:
                error_msg = str(e)
                logger.warning("Cypher attempt %d failed: %s", attempt, error_msg)
                print(f"[Cypher lane ERROR] Attempt {attempt} failed: {error_msg}")
                repair_history.append({"cypher": current_cypher, "error": error_msg})

                if attempt > max_retries:
                    break

                print(f"[Cypher lane] Invoking repair agent...")
                try:
                    current_cypher = self._repair(question, current_cypher, error_msg)
                    print(f"[Cypher lane] Repaired Cypher:\n{current_cypher}")
                except Exception as repair_err:
                    logger.warning("Cypher repair failed: %s", repair_err)
                    break

        print(f"[Cypher lane FAILED] All attempts exhausted.")
        return CypherExecutionResult(
            success=False,
            cypher_executed=current_cypher,
            error=repair_history[-1]["error"] if repair_history else "Unknown error",
            attempts=max_retries + 1,
            repair_history=repair_history,
        )

    def _run_cypher(self, cypher: str) -> pd.DataFrame:
        print(f"  [_run_cypher] Executing against Neo4j...")
        with self.driver.session(database=self.cfg.neo4j_database) as session:
            result = session.run(cypher)
            records = [dict(r) for r in result]
        df = pd.DataFrame(records)
        print(f"  [_run_cypher] Query returned {len(df)} rows")
        return df

    def _repair(self, question: str, failing_cypher: str, error: str) -> str:
        print(f"\n[Cypher lane REPAIR INPUT]")
        print(f"  question       : {question}")
        print(f"  failing_cypher : {failing_cypher}")
        print(f"  error          : {error}")
        user_prompt = (
            f"Original question: {question}\n\n"
            f"Failing Cypher:\n{failing_cypher}\n\n"
            f"Error: {error}\n\n"
            "Return the corrected Cypher only."
        )
        repaired = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            _REPAIR_SYSTEM, user_prompt,
            temperature=0.0, max_tokens=600,
        )
        repaired = (
            repaired.strip()
            .removeprefix("```cypher").removeprefix("```")
            .removesuffix("```").strip()
        )
        _validate_cypher(repaired)
        print(f"[Cypher lane REPAIR OUTPUT] Repaired Cypher:\n{repaired}")
        logger.info("Cypher repair agent produced:\n%s", repaired)
        return repaired
