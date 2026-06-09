"""
pipeline/summarizer.py — Optional AI Summary of query results.

After execution, this module produces a plain-English narrative of the
result set — useful for non-technical stakeholders.
"""
from __future__ import annotations
import pandas as pd

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

SUMMARY_SYSTEM = """You are a P&C insurance business analyst.
Summarize the query results in 3-5 plain-English sentences for a non-technical business user.
Highlight key numbers, patterns, and actionable insights.
Refer to business concepts, not column names (e.g. 'open claims' not 'CLM_STAT_CD=O').
Be concise and precise."""


class ResultSummarizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gpt = get_gpt_client(cfg)

    def summarize(self, question: str, df: pd.DataFrame) -> str:
        """Generate a plain-English summary of query results.

        Args:
            question: The original business question.
            df: The result DataFrame.
        Returns:
            Plain-English narrative string.
        """
        if df is None or df.empty:
            return "The query returned no results."

        # Format a concise table preview (max 20 rows to stay within token budget)
        preview = df.head(20).to_string(index=False)
        shape_info = f"{len(df)} rows × {len(df.columns)} columns"

        user_prompt = (
            f"User question: {question}\n\n"
            f"Result shape: {shape_info}\n\n"
            f"Result preview:\n{preview}"
        )
        return chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            SUMMARY_SYSTEM, user_prompt,
            temperature=0.3, max_tokens=400
        )
