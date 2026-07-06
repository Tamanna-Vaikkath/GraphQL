"""
pipeline/summarizer.py — Optional AI Summary of query results.
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
        print("[ResultSummarizer INIT] Summarizer ready.")

    def summarize(self, question: str, df: pd.DataFrame) -> str:
        print(f"\n{'='*60}")
        print(f"[Summarizer INPUT]")
        print(f"  Question    : {question}")
        print(f"  DataFrame   : {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Columns     : {list(df.columns)}")
        print(f"  Preview:\n{df.head(5).to_string(index=False)}")
        print(f"{'='*60}")

        if df is None or df.empty:
            print("[Summarizer OUTPUT] DataFrame is empty — returning default message.")
            return "The query returned no results."

        preview = df.head(20).to_string(index=False)
        shape_info = f"{len(df)} rows × {len(df.columns)} columns"

        user_prompt = (
            f"User question: {question}\n\n"
            f"Result shape: {shape_info}\n\n"
            f"Result preview:\n{preview}"
        )
        summary = chat_complete(
            self.gpt, self.cfg.openai_deployment_name,
            SUMMARY_SYSTEM, user_prompt,
            temperature=0.3, max_tokens=400
        )
        print(f"[Summarizer OUTPUT] Summary:\n{summary}")
        return summary
