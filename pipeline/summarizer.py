"""
pipeline/summarizer.py — Optional AI Summary of query results.
"""
from __future__ import annotations
import json
import pandas as pd

from config import Config
from utils.llm_client import get_gpt_client, chat_complete

SUMMARY_SYSTEM = """You are a P&C insurance business analyst.
Summarize the query results in 3-5 plain-English sentences for a non-technical business user.
Highlight key numbers, patterns, and actionable insights.
Refer to business concepts, not column names (e.g. 'open claims' not 'CLM_STAT_CD=O').
Be concise and precise.
Do NOT enumerate every row, ID, or list item individually — describe counts, ranges,
and notable patterns instead, even if the preview below lists many items. If a cell
looks truncated (ends in "..."), summarize based on what's shown without trying to
reconstruct or apologize for the missing part."""


def _stringify_cell(value, max_chars: int = 150) -> str:
    """
    Render a single cell as short, LLM-friendly text. List/tuple/dict cells
    (e.g. the claim_ids / connections columns from clustering results, which
    can hold hundreds of items for a large cluster) are truncated to a
    handful of items rather than dumped in full — a single bulky cell can
    otherwise balloon the summarizer prompt and push the model to enumerate
    everything instead of summarizing.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        shown = list(value[:5])
        text = ", ".join(str(v) for v in shown)
        if len(value) > 5:
            text += f", … (+{len(value) - 5} more)"
        return text if len(text) <= max_chars else text[:max_chars] + "…"
    if isinstance(value, dict):
        text = json.dumps(value, default=str)
        return text if len(text) <= max_chars else text[:max_chars] + "…"
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _build_preview(df: pd.DataFrame, max_rows: int = 20, max_total_chars: int = 4000) -> str:
    """
    Build a compact preview for the summarizer prompt: truncates any large
    list/dict cell values per-cell, then caps the overall preview length so
    a handful of bulky rows can't balloon the prompt regardless of shape.
    """
    head = df.head(max_rows).copy()
    for col in head.columns:
        head[col] = head[col].apply(_stringify_cell)
    preview = head.to_string(index=False)
    if len(preview) > max_total_chars:
        preview = preview[:max_total_chars] + "\n… (preview truncated for length)"
    return preview


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

        preview = _build_preview(df)
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