"""
hyde_expander.py — HyDE (Hypothetical Document Embedding) query expansion.

Produces a short, intent-focused hypothetical passage from the user's natural-
language question.  The passage is embedded and used for semantic retrieval
against the schema knowledge graph.

Key design principle
--------------------
Trimming happens HERE, before the passage is embedded, so retrieval sees a
clean 1–2 sentence document rather than a verbose paragraph.  Moving the trim
step upstream (away from the Streamlit UI layer) means:

  • The KG retrieval call receives a tighter embedding — improving recall.
  • The trace panel displays the *actual* text that drove retrieval (not a
    post-hoc trimmed copy), so the trace is truthful.
  • `app.py` has no trimming responsibility and needs no `_trim_hyde` helper.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Trim helper (was previously in app.py)
# ---------------------------------------------------------------------------

def trim_hyde(expanded: str, max_sentences: int = 2) -> str:
    """
    Reduce a verbose HyDE paragraph to at most *max_sentences* sentences.

    Rationale
    ---------
    Embedding models compress long texts by averaging token representations.
    A two-sentence passage that captures the query intent precisely produces a
    sharper embedding than a five-sentence paragraph with filler.  Empirically,
    the first 1–2 sentences of a hypothetical answer carry the most signal.

    Parameters
    ----------
    expanded:
        The raw hypothetical passage returned by the LLM.
    max_sentences:
        Maximum number of sentences to retain (default: 2).

    Returns
    -------
    The trimmed passage, or the original string if splitting fails.
    """
    if not expanded:
        return expanded
    # Split on sentence boundaries (.  !  ?)  followed by whitespace.
    sentences = re.split(r'(?<=[.!?])\s+', expanded.strip())
    trimmed = " ".join(sentences[:max_sentences])
    return trimmed or expanded


# ---------------------------------------------------------------------------
# Expander interface and default implementation
# ---------------------------------------------------------------------------

@runtime_checkable
class HyDEExpander(Protocol):
    """Minimal protocol for any HyDE expander implementation."""

    def expand(self, question: str) -> str:
        """Return the trimmed hypothetical passage for *question*."""
        ...


class LLMHyDEExpander:
    """
    Default HyDE expander: calls an LLM to generate a hypothetical document,
    then trims the result to ``max_sentences`` before returning.

    Parameters
    ----------
    llm_client:
        Any object with a ``complete(prompt: str) -> str`` method
        (e.g. an AzureOpenAI wrapper).
    max_sentences:
        Number of sentences to retain after trimming (default: 2).
    system_prompt:
        Optional override for the LLM system instruction.
    """

    _DEFAULT_SYSTEM = (
        "You are a P&C insurance data analyst. Given a natural-language "
        "question, write 1–2 sentences that directly mirror the question's "
        "intent using precise insurance domain terminology. "
        "Name the exact tables, status codes, and column values relevant to "
        "the question (e.g. CLM_STAT_CD='P' for pending, PMT_STAT_CD for "
        "payment status). Do NOT generate SQL. Do NOT describe generic "
        "database structure — stay tightly focused on what the question is "
        "actually asking for."
    )

    def __init__(
        self,
        llm_client,
        max_sentences: int = 2,
        system_prompt: str | None = None,
    ) -> None:
        self._client = llm_client
        self._max_sentences = max_sentences
        self._system = system_prompt or self._DEFAULT_SYSTEM

    def expand(self, question: str) -> str:
        """Generate and trim the hypothetical passage for *question*."""
        prompt = f"{self._system}\n\nQuestion: {question}"
        raw: str = self._client.complete(prompt)
        return trim_hyde(raw, max_sentences=self._max_sentences)