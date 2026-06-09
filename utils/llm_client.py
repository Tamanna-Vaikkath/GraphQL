"""
utils/llm_client.py — Thin wrappers around Azure OpenAI for GPT and embeddings.
All LLM calls in the pipeline flow through these two functions.
"""
from __future__ import annotations
import logging
from typing import List

from openai import AzureOpenAI

from config import Config

logger = logging.getLogger(__name__)


def get_gpt_client(cfg: Config) -> AzureOpenAI:
    """Return an AzureOpenAI client for GPT (HyDE, SQL gen, self-healing)."""
    return AzureOpenAI(
        azure_endpoint=cfg.openai_endpoint,
        api_key=cfg.openai_api_key,
        api_version=cfg.openai_api_version,
    )


def get_embedding_client(cfg: Config) -> AzureOpenAI:
    """Return an AzureOpenAI client for the embedding model endpoint."""
    return AzureOpenAI(
        azure_endpoint=cfg.embedding_endpoint,
        api_key=cfg.embedding_api_key,
        api_version=cfg.openai_api_version,
    )


def chat_complete(client: AzureOpenAI, deployment: str, system: str, user: str,
                  temperature: float = 0.0, max_tokens: int = 1500) -> str:
    """Single-turn chat completion. Returns the assistant message text."""
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("GPT call failed: %s", e)
        raise


def embed_text(client: AzureOpenAI, deployment: str, text: str) -> List[float]:
    """Embed a single string. Returns a 1536-dim float list."""
    try:
        response = client.embeddings.create(
            model=deployment,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Embedding call failed: %s", e)
        raise


def embed_batch(client: AzureOpenAI, deployment: str, texts: List[str]) -> List[List[float]]:
    """Embed a list of strings in one API call (max 2048 items)."""
    try:
        response = client.embeddings.create(
            model=deployment,
            input=texts,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    except Exception as e:
        logger.error("Batch embedding call failed: %s", e)
        raise


class GPTClient:
    """
    Thin adapter that exposes a .complete() method so pipeline components
    (HyDE expander, SQL generator, summarizer, etc.) stay decoupled from
    Azure OpenAI internals.

    LLMHyDEExpander (and similar classes) call:
        self._client.complete(prompt)  ->  str

    This adapter satisfies that contract by delegating to chat_complete(),
    passing the combined prompt string as the user message and leaving the
    system message empty (the callers embed their system instruction directly
    inside the prompt string they build).
    """

    def __init__(self, cfg: Config) -> None:
        self._client = get_gpt_client(cfg)
        self._deployment = cfg.openai_deployment_name

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> str:
        """
        Send *prompt* as a user message and return the assistant reply.

        Parameters
        ----------
        prompt:
            The full prompt string (may already contain a system preamble
            concatenated by the caller).
        temperature:
            Sampling temperature (default 0.0 for deterministic output).
        max_tokens:
            Maximum tokens in the completion.
        """
        return chat_complete(
            client=self._client,
            deployment=self._deployment,
            system="",        # callers embed system instructions inside prompt
            user=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )