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
    print("\n" + "="*60)
    print("[LLM INPUT] chat_complete()")
    print(f"  Deployment : {deployment}")
    print(f"  Temperature: {temperature}  |  max_tokens: {max_tokens}")
    print(f"  System prompt ({len(system)} chars):\n{system[:500]}{'...' if len(system) > 500 else ''}")
    print(f"  User prompt ({len(user)} chars):\n{user[:500]}{'...' if len(user) > 500 else ''}")
    print("="*60)
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
        result = response.choices[0].message.content.strip()
        print(f"[LLM OUTPUT] chat_complete() → ({len(result)} chars):\n{result[:500]}{'...' if len(result) > 500 else ''}")
        print("="*60 + "\n")
        return result
    except Exception as e:
        logger.error("GPT call failed: %s", e)
        print(f"[LLM ERROR] chat_complete() raised: {e}")
        print("="*60 + "\n")
        raise


def embed_text(client: AzureOpenAI, deployment: str, text: str) -> List[float]:
    """Embed a single string. Returns a 1536-dim float list."""
    print(f"\n[EMBED INPUT]  embed_text() | deployment={deployment}")
    print(f"  Text ({len(text)} chars): {text[:200]}{'...' if len(text) > 200 else ''}")
    try:
        response = client.embeddings.create(
            model=deployment,
            input=text,
        )
        vec = response.data[0].embedding
        print(f"[EMBED OUTPUT] embed_text() → vector dim={len(vec)}, first5={[round(v,4) for v in vec[:5]]}")
        return vec
    except Exception as e:
        logger.error("Embedding call failed: %s", e)
        print(f"[EMBED ERROR]  embed_text() raised: {e}")
        raise


def embed_batch(client: AzureOpenAI, deployment: str, texts: List[str]) -> List[List[float]]:
    """Embed a list of strings in one API call (max 2048 items)."""
    print(f"\n[EMBED INPUT]  embed_batch() | deployment={deployment} | num_texts={len(texts)}")
    for i, t in enumerate(texts[:3]):
        print(f"  [{i}] {t[:120]}{'...' if len(t) > 120 else ''}")
    if len(texts) > 3:
        print(f"  ... ({len(texts)-3} more)")
    try:
        response = client.embeddings.create(
            model=deployment,
            input=texts,
        )
        result = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        print(f"[EMBED OUTPUT] embed_batch() → {len(result)} vectors, each dim={len(result[0]) if result else 0}")
        return result
    except Exception as e:
        logger.error("Batch embedding call failed: %s", e)
        print(f"[EMBED ERROR]  embed_batch() raised: {e}")
        raise


class GPTClient:
    """
    Thin adapter that exposes a .complete() method so pipeline components
    (HyDE expander, SQL generator, summarizer, etc.) stay decoupled from
    Azure OpenAI internals.
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
        print(f"\n[GPTClient INPUT]  complete() | deployment={self._deployment}")
        print(f"  Prompt ({len(prompt)} chars):\n{prompt[:400]}{'...' if len(prompt) > 400 else ''}")
        result = chat_complete(
            client=self._client,
            deployment=self._deployment,
            system="",
            user=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(f"[GPTClient OUTPUT] complete() → ({len(result)} chars): {result[:300]}{'...' if len(result) > 300 else ''}")
        return result
