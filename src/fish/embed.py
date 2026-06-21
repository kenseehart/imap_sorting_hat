from __future__ import annotations

import tiktoken
from openai import AuthenticationError, OpenAI

from fish.config import EMBED_REQUEST_MAX_TOKENS, MAX_EMBED_TOKENS, embedding_model, openai_api_key

_client: OpenAI | None = None
_encoding: tiktoken.Encoding | None = None


def reset_client() -> None:
    global _client
    _client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=openai_api_key())
    return _client


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def truncate_for_embed(text: str, *, max_tokens: int | None = None) -> str:
    limit = max_tokens if max_tokens is not None else MAX_EMBED_TOKENS
    tokens = _get_encoding().encode(text)
    if len(tokens) <= limit:
        return text
    return _get_encoding().decode(tokens[:limit])


def token_count(text: str) -> int:
    return len(_get_encoding().encode(truncate_for_embed(text)))


def _embed_api_batch(texts: list[str]) -> list[list[float]]:
    client = get_client()
    try:
        response = client.embeddings.create(
            input=texts,
            model=embedding_model(),
        )
    except AuthenticationError as exc:
        reset_client()
        raise exc
    return [item.embedding for item in response.data]


def _token_batches(texts: list[str]) -> list[list[str]]:
    """Split texts into API batches that stay under EMBED_REQUEST_MAX_TOKENS."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        n = token_count(text)
        if current and current_tokens + n > EMBED_REQUEST_MAX_TOKENS:
            batches.append(current)
            current = [text]
            current_tokens = n
        else:
            current.append(text)
            current_tokens += n
    if current:
        batches.append(current)
    return batches


def embed_text(text: str) -> list[float]:
    return _embed_api_batch([truncate_for_embed(text)])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    truncated = [truncate_for_embed(t) for t in texts]
    vectors: list[list[float]] = []
    for batch in _token_batches(truncated):
        vectors.extend(_embed_api_batch(batch))
    return vectors
