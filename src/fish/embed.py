from __future__ import annotations

import tiktoken
from openai import AuthenticationError, OpenAI

from fish.config import MAX_EMBED_TOKENS, embedding_model, openai_api_key

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


def embed_text(text: str) -> list[float]:
    client = get_client()
    response = client.embeddings.create(
        input=truncate_for_embed(text),
        model=embedding_model(),
    )
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_client()
    try:
        response = client.embeddings.create(
            input=[truncate_for_embed(t) for t in texts],
            model=embedding_model(),
        )
    except AuthenticationError as exc:
        reset_client()
        raise exc
    return [item.embedding for item in response.data]
