from __future__ import annotations

from fish.config import MAX_EMBED_CHARS, embedding_model, openai_api_key
from openai import AuthenticationError, OpenAI

_client: OpenAI | None = None


def reset_client() -> None:
    global _client
    _client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=openai_api_key())
    return _client


def truncate_for_embed(text: str) -> str:
    return text[:MAX_EMBED_CHARS]


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
