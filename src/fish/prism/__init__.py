"""PRISM dual-adapter retrieval for fish personal corpus."""

from fish.prism.inference import adapt_chunk_embedding, adapt_query_embedding, get_prism_model
from fish.prism.model import PrismModel, load_prz, new_identity_model, save_prz

__all__ = [
    "PrismModel",
    "adapt_chunk_embedding",
    "adapt_query_embedding",
    "get_prism_model",
    "load_prz",
    "new_identity_model",
    "save_prz",
]
