"""PRISM / retrieval model configs (YAML) and model_id helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fish.config import CONFIG_DIR, EMBED_DIM

# Shipped defaults; user overrides in ~/.config/fish/prism_models.yaml
_PACKAGE_DEFAULTS = Path(__file__).resolve().parents[3] / "config" / "prism_models.yaml"
USER_MODELS_YAML = CONFIG_DIR / "prism_models.yaml"

LEGACY_MODEL_ID = "legacy"
LEGACY_VEC_TABLE = "corpus_vec"  # historical sqlite-vec name
LEGACY_COLLECTION = "fish_legacy"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_model_id(config_name: str, *, timestamp: str | None = None) -> str:
    name = config_name.strip()
    if not name or name == LEGACY_MODEL_ID:
        raise ValueError(f"Invalid PRISM config name: {config_name!r}")
    if "." in name:
        raise ValueError(f"config_name must not contain '.': {config_name!r}")
    return f"{name}.{timestamp or _utc_stamp()}"


def parse_model_id(model_id: str) -> tuple[str, str | None]:
    """Return (config_name, timestamp_or_None). legacy → ('legacy', None)."""
    mid = model_id.strip()
    if mid == LEGACY_MODEL_ID:
        return LEGACY_MODEL_ID, None
    if "." not in mid:
        raise ValueError(
            f"model_id must be 'legacy' or '{{config}}.{{timestamp}}', got {model_id!r}"
        )
    config_name, ts = mid.split(".", 1)
    return config_name, ts


def collection_for_model_id(model_id: str) -> str:
    """Qdrant collection name (also stored in retrieval_models.vec_table)."""
    if model_id == LEGACY_MODEL_ID:
        return LEGACY_COLLECTION
    safe = model_id.replace(".", "_").replace("-", "_")
    return f"fish_{safe}"


def vec_table_for_model_id(model_id: str) -> str:
    """Backward-compatible alias: Qdrant collection name."""
    return collection_for_model_id(model_id)


def load_prism_model_configs() -> dict[str, dict[str, Any]]:
    """Merge package defaults with optional user YAML."""
    configs: dict[str, dict[str, Any]] = {}
    for path in (_PACKAGE_DEFAULTS, USER_MODELS_YAML):
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"PRISM models YAML must be a mapping: {path}")
        for name, cfg in data.items():
            if name == LEGACY_MODEL_ID:
                continue
            if not isinstance(cfg, dict):
                raise ValueError(f"Config {name!r} in {path} must be a mapping")
            configs[str(name)] = {**configs.get(str(name), {}), **cfg}
    return configs


def get_prism_config(config_name: str) -> dict[str, Any]:
    configs = load_prism_model_configs()
    if config_name not in configs:
        known = ", ".join(sorted(configs)) or "(none)"
        raise KeyError(
            f"Unknown PRISM config {config_name!r}. Known: {known}. "
            f"Add it to {_PACKAGE_DEFAULTS} or {USER_MODELS_YAML}"
        )
    cfg = dict(configs[config_name])
    cfg.setdefault("embed_dim", EMBED_DIM)
    cfg.setdefault("epochs", 5)
    cfg.setdefault("lr", 2.0e-5)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("chunk_repr", "combined")
    cfg.setdefault("adapter_sharing", "dual")
    cfg.setdefault("scoring", "cosine")
    cfg.setdefault("head_hidden", 512)
    # 0 = train all epochs; >0 = stop when holdout Spearman stalls this many epochs
    cfg.setdefault("early_stop_patience", 0)
    cfg.setdefault("early_stop_min_delta", 0.0)
    chunk_repr = str(cfg["chunk_repr"])
    if chunk_repr not in ("combined", "header_body"):
        raise ValueError(
            f"Config {config_name!r}: chunk_repr must be 'combined' or "
            f"'header_body', got {chunk_repr!r}"
        )
    cfg["chunk_repr"] = chunk_repr
    sharing = str(cfg["adapter_sharing"]).strip().lower()
    if sharing not in ("dual", "siamese"):
        raise ValueError(
            f"Config {config_name!r}: adapter_sharing must be 'dual' or "
            f"'siamese', got {cfg['adapter_sharing']!r}"
        )
    if sharing == "siamese" and chunk_repr != "combined":
        raise ValueError(
            f"Config {config_name!r}: siamese adapters require "
            f"chunk_repr=combined (got {chunk_repr!r}); header_body needs "
            f"asymmetric A_c input dim"
        )
    cfg["adapter_sharing"] = sharing
    scoring = str(cfg["scoring"]).strip().lower()
    if scoring not in ("cosine", "mlp_head"):
        raise ValueError(
            f"Config {config_name!r}: scoring must be 'cosine' or 'mlp_head', "
            f"got {cfg['scoring']!r}"
        )
    if scoring == "mlp_head" and sharing == "siamese":
        raise ValueError(
            f"Config {config_name!r}: scoring=mlp_head requires dual adapters "
            f"(personal_fields-style Aq||Ac)"
        )
    if scoring == "mlp_head" and chunk_repr != "header_body":
        raise ValueError(
            f"Config {config_name!r}: scoring=mlp_head requires "
            f"chunk_repr=header_body (Aq(E(q))||Ac(E(h)|E(b)))"
        )
    cfg["scoring"] = scoring
    cfg["head_hidden"] = max(1, int(cfg["head_hidden"]))
    cfg["early_stop_patience"] = max(0, int(cfg["early_stop_patience"]))
    cfg["early_stop_min_delta"] = float(cfg["early_stop_min_delta"])
    return cfg
