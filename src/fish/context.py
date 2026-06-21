from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from fish.config import CONFIG_DIR

CONTEXT_RULES_PATH = CONFIG_DIR / "context_rules.yaml"

DEFAULT_CONTEXT_RULES: dict[str, Any] = {
    "boosts": [
        {
            "when": {"location.in_vehicle": True},
            "kind_boost": {"sms": 0.15, "memory": 0.1},
            "tag_boost": {"navigation": 0.2, "destination": 0.2, "calendar": 0.1},
        },
    ]
}


def parse_context(context_json: str | dict[str, Any] | None) -> dict[str, Any]:
    if context_json is None:
        return {}
    if isinstance(context_json, dict):
        return context_json
    text = context_json.strip()
    if not text:
        return {}
    return json.loads(text)


def _get_nested(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_context_rules() -> dict[str, Any]:
    if not CONTEXT_RULES_PATH.exists():
        return DEFAULT_CONTEXT_RULES
    with CONTEXT_RULES_PATH.open() as f:
        return yaml.safe_load(f) or DEFAULT_CONTEXT_RULES


def augment_query(query: str, context: dict[str, Any] | None) -> str:
    if not context:
        return query
    hints: list[str] = []
    projects = context.get("active_projects")
    if projects:
        hints.append(f"projects: {', '.join(projects)}")
    intent = context.get("intent_hints")
    if intent:
        hints.append(f"intent: {', '.join(intent)}")
    location = context.get("location")
    if isinstance(location, dict):
        if location.get("in_vehicle"):
            vehicle = location.get("vehicle", "vehicle")
            hints.append(f"in {vehicle}")
    if not hints:
        return query
    return f"[context: {'; '.join(hints)}] {query}"


def compute_context_boosts(
    context: dict[str, Any] | None,
    *,
    kind: str,
    tags: list[str] | None = None,
) -> float:
    if not context:
        return 0.0
    rules = load_context_rules()
    boost = 0.0
    item_tags = set(tags or [])
    for rule in rules.get("boosts", []):
        when = rule.get("when") or {}
        matched = all(_get_nested(context, key) == value for key, value in when.items())
        if not matched:
            continue
        kind_boost = rule.get("kind_boost") or {}
        boost += float(kind_boost.get(kind, 0.0))
        tag_boost = rule.get("tag_boost") or {}
        for tag in item_tags:
            boost += float(tag_boost.get(tag, 0.0))
    return boost


def format_prompt(
    query: str,
    retrieved: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> str:
    ctx_block = json.dumps(context or {}, indent=2)
    retrieved_lines = []
    for i, item in enumerate(retrieved, 1):
        kind = item.get("kind", "item")
        occurred = (item.get("occurred_at") or "")[:10]
        preview = (item.get("body_text") or item.get("text_for_embed") or "")[:500]
        retrieved_lines.append(f"{i}. [{kind}] {occurred} id={item.get('id')}\n{preview}")
    retrieved_block = "\n\n".join(retrieved_lines) if retrieved_lines else "(none)"
    return (
        f"<context>\n{ctx_block}\n</context>\n"
        f"<retrieved>\n{retrieved_block}\n</retrieved>\n"
        f"<query>\n{query}\n</query>"
    )


def ensure_default_context_rules() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONTEXT_RULES_PATH.exists():
        CONTEXT_RULES_PATH.write_text(
            yaml.safe_dump(DEFAULT_CONTEXT_RULES, sort_keys=False)
        )
    return CONTEXT_RULES_PATH
