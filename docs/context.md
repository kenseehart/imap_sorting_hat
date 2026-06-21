# Session context JSON

Ephemeral session state passed to `fish_search` / MCP `fish_search` via `context_json`. Not stored in the corpus.

## Example

```json
{
  "location": { "in_vehicle": true, "vehicle": "tesla" },
  "intent_hints": ["navigation", "destinations", "calendar"],
  "active_projects": ["cristopoly", "fish"],
  "time_local": "2026-06-20T14:30:00-07:00"
}
```

## Behavior

1. **Query augmentation** — context is summarized and prepended to the embed query before PRISM/query embedding.
2. **Retrieval boosts** — rules in `~/.config/fish/context_rules.yaml` add score boosts by kind and tag when context fields match.
3. **Prompt assembly** — search responses include a `<context>` / `<retrieved>` / `<query>` block when context is provided.

## Boost rules

Default rules ship on first MCP startup. Edit `context_rules.yaml` to customize:

```yaml
boosts:
  - when:
      location.in_vehicle: true
    kind_boost:
      sms: 0.15
      memory: 0.1
    tag_boost:
      navigation: 0.2
      destination: 0.2
```

Rule keys use dot paths into the context object (`location.in_vehicle` → `context["location"]["in_vehicle"]`).
