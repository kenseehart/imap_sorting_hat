# PRISM for fish (personal email corpus)

PRISM is **not** standard semantic similarity. It trains dual adapters so that
**adapted cosine approximates RelevanceAgent scores**.

\[
\mathcal{L} = \mathrm{MSE}\big(\cos(A_q(q),\; A_c(c)),\; r\big)
\]

| Symbol | Meaning in fish |
|--------|-----------------|
| \(q, c\) | Frozen OpenAI embeddings (`text-embedding-3-small`) |
| \(A_q, A_c\) | Dual residual adapters in a binary zip `.prz` |
| \(r\) | **RelevanceAgent** label `target_relevance` ∈ [0, 1] |

## Retrieval models & indexes

Table `retrieval_models` registers every searchable index:

| `model_id` | Index | Meaning |
|------------|-------|---------|
| `legacy` | `corpus_vec` | Raw cosine (OpenAI \(c\)) |
| `{config}.{timestamp}` | `corpus_vec__{config}_{timestamp}` | \(A_c(c)\) for that `.prz` |

- `model_id = {config_name}.{UTC_timestamp}` e.g. `smoke.20260813T120000Z`
- Weights file: `models/{model_id}.prz` (numpy zip, not JSON)
- MLP hyperparameters: `config/prism_models.yaml` (override `~/.config/fish/prism_models.yaml`)
- Active model: `FISH_PRISM_MODEL={model_id}` or `retrieval_models.active`
- Vec tables store **only** `rowid → float[1536]` (reference `corpus_items.id`). No email text in indexes.
- Delete / supersede calls `unindex_corpus_item` across **all** registered indexes.
- `fish index-cleanup` removes orphan ANN rows.
- Corrupt recovery: `wipe_all_vector_indexes` **retires** indexes by pointing
  `retrieval_models.vec_table` at a fresh empty table (no DROP/RENAME/COUNT on
  huge sqlite-vec — those hang). Old vec tables stay as disk trash for offline cleanup.

## Training pipeline (start small)

```bash
# 0. Curated gold queries (optional; dump anytime)
fish corpus add-gold                    # loads config/gold_queries.jsonl
fish corpus queries --origin gold --json

# 1. Gold/real queries + synthetic fill if needed
fish corpus collect --retriever legacy --min-queries 20 --top-k 10

# 2. Cold-start hard positives
fish corpus inject-positives --query "Burning Man" \
  --like "%Interaction Café%,%Burn CREW%" --since 2026-01-01

# 3. Label
fish corpus label --limit 400

# 4. Train → registers model_id + writes binary .prz
fish prism-train --config smoke

# 5. Embed a smoke slice into legacy index (OpenAI once)
fish embed --limit 100 --kinds email \
  --like "%Interaction Café%,%Burn CREW%,%Burning Man%" --since 2026-01-01

# 6. Fill that model's ANN from raw (no OpenAI)
fish prism-reembed --limit 100 --like "%Interaction Café%,%Burn CREW%"

# 7. Eval
FISH_PRISM_MODEL=smoke.<timestamp> fish search "Burning Man" --since 2026-07-16
```

## Eval

| Check | How |
|-------|-----|
| Adapter fidelity | `fish prism-train` reports Spearman raw vs PRISM on held-out pairs |
| Personal goal | Adapted-cosine top-10 for `Burning Man` includes camp mail without requiring those words |

## Ops

| Artifact | Cloud |
|----------|-------|
| Corpus | `/data/fish/fish.db` |
| Models | `/data/fish/models/{model_id}.prz` |
| Env | `FISH_PRISM_MODEL={model_id}` |

See [`cloud.md`](cloud.md).
