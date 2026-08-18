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

Table `retrieval_models` registers every searchable index. ANN lives in **Qdrant**;
frozen OpenAI vectors live in SQLite `corpus_raw_embeddings` (durable — **not**
Qdrant-only):

| SQLite column | Meaning | In Qdrant? |
|---------------|---------|------------|
| `embedding` | Combined `text_for_embed` vector \(c\) | Yes → `fish_legacy` |
| `header_embedding` | Embed of `corpus_items.header_json` | **No** (composition / train) |
| `body_embedding` | Embed of `corpus_items.body_text` | **No** (composition / train) |

`header_json` is structured metadata (From/To/Subject/… as JSON). Field vectors
exist so PRISM can compute e.g. \(A_c(E(h)\,\|\,E(b))\) without calling OpenAI again.

Configs in [`config/prism_models.yaml`](../config/prism_models.yaml):

| Config | `chunk_repr` | `adapter_sharing` | Chunk vector |
|--------|--------------|-------------------|--------------|
| `smoke_combined` / `personal_combined` | `combined` | `dual` | \(E(h{+}b)\) = embed(`text_for_embed`) |
| `smoke_fields` / `personal_fields` | `header_body` | `dual` | \(E(h)\,\&\,E(b)\) = concat(header, body) embeds |
| `smoke_siamese` / `personal_siamese` | `combined` | `siamese` | same as combined; shared \(A=A_q=A_c\) |
| `personal_rerank` | `header_body` | `dual` + `scoring: mlp_head` | \(\sigma(\mathrm{MLP}(A_q(E(q))\,\|\,A_c(E(h)\|E(b))))\) — no cosine; reranks fields candidates |

`siamese` is incompatible with `header_body` (asymmetric \(A_c\) input dim).
`personal_rerank` is not an ANN index; NWRA applies it as a second-stage scorer on the same labeled candidate sets (fields-style vectors).

Adapters are plain MLPs (**no residual α**). Dual PRISM uses separate \(A_q, A_c\);
siamese uses one shared adapter (still serialized into both `.prz` slots). Overfit
smoke (train-set only; `header_body` auto-preps field embeds for labeled items —
not the full corpus):

```bash
fish prism-train --config smoke_combined --overfit --json
fish prism-train --config smoke_fields --overfit --json
fish prism-train --config smoke_siamese --overfit --json
```

Train-set `spearman_prism` should rise well above `spearman_raw` if the pipeline works.

Checkpoints: each epoch writes `models/checkpoints/{config}.pt` (weights + optimizer).
Re-run the same command to **resume**; `--fresh` discards the checkpoint and starts a
new `model_id`. Finished runs delete the checkpoint after writing the `.prz`.

Training snapshot: ``fish corpus freeze-training`` writes
``models/corpora/train_corpus_{UTC}.tcz`` (v2: JSON header with ``n_labels``,
then a zip of float32 ``q``/``c``/``rel``). List with ``fish corpus corpora``.
Keeps at most **3** frozen corpora (oldest deleted automatically). v1 bare-zip
``.tcz`` files are deleted, not migrated.
``fish prism-train`` defaults to ``--corpus latest`` and **never opens
fish.db** for epochs (resume fingerprint binds to the ``.tcz`` ``corpus_id``).
Freeze a new snapshot when labels change; ``--from-db`` freezes then trains;
``--gpu`` is an alias for ``--device cuda``; ``--no-register`` skips the post-train
DB write entirely.

Each epoch also updates `models/checkpoints/{config}.progress.json` and prints a line:
`epoch E/N  elapsed=…s  ep/s=…  holdout=…  best=…@epoch  device=…`
(plus `@compute progress E N` on stderr for detached job tracking).

Device: `--device auto|cpu|cuda|cuda:N` (default `auto` from config). Use `--device cuda`
on a GPU box for a CPU-vs-GPU speed test; adapters are small so gains may be modest.

Personal configs use **early stopping**: `epochs` is a ceiling (default 200);
training stops when holdout Spearman fails to improve by `early_stop_min_delta`
for `early_stop_patience` epochs (default 15 / 0.001). The best holdout weights
are written to the `.prz`. Smoke configs set `early_stop_patience: 0` (fixed
epoch count).

Full-corpus field backfill (for deploying `header_body` retrieval over all mail):

```bash
fish embed --fields          # all missing header/body vectors
fish embed --fields --limit 500
fish embed --fields --training-only   # labeled training items only (manual)
```

New `fish embed` / sync embeds already write combined + header + body.

| `model_id` | Qdrant collection (`vec_table`) | Meaning |
|------------|----------------------------------|---------|
| `legacy` | `fish_legacy` | Raw cosine (OpenAI \(c\)) |
| `{config}.{timestamp}` | `fish_{config}_{timestamp}` | \(A_c(c)\) for that `.prz` |

- `model_id = {config_name}.{UTC_timestamp}` e.g. `smoke.20260813T120000Z`
- Weights file: `models/{model_id}.prz` (numpy zip, not JSON)
- MLP hyperparameters: `config/prism_models.yaml` (override `~/.config/fish/prism_models.yaml`)
- Active model: `FISH_PRISM_MODEL={model_id}` or `retrieval_models.active`
- Env: `FISH_QDRANT_URL` (required; fail-fast if unreachable), optional `FISH_QDRANT_API_KEY`
- Delete / supersede calls `unindex_corpus_item` across raw SQLite + all Qdrant collections
- `fish index-cleanup` removes orphan points / raw rows
- Corrupt recovery: `wipe_all_vector_indexes` deletes/recreates Qdrant collections (keeps `corpus_raw_embeddings`); then `fish qdrant-reindex` / `fish prism-reembed`
- `fish qdrant-reindex` is **resumable**: existing point ids are skipped; use `--force` to rewrite all
- `fish prism-reembed` is **resumable** the same way (streamed pages + skip existing; `--force` rewrites)
- One-time from sqlite-vec: `fish qdrant-migrate` (copy blobs + upsert legacy collection)

## Training pipeline (start small)

```bash
# 0. Curated seeds (optional; dump anytime)
fish corpus add-curated                 # loads config/gold_queries.jsonl → origin=curated
fish corpus queries --origin curated --json
fish corpus queries --origin gold --json   # logged searches

# 1. Gold (logged) + curated + synth fill if needed
fish corpus collect --retriever legacy --min-queries 20 --top-k 10

# 2. Cold-start hard positives
fish corpus inject-positives --query "Burning Man" \
  --like "%Interaction Café%,%Burn CREW%" --since 2026-01-01

# 3. Label (no Fish write lock — safe beside prism-train epochs)
fish corpus label --limit 400

# 3b. Freeze labeled pairs → models/corpora/train_corpus_*.tcz
fish corpus freeze-training --chunk-repr combined

# 4. Train from frozen .tcz (default --corpus latest; epochs never touch fish.db)
#    (resumable via models/checkpoints/{config}.pt; --fresh to restart)
fish prism-train --config smoke_combined --overfit
fish prism-train --config smoke_fields --overfit --from-db   # freeze header_body then train
fish prism-train --config smoke_combined --overfit --gpu     # CUDA A/B vs CPU

# 5. Embed a smoke slice (OpenAI once → SQLite raw + Qdrant legacy)
fish embed --limit 100 --kinds email \
  --like "%Interaction Café%,%Burn CREW%,%Burning Man%" --since 2026-01-01

# 6. Fill that model's Qdrant collection from raw (no OpenAI)
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
