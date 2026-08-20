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

| Config | `chunk_repr` | Sharing / scoring | Notes |
|--------|--------------|-------------------|-------|
| `joint_h{1536,3072}` | `joint` | dual | \(E(\mathrm{text}(h{+}b))\); `hidden_dim` = adapter width |
| `siamese_h{1536,3072}` | `joint` | siamese | shared \(A=A_q=A_c\) |
| `split_h{1536,3072}` | `split` | dual | \(E(h)\,\|\,E(b)\) |
| `rerank_h{1536,3072}` | `split` | dual + `mlp_head` | \(\sigma(\mathrm{MLP}(A_q\|A_c))\) — reranks split |
| `smoke_*` | … | … | overfit pipeline checks |

`chunk_repr`: **joint** = join text then embed once; **split** = embed header/body
separately then concat vectors. `fish prism-train --config bakeoff` expands to
the 8 arch×hidden configs (`joint_h*`, `siamese_h*`, `split_h*`, `rerank_h*`) —
it excludes `smoke_*`, YAML anchors (`_*`), **and** the legacy `personal_*`
aliases so the sweep stays a fair 8-way compare (`list_bakeoff_config_names()`
in `src/fish/prism/configs.py`). Pass `personal_*` names explicitly if you
want the old models. Freeze defaults to `--chunk-repr both` (joint + split in
one `.tcz`).

`siamese` is incompatible with `split` (asymmetric \(A_c\) input dim).
`rerank_*` is not an ANN index; NWRA applies it as a second-stage scorer on the
same labeled candidate sets (split-style vectors).

Adapters are plain MLPs (**no residual α**). Dual PRISM uses separate \(A_q, A_c\);
siamese uses one shared adapter (still serialized into both `.prz` slots). Overfit
smoke (train-set only; `split` auto-preps field embeds for labeled items —
not the full corpus):

```bash
fish prism-train --config smoke_joint --overfit --json
fish prism-train --config smoke_split --overfit --json
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

Full-corpus field backfill (for deploying `split` retrieval over all mail):

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
fish prism-train --config smoke_joint --overfit
fish prism-train --config smoke_split --overfit --from-db   # freeze both then train
fish prism-train --config smoke_joint --overfit --gpu     # CUDA A/B vs CPU

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
| Competitive bakeoff | `fish.prism.nwra_eval` — see below |

### NWRA (competitive multi-model eval)

`src/fish/prism/nwra_eval.py` ranks candidates within each query and compares
model order to the RA-optimal order:

\[
\mathrm{NWRA} = \frac{\mathrm{WRA}(\text{model order})}{\mathrm{WRA}(\text{RA-optimal order})},
\qquad \mathrm{WRA}(\text{rels}) = \frac{\sum_i w_i \, r_i}{\sum_i w_i},\quad w_i = \frac{1}{2+i}
\]

Also reports Spearman(model_score, RA) over the same pairs. Null when the
perfect-order WRA is 0 (all-zero relevance query).

**Scores from the frozen dual `.tcz` only** (`q` / `c_joint` / `c_split` /
`rel` — the same arrays `fish prism-train` uses), batched per model with
NumPy. It does **not** open `fish.db` or hit SQLite in the hot path — an
earlier version loaded `training_samples` live and re-fetched
header/body raw embeddings per pair per model, which meant ~10⁵ SQLite
lookups plus full blob decode of ~11k embeddings just to build a ranking
report. Same anti-pattern as copying `fish.db` to a RunPod: using the
canonical DB as scratch compute for something a 100 MB frozen artifact
already covers. Do not revert to a live-DB per-pair path.

Registers as compute task `nwra` (module `fish`) via `TaskProgress` so it's
visible in `compute tasks` / the tasks UI — a job that never wraps itself in
`TaskProgress` runs invisibly even at 100% CPU (see `compute/AGENTS.md`).

```bash
# On the host holding the frozen .tcz (or with FISH_DATA_DIR pointed at it):
python -m fish.prism.nwra_eval \
  --corpus latest \
  --model-id joint_h1536.<ts> --model-id split_h3072.<ts> ... \
  --out /data/fish/nwra_report.json
```

`legacy` (identity/raw cosine) is always included as the baseline system.
`--corpus` accepts `latest`, a `train_corpus_*` id, or a path — same
resolution as `fish prism-train --corpus`.

## Ops

| Artifact | Cloud |
|----------|-------|
| Corpus | `/data/fish/fish.db` |
| Models | `/data/fish/models/{model_id}.prz` |
| Env | `FISH_PRISM_MODEL={model_id}` |

See [`cloud.md`](cloud.md).
