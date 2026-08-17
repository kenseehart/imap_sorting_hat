# Canonical message identity

Fish stores many sources in one SQLite corpus. Integer autoincrement IDs are
**surrogates** (Qdrant point ids, training FKs). The **ingest identity** is:

```text
{source_id}.{message_id}
```

stored as `corpus_items.source_key` (UNIQUE). Re-sync / re-import must
lookup-then-update on this key — never blind-insert, never treat
`messages.id == corpus_items.id` as identity.

## Why

1. **Cross-source PK collision** — Non-email rows can occupy the same integer
   space as IMAP messages; preferring `corpus.id = messages.id` caused skewed
   ids and poisoned header joins.
2. **Retrievability** — IMAP UID is a folder slot, not a portable id. Prefer
   RFC Message-ID (Dovecot) or `X-GM-MSGID` (Gmail).
3. **Write-time uniqueness** — Dedup at ingest on the composite key.
4. **Folder / label moves** — Identity survives moves; folder/uid stay as
   locator metadata in payload / `messages`.
5. **ANN soft-coupling** — Keep integer PK so identity migration does not
   require rewriting Qdrant points.

## Source codes (5-char)

| source_id | Platform / account |
|-----------|--------------------|
| `emksc` | email `ken@seehart.com` |
| `emkag` | email `ken@agi.green` |
| `emkgc` | email `kenseehart@gmail.com` |
| `smskn` | Android SMS (personal filter) |
| `memag` | agent memory (`fish_memory_upsert`) |
| `cckag` | Claude Code (`{session_uuid}:{message_uuid}`) |
| `cuksc` | Cursor — **not frozen** (open item below) |

Mapped in `fish.identity.EMAIL_SOURCE_BY_ADDRESS` and helpers in
`src/fish/identity.py`.

## Email derivation

| Account | `message_id` |
|---------|----------------|
| `emksc` / `emkag` | RFC Message-ID with `<>` stripped; else `syn:` + sha256 |
| `emkgc` | decimal `X-GM-MSGID` from IMAP |

**Synthetic fallback** (Dovecot when Message-ID missing; Gmail backfill only):

```text
syn: + sha256(from|date|subject|body[:4096])[:32]
```

- Algorithm: SHA-256; body length `N=4096` UTF-8 bytes.
- Payload / `header_json` set `synthetic: true`.
- Synthetic ids are **not** re-fetchable from the mail server.

**Gmail backfill:** until `gm_msgid` is present, migration may use
`rfc:{Message-ID}` or synthetic (`allow_gmail_rfc_fallback=True`). Live IMAP
sync requires `X-GM-MSGID`.

**Locator metadata** (mutable): `account_id`, `folder`, `uid` in payload;
`messages` still UNIQUE on `(account_id, folder, uid)`.

## Other sources

| Kind | Canonical form |
|------|----------------|
| SMS | `smskn.{native_sms_id\|syn:hash}` |
| Memory | `memag.{sha256(fact.lower())[:24]}` |
| Claude Code | `cckag.{session_uuid}:{message_uuid}` |
| ChatGPT / Claude.ai exports | legacy `chatgpt:` / `claude:` prefixes until migrated |

## Storage

| Column | Role |
|--------|------|
| `corpus_items.source_key` | Composite canonical id (UNIQUE ingest key) |
| `corpus_items.id` | Integer surrogate PK → Qdrant / `training_samples` |
| `messages.canonical_id` | Same composite as email corpus `source_key` |
| `messages.gm_msgid` | Gmail `X-GM-MSGID` when known |
| `messages.message_id` | Raw RFC Message-ID header |

Delete / unindex resolve corpus via `canonical_id` or payload locator — never
`DELETE … WHERE corpus_items.id = messages.id`.

### Why keep the integer surrogate?

This is **correct relational design**, not a training shortcut.

- **Natural key** (`source_key`) enforces uniqueness and re-sync dedup.
- **Surrogate PK** (`id`) is the stable internal pointer for FKs and ANN
  point ids. Those systems need a compact immutable handle; the composite
  string is the *identity*, the integer is the *row address*.
- The corpus integrity bug was **preferring** `corpus.id = messages.id` and
  joining on that equality — not the existence of an autoincrement column.
  Equal integers across tables are meaningless once sources share one store.

Replacing `id` with the composite string everywhere (Qdrant string ids, FK
rewrites) would not make the database more correct if `source_key` is already
UNIQUE and all cross-table links use it. Do that only if we later want a
single public id surface — optional Phase D, not required for coherence.

Migration **updates `source_key` in place** so existing `training_samples.corpus_item_id`
and Qdrant points remain valid. Labels collected under poisoned headers may
still be wrong; purge/re-label if eval shows it — that is training quality,
not schema coherence.

**Duplicate Message-IDs:** the same RFC Message-ID can appear under different
IMAP locators (copies / multi-folder). Migration keeps the first corpus row;
duplicate corpus mirrors are deleted after migrate (messages locators remain
so IMAP sync still works). Uniqueness is on canonical identity.

Gmail rows may temporarily use `emkgc.rfc:…` / `emkgc.syn:…` until the next
IMAP sync fills `X-GM-MSGID` and upgrades `source_key`.

## Migration

```bash
fish migrate-canonical-ids --dry-run
fish migrate-canonical-ids
```

Rewrites legacy `imap:…` / `android_sms:…` / `memory:…` source keys to
composites **without** changing integer PKs. Collisions are logged and skipped.

## Open item: Cursor (`cuksc`)

**Status:** deferred — do not invent a format.

**Inspection (2026-08-16):**

| Store | Finding |
|-------|---------|
| `…/globalStorage/conversation-search.db` | Conversation UUID (`conversations.id`); FTS body; **no per-message ids** |
| `…/workspaceStorage/*/state.vscdb` `composer.composerData` | Composer UUIDs only |
| `~/.cursor/projects/*/agent-transcripts/{uuid}.jsonl` | Conversation UUID in path; lines are `{role, message}` **without** stable message UUID |
| `…/globalStorage/state.vscdb` (~6 GB) | **Unreadable** here (`disk I/O error` via WSL) |

When a retrievable per-message id is confirmed (e.g. bubble id in
`cursorDiskKV`), freeze `cuksc.{composer_or_conversation_uuid}:{message_uuid}`
in `identity.cursor_canonical_id` and document it here. Until then
`cursor_canonical_id()` raises `NotImplementedError`.
