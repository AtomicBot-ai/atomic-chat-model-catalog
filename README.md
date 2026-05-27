# atomic-chat-model-catalog

Curated Hugging Face model catalog consumed by the
[Atomic Chat](https://github.com/AtomicBot-ai/Atomic-Chat) desktop client.

A GitHub Actions cron scrapes a whitelisted set of HF organizations every 12
hours, builds a single `catalog.json` with the same shape the Atomic Chat
client already consumes, plus a pre-built [MiniSearch](https://lucaong.github.io/minisearch/)
index for instant Google-like in-app search. Both artefacts are published as
assets on a `latest` GitHub Release; the client fetches them at startup and
caches them in `localStorage` for one hour.

The point of this repo is to:

1. Replace the legacy `janhq/model-catalog` source we were stuck with.
2. Curate which orgs we surface (GGUF Pareto frontier + MLX leaders +
   first-party model providers) so users see a high-quality default list.
3. Pre-build a search index so Hub search feels instant on any device.

Anything outside the whitelist is still reachable in-app via Hugging Face
fallback (direct `owner/repo` lookup, plus auto-fallback search when local
hits are sparse).

---

## Repo layout

```
config/
  orgs.json            # whitelist of HF orgs + per-org filters (THE only human-edited file)
  schema.orgs.json     # JSON Schema for orgs.json
  schema.catalog.json  # JSON Schema for the generated catalog.json artefact
scripts/
  scrape.py            # main scraper (Python 3.13, uv-managed)
  build_index.mjs      # Node helper that pre-builds the MiniSearch index
  pyproject.toml
  uv.lock
.github/workflows/
  cron.yml             # 12h cron + workflow_dispatch + repository_dispatch
  validate.yml         # PR validation: schemas + scraper dry-run
README.md
```

The published artefacts (not in git) live on the `latest` GitHub Release:

| Asset                  | Content                                                 |
| ---------------------- | ------------------------------------------------------- |
| `catalog.json` (gz)    | Array of `CatalogModel` (same shape used by the client) |
| `catalog.idx.json` (gz)| `MiniSearch.toJSON()` snapshot for instant client search|
| `stats.json`           | Per-org counts, elapsed time, etc.                      |

The fixed download URLs the client uses are:

```
https://github.com/AtomicBot-ai/atomic-chat-model-catalog/releases/latest/download/catalog.json
https://github.com/AtomicBot-ai/atomic-chat-model-catalog/releases/latest/download/catalog.idx.json
```

---

## How to add or remove an org

1. Open [`config/orgs.json`](config/orgs.json) on GitHub.
2. Click the pencil icon, append (or modify, or set `"active": false`) an
   entry. Bump `updated_at` to today.
3. Open a PR. CI validates the schema and runs a scraper dry-run against the
   diff.
4. After merge, the next 12h cron picks up the change — or trigger
   `cron.yml` → "Run workflow" with `force_rebuild=true` to publish
   immediately.

### Entry shape

```json
{ "name": "bartowski", "priority_boost": 1.3 }
```

| Field             | Required | Notes                                                                                        |
| ----------------- | -------- | -------------------------------------------------------------------------------------------- |
| `name`            | yes      | Hugging Face org id (case-sensitive). Used as `?author={name}` in the HF API.                |
| `priority_boost`  | no       | Static, platform-neutral baseline weight. Defaults to 1.0.                                   |
| `active`          | no       | Defaults to true. Set false to keep the entry for documentation but skip scraping.            |
| `min_downloads`   | no       | Skip repos with fewer downloads than this threshold.                                          |
| `tags_required`   | no       | Skip repos that do not carry ALL of these tags.                                               |
| `library_required`| no       | Optional exact-match on HF `library_name` (e.g. `"mlx"`).                                     |
| `max_repos`       | no       | Cap on the number of repos kept after sorting by downloads (top-N).                           |

### Why these orgs?

Curation is rationalised in the Atomic Chat plan document. tl;dr:

- **GGUF Pareto frontier** (`bartowski`, `unsloth`, `mradermacher`,
  `ubergarm`) — independently confirmed quant-quality leaders per the
  April 2026 KL-divergence benchmark on Qwen 3.5 27B
  (`localbench.substack.com`).
- **MLX leaders** (`mlx-community`, `prince-canuma`, `apple`,
  `Goekdeniz-Guelmez`) — covers the Apple Silicon backend used by the MLX
  extension in the Atomic Chat client.
- **First-party providers** (`Qwen`, `microsoft`, `meta-llama`,
  `mistralai`, `google`, `deepseek-ai`, `nvidia`, …) — official model
  homes for the top families.
- **Demoted but kept** (`lmstudio-community`, `MaziyarPanahi`,
  `QuantFactory`) — useful coverage but not Pareto-frontier on quant
  quality.
- **Marked inactive** (`TheBloke`, `janhq`) — kept as documentation;
  scraper skips them. `TheBloke` is essentially silent since 2024;
  `janhq` is the legacy upstream we are migrating away from.

Platform-aware boosts (e.g. "weight `mlx-community` heavily on macOS,
zero on Windows / Linux") live in the **client**, not here — the
artefact must stay platform-neutral so a single sync run serves all
operating systems.

---

## Catalog shape (do-not-break contract)

The scraper output `catalog.json` MUST be a **strict superset** of the
[`CatalogModel`](https://github.com/AtomicBot-ai/Atomic-Chat/blob/main/web-app/src/services/models/types.ts)
TypeScript interface used by the client. The client reads:

- `model_name`, `developer`, `downloads`, `created_at`, `description`
- `quants[]` with `{ model_id, path, file_size }` — `path` is a full
  `https://huggingface.co/{repo}/resolve/main/{file}` URL so the
  existing download pipeline works unchanged.
- `mmproj_models[]` with the same shape — needed by the vision UI.
- `safetensors_files[]` with `{ model_id, path, file_size, sha256? }` —
  `sha256` is required for MLX integrity verification when present.
- `is_mlx`, `num_quants`, `num_mmproj`, `num_safetensors` — derived.
- `readme` — full URL to the repo's README.md.

The scraper adds **derived** fields:

- `tags_normalized: string[]` — lowercased tags + extracted quant codes
  (`q4_k_m`, `iq4_xs`, …) used by MiniSearch.
- `last_modified: string` — passed through from HF.
- `likes: number` — passed through from HF.

Client code ignores unknown fields, so additions are backwards-compatible.

---

## Sync schedule

`cron.yml` runs:

- **`schedule`**: `'0 */12 * * *'` (twice daily, 00:00 + 12:00 UTC).
- **`workflow_dispatch`**: manual emergency re-sync with optional
  `force_rebuild` (bypass per-repo ETag cache) and `orgs_subset`
  (comma-separated list to limit the run).
- **`repository_dispatch`**: external trigger for future webhooks
  (e.g. when AtomicChat publishes a new HF model).

End-to-end staleness budget: up to **12h** (next scheduled scrape) plus
up to **1h** (client cache TTL) ≈ **13h**. Manual trigger collapses to
minutes.

---

## CI validation

[`.github/workflows/validate.yml`](.github/workflows/validate.yml) runs on
every push and PR:

- `ajv` validates `config/orgs.json` against `config/schema.orgs.json`.
- Scraper smoke test: build the catalog for one tiny org (`AtomicChat`)
  in dry-run mode and validate the output against
  `config/schema.catalog.json`.

Local validation:

```bash
npx ajv-cli@5 validate \
  -s config/schema.orgs.json \
  -d config/orgs.json \
  --strict=false

# Dry-run the scraper (writes to ./out/, no upload)
cd scripts
uv run python scrape.py --dry-run --orgs AtomicChat
node build_index.mjs --in ../out/catalog.json --out ../out/catalog.idx.json
```

---

## Security

- No API keys live here. The scraper uses an optional `HF_TOKEN` secret
  to lift the anonymous HF API rate limit (5k req/h → 30k req/h with a
  token); without it, the scraper still runs but takes longer.
- Generated artefacts are signed only by GitHub's release attestation —
  the client treats `catalog.json` as untrusted JSON and validates it
  against the catalog schema before using it.
- The published `path` URLs always point at `huggingface.co` — the
  scraper rejects any path that would break the prefix contract.

---

## Why a separate repo?

`atomic-chat-conf` holds tiny, human-edited config files (provider
registry, recommended-models list). A model catalog is generated by a
machine, can grow to several megabytes, and triggers a Release on every
cron tick. Putting that volume in `atomic-chat-conf` would drown the
hand-curated content in autocommits.

Both repos are consumed by the same client; see [`web-app/src/services/AGENTS.md`](https://github.com/AtomicBot-ai/Atomic-Chat/blob/main/web-app/src/services/AGENTS.md)
for the loader pattern.
