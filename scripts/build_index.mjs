#!/usr/bin/env node
/**
 * Pre-build the MiniSearch index for the Atomic Chat client.
 *
 * Consumes the `catalog.json` written by `scrape.py` and emits
 * `catalog.idx.json` -- the JSON-serialised form of a configured
 * MiniSearch instance. The client loads it via `MiniSearch.loadJSON(...)`
 * and starts answering queries immediately, without paying the per-launch
 * indexing cost (a few hundred ms on ~20k models).
 *
 * Field weights and tokenisation MUST stay in sync with the client's
 * search service (`web-app/src/services/model-search.ts`). When you bump
 * one, bump both, and bump `INDEX_VERSION` so the client can refuse to
 * load a snapshot built with an incompatible config.
 *
 * Usage:
 *   node build_index.mjs --in ../out/catalog.json --out ../out/catalog.idx.json
 */

import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import MiniSearch from 'minisearch'

const INDEX_VERSION = 1

function parseArgs(argv) {
  const args = { in: '../out/catalog.json', out: '../out/catalog.idx.json' }
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--in') args.in = argv[++i]
    else if (a === '--out') args.out = argv[++i]
    else if (a === '--help' || a === '-h') {
      console.log(
        'Usage: build_index.mjs --in <catalog.json> --out <catalog.idx.json>'
      )
      process.exit(0)
    } else {
      console.error(`Unknown argument: ${a}`)
      process.exit(2)
    }
  }
  return args
}

function loadCatalog(filePath) {
  const text = fs.readFileSync(filePath, 'utf8')
  const data = JSON.parse(text)
  if (!data || !Array.isArray(data.models)) {
    throw new Error(`${filePath}: expected { models: [...] }`)
  }
  return data
}

function buildIndex(models) {
  const miniSearch = new MiniSearch({
    idField: 'id',
    fields: ['model_name', 'developer', 'tags_normalized', 'description'],
    storeFields: [
      'model_name',
      'developer',
      'downloads',
      'likes',
      'is_mlx',
      'created_at',
      'last_modified',
      'num_quants',
      'num_mmproj',
      'num_safetensors',
    ],
    searchOptions: {
      boost: {
        model_name: 5,
        developer: 3,
        tags_normalized: 2,
        description: 1,
      },
      fuzzy: 0.2,
      prefix: true,
      combineWith: 'AND',
    },
    tokenize: (text) =>
      String(text)
        .toLowerCase()
        .split(/[\s\-_./:,;()[\]<>+]+/u)
        .filter(Boolean),
    processTerm: (term) => (term.length < 2 ? null : term.toLowerCase()),
  })

  const documents = models.map((m, i) => ({
    id: m.model_name || `idx-${i}`,
    model_name: m.model_name || '',
    developer: m.developer || '',
    tags_normalized: Array.isArray(m.tags_normalized)
      ? m.tags_normalized.join(' ')
      : '',
    description: typeof m.description === 'string' ? m.description : '',
    downloads: Number(m.downloads || 0),
    likes: Number(m.likes || 0),
    is_mlx: Boolean(m.is_mlx),
    created_at: m.created_at || '',
    last_modified: m.last_modified || '',
    num_quants: Number(m.num_quants || 0),
    num_mmproj: Number(m.num_mmproj || 0),
    num_safetensors: Number(m.num_safetensors || 0),
  }))

  miniSearch.addAll(documents)
  return miniSearch
}

function main() {
  const args = parseArgs(process.argv)
  const inPath = path.resolve(args.in)
  const outPath = path.resolve(args.out)

  console.log(`[build_index] reading ${inPath}`)
  const catalog = loadCatalog(inPath)
  console.log(`[build_index] indexing ${catalog.models.length} models`)

  const t0 = Date.now()
  const index = buildIndex(catalog.models)
  const elapsedMs = Date.now() - t0
  console.log(`[build_index] built MiniSearch in ${elapsedMs}ms`)

  const wrapped = {
    index_version: INDEX_VERSION,
    catalog_updated_at: catalog.updated_at,
    catalog_total_models: catalog.models.length,
    minisearch: index.toJSON(),
  }

  fs.mkdirSync(path.dirname(outPath), { recursive: true })
  fs.writeFileSync(outPath, JSON.stringify(wrapped), 'utf8')
  const sizeMb = fs.statSync(outPath).size / 1024 / 1024
  console.log(
    `[build_index] wrote ${outPath} (${sizeMb.toFixed(2)} MB, index_version=${INDEX_VERSION})`
  )
}

main()
