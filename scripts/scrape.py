"""Hugging Face scraper for the Atomic Chat curated model catalog.

Reads the whitelist from ``../config/orgs.json``, paginates ``GET
/api/models?author={org}`` for every active org, fetches per-repo file
metadata (``?blobs=true&files_metadata=true``) for the surviving repos,
and emits two files into ``out/``:

* ``catalog.json`` -- array of ``CatalogModel`` entries with shape
  identical to today's ``web-app/src/services/models/types.ts``
  ``CatalogModel`` interface, plus derived ``tags_normalized`` and
  ``last_modified`` fields.
* ``stats.json`` -- per-org counts and elapsed time for the cron run.

The companion ``build_index.mjs`` consumes ``catalog.json`` and produces
``catalog.idx.json`` (a MiniSearch snapshot for instant client-side
search).

Usage:

    uv run python scrape.py [--dry-run] [--orgs name1,name2] [--out PATH]
                            [--limit N] [--force]

The scraper is **read-only** against Hugging Face: it never writes to
HF, only reads public model metadata. ``HF_TOKEN`` (env var) is
optional and only raises the rate-limit ceiling.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

CATALOG_MANIFEST_VERSION = 1
CATALOG_SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORGS_PATH = REPO_ROOT / "config" / "orgs.json"
DEFAULT_OUT_DIR = REPO_ROOT / "out"

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co"

PER_PAGE = 1000
DEFAULT_PER_ORG_CAP = 800
# HF starts throttling around 8-10 concurrent reads from the same IP / token
# bucket. We default to 4 so prolific shotgun-quant orgs (MaziyarPanahi,
# mradermacher, QuantFactory) do not turn into a 429-storm. Override via
# --concurrency on the CLI when a fast token is available.
DEFAULT_DETAIL_CONCURRENCY = 4
# Listings run sequentially: at 21 orgs they finish in well under a minute
# even single-threaded, and parallelising them spikes the bucket needlessly.
DEFAULT_LIST_CONCURRENCY = 1
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=60.0)
# Maximum seconds to honour from an HF `Retry-After` response. HF
# occasionally returns minute-scale waits during bursts; we cap to keep CI
# under the 60-minute timeout.
MAX_RETRY_AFTER_SECONDS = 45.0
# Steady-state request rate target. HF Hub API quotas (Sept '25):
#   anonymous = 500 / 5min = 1.67 req/s
#   Free      = 1000 / 5min = 3.33 req/s
#   PRO       = 2500 / 5min = 8.33 req/s
# We aim for ~75% of Free so 5-min fixed-window resets do not push us over
# the edge. Override via --rate-limit when a stronger token is available.
DEFAULT_RATE_LIMIT_RPS = 2.5

QUANT_CODE_RE = re.compile(
    r"\b(?:q\d_k(?:_[a-z])?|q\d_\d(?:_[a-z])?|iq\d_[a-z0-9_]+|f16|f32|bf16|fp16|fp32|"
    r"int4|int8|4bit|8bit|3bit|2bit|tq[234]_\d[sk]?|tq[234]_\d|"
    r"ud-?q\d_[a-z_]+|ud-?iq\d_[a-z_]+|ud-?q\d_[xk]_xl)\b",
    re.IGNORECASE,
)

GGUF_SUFFIX = ".gguf"
SAFETENSORS_SUFFIX = ".safetensors"
MLX_TAGS = frozenset({"mlx"})

logger = logging.getLogger("scraper")


@dataclass(slots=True)
class OrgConfig:
    """Sanitised view of one entry from ``config/orgs.json``."""

    name: str
    priority_boost: float = 1.0
    active: bool = True
    min_downloads: int = 0
    tags_required: tuple[str, ...] = ()
    library_required: str | None = None
    max_repos: int | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> OrgConfig:
        return cls(
            name=str(raw["name"]).strip(),
            priority_boost=float(raw.get("priority_boost", 1.0)),
            active=bool(raw.get("active", True)),
            min_downloads=int(raw.get("min_downloads", 0)),
            tags_required=tuple(str(t) for t in raw.get("tags_required", [])),
            library_required=raw.get("library_required"),
            max_repos=raw.get("max_repos"),
        )


@dataclass(slots=True)
class ScrapeStats:
    by_org: dict[str, int] = field(default_factory=dict)
    by_org_skipped: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    mlx_models: int = 0
    gguf_models: int = 0
    total_models: int = 0
    errors: list[str] = field(default_factory=list)


def load_orgs(path: Path) -> list[OrgConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError(
            f"orgs.json schema_version must be 1 (got {raw.get('schema_version')!r})"
        )
    return [OrgConfig.from_raw(o) for o in raw["orgs"]]


def http_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    headers = {
        "Accept": "application/json",
        "User-Agent": "atomic-chat-model-catalog/0.1 (+https://github.com/AtomicBot-ai/atomic-chat-model-catalog)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Return seconds to sleep based on HF rate-limit response headers.

    HF responds to 429s with either ``Retry-After: <seconds>`` (RFC 7231),
    ``Retry-After: <HTTP-date>``, or — sometimes — ``X-RateLimit-Reset``
    (Unix epoch seconds). We honour whichever is present.
    """
    raw = response.headers.get("retry-after")
    if raw:
        raw = raw.strip()
        try:
            return min(float(raw), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
        try:
            target = parsedate_to_datetime(raw)
            if target.tzinfo is None:
                target = target.replace(tzinfo=UTC)
            delta = (target - datetime.now(tz=UTC)).total_seconds()
            if delta > 0:
                return min(delta, MAX_RETRY_AFTER_SECONDS)
        except (TypeError, ValueError):
            pass
    reset = response.headers.get("x-ratelimit-reset")
    if reset:
        try:
            delta = float(reset) - datetime.now(tz=UTC).timestamp()
            if delta > 0:
                return min(delta, MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return None


class RateLimited(httpx.HTTPError):
    """Carries the suggested cooldown from HF's response."""

    def __init__(self, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AsyncRateLimiter:
    """Simple async token bucket that paces all HTTP calls to ``rate_rps`` per second.

    HF Hub APIs are throttled per token (or per IP when anonymous) on
    5-minute fixed windows. A token bucket sized at ~75% of the plan's
    steady-state rate keeps the scraper well under the quota even when
    the previous window's allotment has been spent and the new window
    has not yet reset. Burstiness is intentionally low (``capacity=1``)
    because HF's per-second limiter is much tighter than the 5-min one.
    """

    __slots__ = ("rate_rps", "_min_interval", "_next_available", "_lock")

    def __init__(self, rate_rps: float) -> None:
        if rate_rps <= 0:
            raise ValueError(f"rate_rps must be > 0, got {rate_rps!r}")
        self.rate_rps = rate_rps
        self._min_interval = 1.0 / rate_rps
        self._next_available = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_available - now
            if wait > 0:
                # Release the lock while sleeping so other coroutines can queue.
                self._next_available = now + wait + self._min_interval
            else:
                self._next_available = now + self._min_interval
                wait = 0.0
        if wait > 0:
            await asyncio.sleep(wait)


# Module-level singleton populated by ``run()``. ``get_json`` paces every
# request through this limiter so all parallel tasks share the same budget.
_rate_limiter: AsyncRateLimiter | None = None


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_attempts: int = 6,
) -> tuple[Any, httpx.Response]:
    """``GET url`` with retries that honour HF's ``Retry-After`` headers.

    Returns ``(json, response)``. Soft-failure status codes (``401/403/404/410``)
    return ``(None, response)`` so the caller can decide whether to skip.
    """
    attempt = 0
    last_response: httpx.Response | None = None
    while attempt < max_attempts:
        attempt += 1
        if _rate_limiter is not None:
            await _rate_limiter.acquire()
        try:
            r = await client.get(url, params=params)
        except (httpx.HTTPError, httpx.RequestError):
            if attempt >= max_attempts:
                raise
            # Exponential backoff for transport-level errors with jitter.
            await asyncio.sleep(min(2 ** attempt, 30.0) + random.random())
            continue

        last_response = r

        if r.status_code == 429:
            cooldown = _parse_retry_after(r) or min(2 ** attempt, 30.0)
            # Add jitter so the N concurrent retriers don't fire in lockstep.
            cooldown += random.random() * 1.5
            logger.warning(
                "429 from %s, sleeping %.1fs (attempt %d/%d)",
                url,
                cooldown,
                attempt,
                max_attempts,
            )
            await asyncio.sleep(cooldown)
            continue

        if r.status_code >= 500:
            cooldown = min(2 ** attempt, 30.0) + random.random()
            logger.warning(
                "%d from %s, sleeping %.1fs (attempt %d/%d)",
                r.status_code,
                url,
                cooldown,
                attempt,
                max_attempts,
            )
            await asyncio.sleep(cooldown)
            continue

        if r.status_code in (401, 403, 404, 410):
            return None, r

        r.raise_for_status()
        return r.json(), r

    # Exhausted retries — treat as soft failure so one bad repo cannot kill the
    # whole org pass. The caller logs and continues.
    if last_response is not None:
        raise RateLimited(
            f"giving up on {url} after {max_attempts} attempts "
            f"(last status {last_response.status_code})",
            retry_after=None,
        )
    raise RateLimited(f"giving up on {url} after {max_attempts} attempts", None)


async def list_models_for_org(
    client: httpx.AsyncClient, org: OrgConfig
) -> list[dict[str, Any]]:
    """Page through ``/api/models?author={org}``. HF returns at most ``PER_PAGE`` items per page.

    HF's default ordering is ``trendingScore`` (recency-weighted). For our use
    case "top-N by downloads" is the correct ranking — otherwise high-download
    but low-trending entries (e.g. ``mlx-community/Qwen3.5-4B-MLX-4bit`` with
    25k downloads) silently drop out of the catalog when ``max_repos*2`` cap
    is hit on the first few pages. We pass ``sort=downloads&direction=-1`` so
    pagination walks the popularity-sorted list directly.

    HF supports cursor-style pagination via ``Link: <...>; rel="next"`` but the
    field-name has shifted in the past. We fall back to ``cursor`` query-param
    pagination by reading ``Link`` if present, otherwise stopping at the first
    short page.
    """
    out: list[dict[str, Any]] = []
    next_url: str | None = f"{HF_API}/models"
    params: dict[str, Any] | None = {
        "author": org.name,
        "full": "true",
        "limit": PER_PAGE,
        "sort": "downloads",
        "direction": -1,
    }
    pages = 0
    while next_url is not None:
        data, response = await get_json(client, next_url, params=params)
        if data is None:
            logger.warning("listing %s returned HTTP %s", org.name, response.status_code)
            break
        if not isinstance(data, list):
            logger.warning("listing %s returned non-list payload", org.name)
            break
        out.extend(data)
        pages += 1

        link_header = response.headers.get("Link") or response.headers.get("link") or ""
        next_url = _extract_next_link(link_header)
        params = None  # next_url is already absolute and pre-encoded

        if len(data) < PER_PAGE:
            break
        if org.max_repos is not None and len(out) >= org.max_repos * 2:
            # Hard guard: stop fetching once we comfortably exceed the cap;
            # the actual top-N truncation happens after sorting.
            break
        if pages >= 50:
            logger.warning("listing %s hit page guard (50 pages)", org.name)
            break
    logger.info("listed %d repos for %s in %d pages", len(out), org.name, pages)
    return out


def _extract_next_link(link_header: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if not section:
            continue
        if 'rel="next"' not in section and "rel=next" not in section:
            continue
        match = re.match(r"<([^>]+)>", section)
        if match:
            return match.group(1)
    return None


def first_pass_filter(repos: list[dict[str, Any]], org: OrgConfig) -> list[dict[str, Any]]:
    """Cheap structural filter applied before doing per-repo detail fetches."""
    out: list[dict[str, Any]] = []
    for r in repos:
        if r.get("private") or r.get("disabled"):
            continue
        if (downloads := int(r.get("downloads") or 0)) < org.min_downloads:
            continue
        tags = {str(t).lower() for t in (r.get("tags") or [])}
        if org.tags_required and not all(t.lower() in tags for t in org.tags_required):
            continue
        if org.library_required and r.get("library_name") != org.library_required:
            continue
        has_gguf_hint = (
            "gguf" in tags
            or "gguf" in (r.get("modelId") or r.get("id") or "").lower()
            or r.get("library_name") == "gguf"
        )
        has_mlx_hint = (
            r.get("library_name") == "mlx"
            or any(t in MLX_TAGS for t in tags)
            or "mlx" in (r.get("modelId") or r.get("id") or "").lower()
        )
        # Drop pure transformers / pytorch repos with no GGUF / MLX surface.
        # We rely on tags here -- the detail fetch will confirm via siblings.
        if not (has_gguf_hint or has_mlx_hint):
            continue
        r["_downloads_int"] = downloads
        out.append(r)
    out.sort(key=lambda r: r["_downloads_int"], reverse=True)
    cap = org.max_repos or DEFAULT_PER_ORG_CAP
    return out[:cap]


async def fetch_repo_detail(
    client: httpx.AsyncClient,
    repo_id: str,
    *,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with sem:
        url = f"{HF_API}/models/{repo_id}"
        params = {"blobs": "true", "files_metadata": "true"}
        try:
            data, response = await get_json(client, url, params=params)
        except httpx.HTTPError as exc:
            logger.warning("detail %s failed: %s", repo_id, exc)
            return None
        if data is None:
            logger.info("detail %s returned HTTP %s, skipping", repo_id, response.status_code)
            return None
        return data


def format_size(n: int | None) -> str:
    if not n or n <= 0:
        return "Unknown size"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.1f} GB"


def sanitize_model_id(value: str) -> str:
    """Mirrors ``sanitizeModelId`` in ``web-app/src/lib/utils.ts``.

    The client uses the resulting id as a stable handle for the local
    download path, so we MUST produce the same string here.
    """
    # Replace whitespace with dashes, drop characters outside the allow-set.
    value = re.sub(r"\s+", "-", value)
    return re.sub(r"[^a-zA-Z0-9\-_./]", "", value)


def normalize_tags(repo: dict[str, Any], siblings: list[dict[str, Any]]) -> list[str]:
    out: set[str] = set()
    for t in repo.get("tags") or []:
        if isinstance(t, str):
            out.add(t.lower())
    if isinstance(repo.get("library_name"), str):
        out.add(repo["library_name"].lower())
    if isinstance(repo.get("pipeline_tag"), str):
        out.add(repo["pipeline_tag"].lower())
    repo_lower = (repo.get("modelId") or repo.get("id") or "").lower()
    for token in QUANT_CODE_RE.findall(repo_lower):
        out.add(token.lower())
    for sib in siblings:
        rfilename = sib.get("rfilename", "")
        for token in QUANT_CODE_RE.findall(rfilename.lower()):
            out.add(token.lower())
    return sorted(out)


def build_catalog_model(repo: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a HF detail-fetch payload into the client's ``CatalogModel`` shape.

    Returns ``None`` if the repo carries no consumable model files
    (no GGUF, no safetensors).
    """
    model_id: str = repo.get("modelId") or repo.get("id") or ""
    if "/" not in model_id:
        return None
    author = repo.get("author") or model_id.split("/", 1)[0]

    siblings = repo.get("siblings") or []

    def _resolve(filename: str) -> str:
        return f"{HF_RESOLVE}/{model_id}/resolve/main/{filename}"

    gguf_files = [s for s in siblings if s.get("rfilename", "").lower().endswith(GGUF_SUFFIX)]
    regular = [s for s in gguf_files if "mmproj" not in s["rfilename"].lower()]
    mmproj = [s for s in gguf_files if "mmproj" in s["rfilename"].lower()]
    safetensors = [
        s for s in siblings if s.get("rfilename", "").lower().endswith(SAFETENSORS_SUFFIX)
    ]

    if not regular and not safetensors:
        return None

    def _size(s: dict[str, Any]) -> int | None:
        if s.get("size"):
            return int(s["size"])
        lfs = s.get("lfs") or {}
        if lfs.get("size"):
            return int(lfs["size"])
        return None

    quants = [
        {
            "model_id": f"{author}/{sanitize_model_id(s['rfilename'].rsplit('.', 1)[0])}",
            "path": _resolve(s["rfilename"]),
            "file_size": format_size(_size(s)),
        }
        for s in regular
    ]
    mmproj_models = [
        {
            "model_id": sanitize_model_id(s["rfilename"].rsplit(".", 1)[0]),
            "path": _resolve(s["rfilename"]),
            "file_size": format_size(_size(s)),
        }
        for s in mmproj
    ]
    safetensors_files: list[dict[str, Any]] = []
    for s in safetensors:
        entry: dict[str, Any] = {
            "model_id": sanitize_model_id(s["rfilename"].rsplit(".", 1)[0]),
            "path": _resolve(s["rfilename"]),
            "file_size": format_size(_size(s)),
        }
        lfs = s.get("lfs") or {}
        if isinstance(lfs.get("sha256"), str):
            entry["sha256"] = lfs["sha256"]
        safetensors_files.append(entry)

    tags = [str(t) for t in (repo.get("tags") or []) if isinstance(t, str)]
    description = f"**Tags**: {', '.join(tags)}" if tags else "**Tags**: "

    is_mlx = repo.get("library_name") == "mlx" or any(t.lower() == "mlx" for t in tags)

    model: dict[str, Any] = {
        "model_name": model_id,
        "developer": author,
        "downloads": int(repo.get("downloads") or 0),
        "likes": int(repo.get("likes") or 0),
        "description": description,
        "num_quants": len(quants),
        "quants": quants,
        "num_mmproj": len(mmproj_models),
        "mmproj_models": mmproj_models,
        "num_safetensors": len(safetensors_files),
        "safetensors_files": safetensors_files,
        "is_mlx": bool(is_mlx),
        "tags_normalized": normalize_tags(repo, siblings),
        "readme": f"{HF_RESOLVE}/{model_id}/resolve/main/README.md",
    }
    if repo.get("library_name"):
        model["library_name"] = repo["library_name"]
    if repo.get("createdAt"):
        model["created_at"] = repo["createdAt"]
    if repo.get("lastModified") or repo.get("last_modified"):
        model["last_modified"] = repo.get("lastModified") or repo.get("last_modified")
    return model


async def scrape_org(
    client: httpx.AsyncClient,
    org: OrgConfig,
    *,
    detail_sem: asyncio.Semaphore,
) -> tuple[list[dict[str, Any]], int]:
    """Returns ``(catalog_entries, skipped_count)``.

    Soft-fails on listing-level :class:`RateLimited` so one bursting org
    does not lose results for the other 20+. The next cron run will pick
    the missing org back up. ``run()`` additionally wraps this call in
    ``asyncio.gather(return_exceptions=True)`` for symmetry on the detail
    path.
    """
    try:
        listing = await list_models_for_org(client, org)
    except RateLimited as exc:
        logger.warning(
            "%s -- listing rate-limited, skipping this run: %s", org.name, exc
        )
        return [], 0
    candidates = first_pass_filter(listing, org)
    logger.info(
        "%s -- %d candidates after first-pass filter (from %d total)",
        org.name,
        len(candidates),
        len(listing),
    )

    tasks = [
        fetch_repo_detail(
            client,
            c.get("modelId") or c.get("id") or "",
            sem=detail_sem,
        )
        for c in candidates
    ]
    details = await asyncio.gather(*tasks)
    out: list[dict[str, Any]] = []
    skipped = 0
    for detail in details:
        if detail is None:
            skipped += 1
            continue
        model = build_catalog_model(detail)
        if model is None:
            skipped += 1
            continue
        out.append(model)
    return out, skipped


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    orgs = load_orgs(Path(args.config))
    if args.orgs:
        wanted = {o.strip() for o in args.orgs.split(",") if o.strip()}
        orgs = [o for o in orgs if o.name in wanted]
        if not orgs:
            logger.error("--orgs %r matched no entries in %s", args.orgs, args.config)
            return 2

    active_orgs = [o for o in orgs if o.active]
    logger.info(
        "scraping %d active orgs (of %d total) -> %s",
        len(active_orgs),
        len(orgs),
        out_dir,
    )

    # Surface HF_TOKEN status without ever leaking the secret. CI logs will
    # show "HF_TOKEN: present (len=37)" or "HF_TOKEN: ABSENT (anonymous)";
    # the difference between Free (1000/5min) and Anonymous (500/5min) is
    # significant enough that we want to be 100% sure the secret made it.
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        logger.info(
            "HF_TOKEN: present (len=%d, prefix=%s***) — authenticated quota active",
            len(token),
            token[:3] if len(token) >= 3 else "***",
        )
    else:
        logger.warning(
            "HF_TOKEN: ABSENT — falling back to anonymous quota (500 / 5min per IP)"
        )

    # Global token bucket. ``--rate-limit`` overrides the default. Apply to
    # the module-level singleton so ``get_json`` (which runs in many parallel
    # tasks) shares the same budget without explicit threading.
    global _rate_limiter
    _rate_limiter = AsyncRateLimiter(rate_rps=max(0.1, args.rate_limit))
    logger.info(
        "rate limit: %.2f req/sec (min interval %.0f ms)",
        _rate_limiter.rate_rps,
        _rate_limiter._min_interval * 1000,
    )

    list_sem = asyncio.Semaphore(DEFAULT_LIST_CONCURRENCY)
    detail_sem = asyncio.Semaphore(max(1, args.concurrency))
    stats = ScrapeStats()
    by_id: dict[str, dict[str, Any]] = {}
    started = time.monotonic()

    async with httpx.AsyncClient(headers=http_headers(), timeout=REQUEST_TIMEOUT) as client:

        async def _wrap(org: OrgConfig) -> tuple[OrgConfig, list[dict[str, Any]], int]:
            async with list_sem:
                models, skipped = await scrape_org(client, org, detail_sem=detail_sem)
                return org, models, skipped

        # ``return_exceptions=True`` so a single org's terminal failure
        # (network blip, unhandled 429 storm, etc.) does not discard the
        # 20+ successful orgs already aggregated. Failed orgs are logged
        # and recorded as ``by_org_skipped`` with the full candidate count.
        raw_results = await asyncio.gather(
            *(_wrap(o) for o in active_orgs), return_exceptions=True
        )

    results: list[tuple[OrgConfig, list[dict[str, Any]], int]] = []
    for org, item in zip(active_orgs, raw_results, strict=True):
        if isinstance(item, BaseException):
            logger.error(
                "%s -- scrape failed terminally: %s: %s",
                org.name,
                type(item).__name__,
                item,
            )
            stats.errors.append(f"{org.name}: {type(item).__name__}: {item}")
            results.append((org, [], 0))
            continue
        results.append(item)

    for org, models, skipped in results:
        kept = 0
        for m in models:
            mid = m["model_name"]
            existing = by_id.get(mid)
            # Dedupe across orgs: prefer the entry from the listing org if
            # the repo's author matches; otherwise keep the higher-download.
            if existing is None:
                by_id[mid] = m
                kept += 1
            elif (m.get("downloads") or 0) > (existing.get("downloads") or 0):
                by_id[mid] = m
        stats.by_org[org.name] = kept
        stats.by_org_skipped[org.name] = skipped
        if args.limit and len(by_id) >= args.limit:
            logger.warning("hit --limit %d, stopping aggregation after %s", args.limit, org.name)
            break

    models_sorted = sorted(
        by_id.values(),
        key=lambda m: (-int(m.get("downloads") or 0), m["model_name"].lower()),
    )
    stats.total_models = len(models_sorted)
    stats.mlx_models = sum(1 for m in models_sorted if m.get("is_mlx"))
    stats.gguf_models = sum(1 for m in models_sorted if (m.get("num_quants") or 0) > 0)
    stats.elapsed_seconds = round(time.monotonic() - started, 2)

    # Guard against publishing an empty catalog: if every org failed
    # (e.g. HF outage, secrets misconfiguration) we'd otherwise overwrite
    # ``dist/catalog.json.gz`` with a 0-model snapshot and break every
    # client until the next run. Exit 1 so cron.yml stops before commit.
    if stats.total_models == 0:
        logger.error(
            "aggregation produced 0 models across %d active orgs; refusing to publish "
            "an empty catalog. Errors: %s",
            len(active_orgs),
            stats.errors or "(none recorded — check listing/detail logs above)",
        )
        return 1

    catalog = {
        "$schema": "../config/schema.catalog.json",
        "manifest_version": CATALOG_MANIFEST_VERSION,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "orgs": [o.name for o in active_orgs],
        "stats": {
            "total_models": stats.total_models,
            "by_org": stats.by_org,
            "by_org_skipped": stats.by_org_skipped,
            "mlx_models": stats.mlx_models,
            "gguf_models": stats.gguf_models,
            "elapsed_seconds": stats.elapsed_seconds,
        },
        "models": models_sorted,
    }

    catalog_path = out_dir / "catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    stats_path = out_dir / "stats.json"
    stats_path.write_text(
        json.dumps(catalog["stats"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "wrote %d models -> %s (%.1f MB) in %.1fs",
        stats.total_models,
        catalog_path,
        catalog_path.stat().st_size / 1024 / 1024,
        stats.elapsed_seconds,
    )
    if args.dry_run:
        logger.info("dry-run mode: skipping upload (artefacts left in %s)", out_dir)
    return 0


def _now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_ORGS_PATH),
        help="path to orgs.json (default: ../config/orgs.json)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR),
        help="output directory for catalog.json + stats.json (default: ../out)",
    )
    parser.add_argument(
        "--orgs",
        default="",
        help="comma-separated subset of org names to scrape (default: all active)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write artefacts locally but do not upload (no-op when no uploader is wired)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="bypass any future ETag short-circuit cache (placeholder for now)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap total catalog size after dedupe (0 = no cap, default)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_DETAIL_CONCURRENCY,
        help=(
            "Per-org detail fetch concurrency (default: "
            f"{DEFAULT_DETAIL_CONCURRENCY}). Raise only when HF_TOKEN is set."
        ),
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=float(os.environ.get("HF_RATE_LIMIT_RPS", DEFAULT_RATE_LIMIT_RPS)),
        help=(
            "Global HF request budget in req/sec (default: "
            f"{DEFAULT_RATE_LIMIT_RPS}). Free token sustains ~3.3/sec, PRO "
            "~8.3/sec — leave headroom against the 5-min fixed-window reset."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
