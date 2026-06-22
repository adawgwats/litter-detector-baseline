"""Hard-negatives corpus ingestion.

Hard negatives are images of visually-confusing non-litter objects (leaves,
mulch, shadows, painted asphalt) — the model must learn to produce NO
detections on them. Per docs/contributor_assist_goal.md, leaves are
EXPLICITLY rejected as litter in V1, and ``Hard-negative precision`` is a
gating success metric (FPR <= 5%).

This module reads two artifacts produced by
``scripts/build_hard_negatives_seed.py``:

  * ``configs/hard_negatives_seed.txt``        — one image URL per line
  * ``configs/hard_negatives_licenses.json``   — per-URL license metadata

It downloads each URL, hashes the bytes, and stores under
``<key_prefix>/photos/<aa>/<bb>/<sha256>.<ext>`` mirroring openlittermap.py
and taco.py. Metadata for each image goes to
``<key_prefix>/_metadata/<sha256>.json`` with the original URL, license,
and the synthetic ``no_litter`` leaf label.

The synthetic ``no_litter`` leaf is registered in
:mod:`litter_detector_baseline.ingest.crosswalk` (SYNTHETIC_LEAVES) so the
training pipeline can treat hard-negatives as a first-class label without
polluting the OLM leaf taxonomy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import requests

from .crosswalk import SYNTHETIC_LEAVES
from .storage import Storage, content_addressed_key

log = logging.getLogger(__name__)

# Repo-relative defaults
_PKG_ROOT = Path(__file__).resolve().parent.parent  # .../litter_detector_baseline
_REPO_ROOT = _PKG_ROOT.parent.parent
DEFAULT_SEED_PATH = _REPO_ROOT / "configs" / "hard_negatives_seed.txt"
DEFAULT_LICENSES_PATH = _REPO_ROOT / "configs" / "hard_negatives_licenses.json"

DEFAULT_USER_AGENT = (
    "litter-detector-baseline/0.1 hard-negatives-ingest "
    "(+https://github.com/adawgwats/litter-detector-baseline; "
    "contact: adawgwats@gmail.com)"
)

NO_LITTER_LEAF = "no_litter"
assert NO_LITTER_LEAF in SYNTHETIC_LEAVES  # Catch any rename in crosswalk.py

IMAGE_DOWNLOAD_INTERVAL_SEC = 0.3
MAX_RETRIES = 6
INITIAL_BACKOFF_SEC = 2.0


@dataclass
class HardNegativeRecord:
    """One hard-negative image as recorded in the per-run manifest."""

    url: str
    sha256: str
    image_key: str
    license_short: str
    license_url: str
    attribution_html: str
    category_hint: str
    olm_leaf: str = NO_LITTER_LEAF


@dataclass
class HardNegativesIngestManifest:
    source: str = "WikimediaCommons+curated"
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    image_count: int = 0
    images: list = field(default_factory=list)
    license_histogram: dict = field(default_factory=dict)
    skipped_count: int = 0
    failed_count: int = 0
    attribution_note: str = (
        "Hard-negatives sourced from Wikimedia Commons under CC0 / Public "
        "Domain / CC-BY / CC-BY-SA licenses. Per-image attribution is in "
        "each `images[].attribution_html`. Downstream model cards must "
        "preserve attribution per each image's license."
    )


# ─── helpers ─────────────────────────────────────────────────────────────


def load_seed_urls(seed_path: Path = DEFAULT_SEED_PATH) -> list[str]:
    """Read the seed manifest (one URL per line, '#' comments allowed)."""
    if not seed_path.exists():
        raise FileNotFoundError(
            f"hard-negatives seed not found at {seed_path}. Run "
            "`python scripts/build_hard_negatives_seed.py` first."
        )
    urls = []
    for line in seed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def load_license_index(licenses_path: Path = DEFAULT_LICENSES_PATH) -> dict[str, dict]:
    """Return a mapping url -> license-metadata record."""
    if not licenses_path.exists():
        raise FileNotFoundError(
            f"hard-negatives licenses manifest not found at {licenses_path}. "
            "Run `python scripts/build_hard_negatives_seed.py` first."
        )
    data = json.loads(licenses_path.read_text(encoding="utf-8"))
    return {r["url"]: r for r in data.get("accepted", [])}


def _backoff_get_bytes(url: str, *, user_agent: str) -> bytes:
    backoff = INITIAL_BACKOFF_SEC
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=120)
        if resp.status_code in (429, 503):
            sleep_for = float(resp.headers.get("Retry-After", backoff))
            log.warning(
                "HTTP %d on %s (attempt %d/%d); backing off %.1fs",
                resp.status_code, url, attempt + 1, MAX_RETRIES, sleep_for,
            )
            time.sleep(sleep_for)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.content
    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries on {url}")


# ─── enumeration (no I/O) ────────────────────────────────────────────────


def iter_seed_records(
    seed_path: Path = DEFAULT_SEED_PATH,
    licenses_path: Path = DEFAULT_LICENSES_PATH,
) -> Iterator[dict]:
    """Yield per-URL pre-download records (url + license + category_hint)."""
    license_idx = load_license_index(licenses_path)
    for url in load_seed_urls(seed_path):
        meta = license_idx.get(url, {})
        yield {
            "url": url,
            "license_short": meta.get("license_short", ""),
            "license_url": meta.get("license_url", ""),
            "attribution_html": meta.get("attribution_html", ""),
            "category_hint": meta.get("category_hint", ""),
            "olm_leaf": NO_LITTER_LEAF,
        }


# ─── top-level ingest ────────────────────────────────────────────────────


def _process_one_record(
    rec: dict,
    *,
    storage: Storage,
    user_agent: str,
    key_prefix: str,
    skip_existing: bool,
) -> dict:
    """Download + store one hard-negative image. Returns per-record status
    dict for the caller to aggregate. Safe to invoke concurrently."""
    try:
        data = _backoff_get_bytes(rec["url"], user_agent=user_agent)
    except Exception as exc:
        return {"status": "download_failed", "url": rec["url"],
                "error": f"{type(exc).__name__}: {exc}"}

    sha256 = hashlib.sha256(data).hexdigest()
    ext = Path(rec["url"].split("?")[0]).suffix.lstrip(".").lower() or "jpg"
    image_key = content_addressed_key(prefix=key_prefix, data=data, ext=ext)
    metadata_key = f"{key_prefix.rstrip('/')}/_metadata/{sha256}.json"

    if skip_existing and storage.exists(metadata_key):
        return {"status": "skipped", "url": rec["url"], "image_key": image_key}

    if not (skip_existing and storage.exists(image_key)):
        storage.put_bytes(image_key, data, content_type=f"image/{ext}")

    record = HardNegativeRecord(
        url=rec["url"],
        sha256=sha256,
        image_key=image_key,
        license_short=rec["license_short"],
        license_url=rec["license_url"],
        attribution_html=rec["attribution_html"],
        category_hint=rec["category_hint"],
    )
    storage.put_bytes(
        metadata_key,
        json.dumps(asdict(record)).encode("utf-8"),
        "application/json",
    )
    return {"status": "ingested", "url": rec["url"], "record": asdict(record)}


def ingest_hard_negatives(
    *,
    storage: Optional[Storage],
    seed_path: Path = DEFAULT_SEED_PATH,
    licenses_path: Path = DEFAULT_LICENSES_PATH,
    user_agent: str = DEFAULT_USER_AGENT,
    key_prefix: str = "hard_negatives/photos",
    manifest_path: Optional[Path] = None,
    skip_existing: bool = True,
    max_photos: Optional[int] = None,
    dry_run: bool = False,
    workers: int = 8,
) -> HardNegativesIngestManifest:
    """Download seed URLs, hash + store, write the per-run manifest.

    When ``dry_run=True``, no downloads or storage writes happen; the
    manifest just enumerates the URLs + licenses for inspection.
    """
    if not dry_run and storage is None:
        raise ValueError("storage is required when dry_run=False")

    manifest = HardNegativesIngestManifest()
    records = list(iter_seed_records(seed_path, licenses_path))
    if max_photos is not None:
        records = records[:max_photos]

    # License histogram is independent of upload outcome — aggregate once.
    for rec in records:
        manifest.license_histogram[rec["license_short"]] = (
            manifest.license_histogram.get(rec["license_short"], 0) + 1
        )

    if dry_run:
        for rec in records:
            manifest.image_count += 1
            manifest.images.append({**rec, "dry_run": True})
    else:
        log.info("dispatching ingest of %d images to %d workers", len(records), workers)
        counter = {"done": 0, "ingested": 0, "skipped": 0, "failed": 0}
        counter_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_record,
                    rec,
                    storage=storage,
                    user_agent=user_agent,
                    key_prefix=key_prefix,
                    skip_existing=skip_existing,
                ): rec
                for rec in records
            }
            for fut in as_completed(futures):
                result = fut.result()
                with counter_lock:
                    counter["done"] += 1
                    if result["status"] == "ingested":
                        counter["ingested"] += 1
                        manifest.image_count += 1
                        manifest.images.append(result["record"])
                    elif result["status"] == "skipped":
                        counter["skipped"] += 1
                        manifest.skipped_count += 1
                    else:
                        counter["failed"] += 1
                        manifest.failed_count += 1
                    done = counter["done"]
                if done % 50 == 0 or done == len(records):
                    log.info(
                        "[%d/%d] ingested=%d skipped=%d failed=%d",
                        done, len(records),
                        counter["ingested"], counter["skipped"], counter["failed"],
                    )

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
        log.info("wrote manifest to %s", manifest_path)

    return manifest
