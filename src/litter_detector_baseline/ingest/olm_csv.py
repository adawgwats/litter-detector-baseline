"""OpenLitterMap "number-based" CSV export ingestion.

Consumes the CSV produced by POST /api/download for a country / state /
city. Parses the per-photo tag counts back into OLM-leaf format (the same
``<category>.<object>`` dotted form our crosswalk uses), then for each
photo:

  1. Fetches a signed image URL via /api/photos/{id}/signed-url
     (Referer-locked endpoint — we set Referer: https://openlittermap.com/)
  2. Downloads image bytes from the signed S3 URL
  3. PUTs to R2 at <key_prefix>/<aa>/<bb>/<sha>.jpg
  4. PUTs a per-photo metadata JSON with the parsed tags

CSV schema (variable-width across regions):

  Cols 0-9:  metadata (id, verification, phone, date_taken, date_uploaded,
             lat, lon, picked_up, address, total_tags)
  Cols 10+:  alternating uppercase CATEGORY markers (always 0-valued, used
             only as section delimiters) followed by lowercase object
             sub-columns whose values are per-object COUNTS. Categories
             present depend on the region. After the litter categories,
             MATERIALS, TYPES, brands, custom_tag_1..3 sections follow.

The parser discovers categories dynamically by matching column headers
against the known OLM categories from the taxonomy snapshot, so adding new
categories in OLM does not require code changes here.

Image fetch politeness:
  - OLM's docs/olm-auth-howto.md caps us at ~1 req/sec metadata, ~3
    concurrent on image downloads. The signed-url endpoint is metadata-
    rate; the S3 redirect download is hosted on AWS and can take more.
  - We default to workers=4 (matches the OLM "3 concurrent images" guidance
    with a little headroom for the signed-url fetch which is fast).
"""
from __future__ import annotations

import csv
import io
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
from PIL import Image

# Register iPhone HEIC support. Many OLM uploads are .HEIC from iOS users
# and the default Pillow build can't decode them.
try:
    import pillow_heif  # type: ignore[import-untyped]

    pillow_heif.register_heif_opener()
except ImportError:
    pass

from .crosswalk import valid_olm_leaves
from .olm_auth import OLM_BASE, OlmSession, load_credentials
from .storage import Storage, content_addressed_key

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "litter-detector-baseline/0.1 OLM-CSV-ingest "
    "(+adawgwats@gmail.com)"
)

# OLM's /api/photos/{id}/signed-url is hard-capped at 60 req/min (verified
# 2026-05-22 via direct probe — see docs/olm-auth-howto.md). Going above
# this trips a 429 with Retry-After ~9-23 sec. We rate-limit at slightly
# under 60/min to leave headroom.
OLM_SIGNED_URL_RATE_PER_SEC = 0.95  # 57/min

# Pre-resize OLM images during ingest. OLM stores ~10 MB contributor-camera
# originals; we don't need that resolution for YOLO11n (imgsz=640) and the
# upgrade path to imgsz=1280 fits within this. Cuts R2 storage ~20x and
# wall-time ~3x.
RESIZE_LONGEST_EDGE_PX = 1280
RESIZE_JPEG_QUALITY = 90

# The actual image bytes live on OLM's public S3 bucket — bare URLs work
# without the signature query string (verified 2026-05-22). So once we have
# a signed URL we just strip the query and download the bare path, which is
# NOT rate-limited by OLM (it's served direct from S3). We can fan out
# downloads + R2 uploads at high concurrency.
MAX_RETRIES = 6
INITIAL_BACKOFF_SEC = 2.0


class _RateLimiter:
    """Token-bucket-ish limiter: enforces a minimum interval between
    acquire() calls, blocking when called too fast. Thread-safe."""

    def __init__(self, rate_per_sec: float) -> None:
        self._min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._last + self._min_interval - now)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

# Static set of OLM litter categories that show up as uppercase CSV markers
# in the "number-based" layout. Source: configs/taxonomies/olm_leaves.json.
# Anything not in this set is treated as a non-litter section (materials,
# types, brands, custom_tags).
LITTER_CATEGORIES = frozenset({
    "alcohol", "art", "civic", "coffee", "dumping", "electronics",
    "food", "industrial", "marine", "medical", "other", "pets",
    "sanitary", "smoking", "softdrinks", "vehicles",
})

# Non-litter section headers — when we hit one of these we stop emitting
# leaves and switch to recording materials/brands/types/custom_tags.
NON_LITTER_SECTIONS = frozenset({"types", "materials", "brands"})


@dataclass
class OlmCsvRecord:
    """One photo's parsed-and-resolved tag bundle from the CSV."""

    photo_id: int
    verification: int
    lat: float
    lon: float
    date_taken: Optional[str]
    date_uploaded: Optional[str]
    picked_up: bool
    address: Optional[str]
    total_tags: int

    # leaves: list of (olm_leaf_label, count), e.g. ("smoking.butts", 3)
    leaves: list[tuple[str, int]] = field(default_factory=list)
    # materials: list of (material_key, count)
    materials: list[tuple[str, int]] = field(default_factory=list)
    # types: list of (type_key, count)
    types: list[tuple[str, int]] = field(default_factory=list)
    # brand and custom-tag rows in the CSV are free-text; preserve verbatim
    brands: Optional[str] = None
    custom_tags: list[str] = field(default_factory=list)


@dataclass
class OlmCsvIngestManifest:
    source: str = "OpenLitterMap"
    source_license: str = "CC-BY-SA-4.0"
    source_url: str = "https://openlittermap.com"
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    csv_path: Optional[str] = None
    image_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    olm_leaf_histogram: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    attribution_note: str = (
        "Photos under CC-BY-SA-4.0 by individual OpenLitterMap contributors. "
        "Derived models trained on this corpus must credit OpenLitterMap and "
        "preserve attribution per CC-BY-SA-4.0 terms."
    )


# ─── CSV parser ──────────────────────────────────────────────────────────


def _column_sections(header: list[str]) -> list[tuple[str, str, int]]:
    """Walk the header row and produce a list of (section, sub_name, col_idx)
    for every column after the fixed metadata prefix.

    `section` is the lowercase category for litter columns (e.g. "smoking"),
    or one of "materials" / "types" / "brands" / "custom_tags" for the
    non-litter sections.
    """
    out: list[tuple[str, str, int]] = []
    current_section: str = ""
    for i, raw in enumerate(header):
        if i < 10:
            continue  # fixed metadata prefix
        name = raw.strip()
        lname = name.lower()
        if lname in LITTER_CATEGORIES:
            current_section = lname
            continue  # the category header column itself is always empty
        if lname in NON_LITTER_SECTIONS:
            current_section = lname
            continue
        if lname.startswith("custom_tag"):
            current_section = "custom_tags"
            out.append((current_section, name, i))
            continue
        if not current_section:
            continue  # column before any section marker — shouldn't happen
        out.append((current_section, name, i))
    return out


def parse_csv(csv_path: Path) -> Iterator[OlmCsvRecord]:
    """Yield :class:`OlmCsvRecord` for every photo row in the OLM CSV."""
    valid = valid_olm_leaves()
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sections = _column_sections(header)
        for row in reader:
            if not row or not row[0]:
                continue
            try:
                rec = OlmCsvRecord(
                    photo_id=int(row[0]),
                    verification=int(row[1] or 0),
                    lat=float(row[5]),
                    lon=float(row[6]),
                    date_taken=row[3] or None,
                    date_uploaded=row[4] or None,
                    picked_up=row[7].lower() in ("true", "1", "yes"),
                    address=row[8] or None,
                    total_tags=int(row[9] or 0),
                )
            except (ValueError, IndexError):
                continue
            for section, sub_name, col_idx in sections:
                if col_idx >= len(row):
                    continue
                cell = (row[col_idx] or "").strip()
                if not cell:
                    continue
                if section in LITTER_CATEGORIES:
                    try:
                        cnt = int(cell)
                    except ValueError:
                        continue
                    if cnt <= 0:
                        continue
                    leaf = f"{section}.{sub_name}"
                    if leaf in valid:
                        rec.leaves.append((leaf, cnt))
                elif section == "materials":
                    try:
                        cnt = int(cell)
                    except ValueError:
                        continue
                    if cnt > 0:
                        rec.materials.append((sub_name, cnt))
                elif section == "types":
                    try:
                        cnt = int(cell)
                    except ValueError:
                        continue
                    if cnt > 0:
                        rec.types.append((sub_name, cnt))
                elif section == "brands":
                    rec.brands = cell
                elif section == "custom_tags":
                    rec.custom_tags.append(cell)
            yield rec


# ─── per-photo worker (signed URL → download → upload) ────────────────────


def _fetch_image_and_upload(
    rec: OlmCsvRecord,
    *,
    session: OlmSession,
    storage: Storage,
    user_agent: str,
    key_prefix: str,
    skip_existing: bool,
    rate_limiter: "_RateLimiter",
) -> dict:
    """Process one CSV record into R2. Returns per-photo status dict.

    Calls the rate limiter BEFORE hitting /signed-url so all workers stay
    under OLM's 60/min cap collectively. After the signed-url returns, the
    image download + R2 upload are NOT rate-limited (S3 is public + R2 is
    our own).
    """
    metadata_key = f"{key_prefix.rstrip('/')}/_metadata/{rec.photo_id}.json"
    if skip_existing and storage.exists(metadata_key):
        return {"status": "skipped", "photo_id": rec.photo_id,
                "record": {**asdict(rec), "skipped": True}}

    # 1. Signed URL (rate-limited to stay under OLM's 60/min cap)
    rate_limiter.acquire()
    try:
        r = session.get(
            f"{OLM_BASE}/api/photos/{rec.photo_id}/signed-url",
            headers={"Referer": "https://openlittermap.com/"},
        )
        r.raise_for_status()
        url = r.json().get("url")
    except Exception as exc:
        return {"status": "signed_url_failed", "photo_id": rec.photo_id,
                "error": f"{type(exc).__name__}: {exc}"}
    if not url:
        return {"status": "signed_url_missing", "photo_id": rec.photo_id}

    # 2. Drop the signature query — OLM's S3 bucket is publicly readable so
    #    the bare path works and isn't rate-limited (verified 2026-05-22).
    bare_url = url.split("?", 1)[0]

    # 3. Download bytes from S3 (the original ~10 MB contributor photo)
    try:
        img_resp = requests.get(bare_url, headers={"User-Agent": user_agent}, timeout=120)
        img_resp.raise_for_status()
    except Exception as exc:
        return {"status": "download_failed", "photo_id": rec.photo_id,
                "error": f"{type(exc).__name__}: {exc}"}

    # 4. Resize to RESIZE_LONGEST_EDGE_PX (preserves aspect ratio) and
    #    re-encode as JPEG. Cuts upload bytes ~20x. If decode fails (rare
    #    format) fall back to uploading the original payload.
    original_bytes = img_resp.content
    original_len = len(original_bytes)
    try:
        img = Image.open(io.BytesIO(original_bytes))
        # convert handles HEIC, RGBA, palette, CMYK, etc.
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > RESIZE_LONGEST_EDGE_PX:
            scale = RESIZE_LONGEST_EDGE_PX / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=RESIZE_JPEG_QUALITY, optimize=True)
        data = buf.getvalue()
        ext = "jpg"
    except Exception as exc:
        log.warning("resize failed for photo_id=%s (%s); uploading original", rec.photo_id, exc)
        data = original_bytes
        ctype = img_resp.headers.get("Content-Type", "")
        ext = "jpg" if "jpeg" in ctype or "jpg" in ctype else (
            "png" if "png" in ctype else "bin"
        )

    image_key = content_addressed_key(prefix=key_prefix, data=data, ext=ext)
    if not (skip_existing and storage.exists(image_key)):
        storage.put_bytes(image_key, data, content_type=f"image/{ext}")

    full_record = {
        **asdict(rec),
        "image_key": image_key,
        "s3_source_url": bare_url,
        "original_bytes": original_len,
        "stored_bytes": len(data),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    storage.put_bytes(
        metadata_key,
        json.dumps(full_record).encode("utf-8"),
        "application/json",
    )
    return {"status": "ingested", "photo_id": rec.photo_id, "record": full_record}


# ─── top-level ingest ────────────────────────────────────────────────────


def ingest_olm_csv(
    *,
    csv_path: Path,
    storage: Storage,
    user_agent: str = DEFAULT_USER_AGENT,
    key_prefix: str = "openlittermap/photos",
    manifest_path: Optional[Path] = None,
    skip_existing: bool = True,
    max_photos: Optional[int] = None,
    workers: int = 4,
    dry_run: bool = False,
) -> OlmCsvIngestManifest:
    """Crosswalk + (optionally) download every photo in an OLM CSV export.

    Requires a valid OLM session (handled internally via OlmSession +
    cookie cache so we don't hit /api/auth/login per call).
    """
    manifest = OlmCsvIngestManifest(csv_path=str(csv_path))
    records = list(parse_csv(csv_path))
    log.info("OLM CSV %s: %d photos", csv_path.name, len(records))
    if max_photos is not None:
        records = records[:max_photos]

    # Build the leaf histogram up-front (independent of upload outcome)
    for rec in records:
        for leaf, cnt in rec.leaves:
            manifest.olm_leaf_histogram[leaf] = (
                manifest.olm_leaf_histogram.get(leaf, 0) + cnt
            )

    if dry_run:
        for rec in records:
            manifest.image_count += 1
            manifest.images.append({**asdict(rec), "dry_run": True})
    else:
        log.info(
            "dispatching %d photos to %d workers (signed-url rate-limited at %.2f/s)",
            len(records), workers, OLM_SIGNED_URL_RATE_PER_SEC,
        )
        counter = {"done": 0, "ingested": 0, "skipped": 0, "failed": 0}
        counter_lock = threading.Lock()
        rate_limiter = _RateLimiter(OLM_SIGNED_URL_RATE_PER_SEC)
        with OlmSession(load_credentials()) as session, \
             ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _fetch_image_and_upload,
                    rec,
                    session=session,
                    storage=storage,
                    user_agent=user_agent,
                    key_prefix=key_prefix,
                    skip_existing=skip_existing,
                    rate_limiter=rate_limiter,
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
                        log.warning(
                            "photo_id=%s status=%s err=%s",
                            result.get("photo_id"), result["status"],
                            result.get("error"),
                        )
                    done = counter["done"]
                if done % 100 == 0 or done == len(records):
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
