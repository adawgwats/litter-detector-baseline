"""OpenLitterMap (CC-BY-SA-4.0) corpus ingestion.

Pulls photo metadata + images from openlittermap.com using the same
public endpoints their official QGIS plugin uses
(NaturalGIS/openlittermap, verified 2026-05-18). No auth required.

Endpoints used:
    GET https://openlittermap.com/global/points?zoom=18&bbox=<json>&year=<int>
        — per-photo metadata for a bounding box
    GET https://openlittermap.com/photos/{id}/signed-url
        — signed URL to image bytes
    GET https://openlittermap.com/api/tags/all
        — full tag taxonomy (200+ classes)

Politeness:
    - Self-throttles metadata requests to ~1 req/sec
    - Self-throttles image downloads to ~3 concurrent
    - Honors HTTP 429 / 503 with exponential backoff
    - Sends a real User-Agent identifying the caller — please customize via
      the ``user_agent=`` arg

Idempotent:
    Object keys in storage are content-addressable by sha256(image_bytes).
    Re-running the ingest over the same bbox is a near-zero-cost no-op
    (only the metadata calls re-execute; image bytes are skipped if the
    storage already has the hash).

Attribution preservation (CC-BY-SA-4.0 requirement):
    Every photo's ``username`` + ``team`` + permalink is preserved in the
    per-batch manifest. Downstream model cards must credit OpenLitterMap
    contributors per CC-BY-SA terms.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import requests

from .storage import Storage, content_addressed_key

log = logging.getLogger(__name__)

OLM_POINTS_URL = "https://openlittermap.com/global/points"
OLM_SIGNED_URL = "https://openlittermap.com/photos/{id}/signed-url"
OLM_TAGS_URL = "https://openlittermap.com/api/tags/all"

DEFAULT_USER_AGENT = (
    "litter-detector-baseline/0.1 (+https://github.com/adawgwats/litter-detector-baseline)"
)

# Politeness throttles — adjust if OLM team asks
METADATA_REQUEST_INTERVAL_SEC = 1.0
IMAGE_DOWNLOAD_INTERVAL_SEC = 0.3

# Backoff for 429 / 503
MAX_RETRIES = 6
INITIAL_BACKOFF_SEC = 2.0


@dataclass
class OlmPhoto:
    """One photo as returned by /global/points (post-normalization)."""

    photo_id: str  # the filename field, which uniquely identifies the photo
    longitude: float
    latitude: float
    datetime_str: str  # ISO-ish; raw from OLM
    verified: bool
    picked_up: bool
    username: Optional[str]
    team: Optional[str]
    # result_string contains the tag bundle for this photo (format TBD —
    # see verify_result_string_format() before assuming structure)
    result_string: Optional[str]


@dataclass
class IngestManifest:
    """Records every photo ingested in this run + attribution."""

    source: str = "OpenLitterMap"
    source_license: str = "CC-BY-SA-4.0"
    source_url: str = "https://openlittermap.com"
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    bbox: Optional[Tuple[float, float, float, float]] = None
    year: Optional[int] = None
    photos: list = field(default_factory=list)
    contributors: dict = field(default_factory=dict)  # username -> photo count
    attribution_note: str = (
        "Photos under CC-BY-SA-4.0 by individual OpenLitterMap contributors. "
        "Derived models trained on this corpus must credit OpenLitterMap and "
        "preserve attribution per CC-BY-SA-4.0 terms. Per-photo username and "
        "team are preserved in `photos[].username` and `photos[].team` "
        "below."
    )


# ─── HTTP helpers with politeness ────────────────────────────────────────


def _backoff_get(url: str, *, user_agent: str, params: Optional[dict] = None) -> requests.Response:
    """GET with exponential backoff on 429/503."""
    headers = {"User-Agent": user_agent}
    backoff = INITIAL_BACKOFF_SEC
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code in (429, 503):
            sleep_for = float(resp.headers.get("Retry-After", backoff))
            log.warning("HTTP %d on %s (attempt %d/%d); backing off %.1fs",
                        resp.status_code, url, attempt + 1, MAX_RETRIES, sleep_for)
            time.sleep(sleep_for)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries on {url}")


# ─── Public API ──────────────────────────────────────────────────────────


def fetch_tag_taxonomy(*, user_agent: str = DEFAULT_USER_AGENT) -> dict:
    """Return OLM's full 200+ class tag taxonomy.

    Call once per ingest run; cache locally. The taxonomy informs which
    classes the trained classifier head should output.
    """
    resp = _backoff_get(OLM_TAGS_URL, user_agent=user_agent)
    return resp.json()


def iter_photos_in_bbox(
    *,
    bbox: Tuple[float, float, float, float],  # (left, bottom, right, top) lon/lat
    year: int,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Iterator[OlmPhoto]:
    """Yield every photo OLM has in the given bbox + year.

    /global/points returns a single FeatureCollection — no pagination —
    so this is a one-shot query. Bbox-tile the world from the caller side
    if you want to pull globally.
    """
    bbox_json = json.dumps({"left": bbox[0], "bottom": bbox[1], "right": bbox[2], "top": bbox[3]})
    params = {"zoom": 18, "bbox": bbox_json, "year": year}
    log.info("OLM /global/points bbox=%s year=%d", bbox, year)
    time.sleep(METADATA_REQUEST_INTERVAL_SEC)
    resp = _backoff_get(OLM_POINTS_URL, user_agent=user_agent, params=params)
    fc = resp.json()
    for feat in fc.get("features", []):
        props = feat.get("properties", {}) or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        yield OlmPhoto(
            photo_id=str(props.get("filename") or ""),
            longitude=float(coords[0]) if coords[0] is not None else 0.0,
            latitude=float(coords[1]) if coords[1] is not None else 0.0,
            datetime_str=str(props.get("datetime") or ""),
            verified=bool(props.get("verified")),
            picked_up=bool(props.get("picked_up")),
            username=props.get("username"),
            team=props.get("team"),
            result_string=props.get("result_string"),
        )


def fetch_signed_image_url(photo_id: str, *, user_agent: str = DEFAULT_USER_AGENT) -> str:
    """Get the temporary signed URL to image bytes for a photo."""
    url = OLM_SIGNED_URL.format(id=photo_id)
    resp = _backoff_get(url, user_agent=user_agent)
    # OLM returns either a redirect or a JSON {"signed_url": "..."} payload
    try:
        return resp.json().get("signed_url") or resp.url
    except json.JSONDecodeError:
        # plain-text URL response or redirect
        return resp.text.strip() or resp.url


def download_image_bytes(signed_url: str, *, user_agent: str = DEFAULT_USER_AGENT) -> bytes:
    """Download image bytes from a previously-fetched signed URL."""
    headers = {"User-Agent": user_agent}
    resp = requests.get(signed_url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content


# ─── Top-level ingest function ───────────────────────────────────────────


def ingest_bbox(
    *,
    bbox: Tuple[float, float, float, float],
    year: int,
    storage: Storage,
    user_agent: str = DEFAULT_USER_AGENT,
    key_prefix: str = "openlittermap/photos",
    manifest_path: Optional[Path] = None,
    skip_existing: bool = True,
    max_photos: Optional[int] = None,
) -> IngestManifest:
    """Pull every photo in a bbox + year from OLM, write to storage,
    return a manifest.

    Args:
        bbox: (left, bottom, right, top) in lon/lat
        year: 2017-current
        storage: configured S3-compatible storage
        user_agent: identify yourself; please customize
        key_prefix: object key prefix in the bucket
        manifest_path: if set, also write the manifest JSON to disk
        skip_existing: skip image download if hash already in storage
        max_photos: cap to N images (useful for regional pilots — set
                    something like 1000-5000 for first runs)

    Idempotency:
        Re-running over the same bbox is safe and near-free — only
        metadata calls re-execute, image bytes are skipped if their hash
        is already in storage.

    Politeness:
        Metadata-request rate self-limited to 1/sec. Per-image downloads
        self-limited to ~3 concurrent equivalent via per-image sleep.
        Honor 429/503 with exponential backoff.
    """
    manifest = IngestManifest(bbox=bbox, year=year)

    photos = list(iter_photos_in_bbox(bbox=bbox, year=year, user_agent=user_agent))
    log.info("OLM returned %d photos for bbox=%s year=%d", len(photos), bbox, year)
    if max_photos is not None:
        photos = photos[:max_photos]
        log.info("Capped to max_photos=%d", max_photos)

    for idx, photo in enumerate(photos, start=1):
        if not photo.photo_id:
            log.warning("photo[%d] has no filename, skipping", idx)
            continue

        # Compute a tentative key from the photo_id (filename) so we can
        # skip pre-image-download if we already have something at that path
        # under our content-addressed scheme. (We still need to compute the
        # actual sha256 after download to get the final key, but we skip
        # the network round-trip for photos seen via a metadata-only key
        # path.)
        metadata_key = f"{key_prefix.rstrip('/')}/_metadata/{photo.photo_id}.json"
        if skip_existing and storage.exists(metadata_key):
            log.debug("skip %s (metadata already in storage)", photo.photo_id)
            manifest.photos.append({**asdict(photo), "skipped": True})
            continue

        # Get signed URL and download
        try:
            time.sleep(IMAGE_DOWNLOAD_INTERVAL_SEC)
            signed = fetch_signed_image_url(photo.photo_id, user_agent=user_agent)
            data = download_image_bytes(signed, user_agent=user_agent)
        except Exception as exc:
            log.warning("photo %s download failed: %s", photo.photo_id, exc)
            continue

        # Determine extension from photo_id or default to .jpg
        ext = Path(photo.photo_id).suffix.lstrip(".") or "jpg"
        image_key = content_addressed_key(prefix=key_prefix, data=data, ext=ext)

        if skip_existing and storage.exists(image_key):
            log.debug("image hash already in storage for %s", photo.photo_id)
        else:
            storage.put_bytes(image_key, data, content_type=f"image/{ext}")

        # Record metadata + write a side-by-side metadata key for future skip-detection
        photo_meta = {
            **asdict(photo),
            "image_key": image_key,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        storage.put_bytes(metadata_key, json.dumps(photo_meta).encode("utf-8"), "application/json")
        manifest.photos.append(photo_meta)

        # Tally contributor for attribution
        if photo.username:
            manifest.contributors[photo.username] = manifest.contributors.get(photo.username, 0) + 1

        if idx % 50 == 0:
            log.info("[%d/%d] ingested", idx, len(photos))

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
        log.info("wrote manifest to %s", manifest_path)

    return manifest


def verify_result_string_format(
    *,
    bbox: Tuple[float, float, float, float] = (-8.5, 51.85, -8.4, 51.92),  # Cork city, dense OLM coverage
    year: int = 2024,
    user_agent: str = DEFAULT_USER_AGENT,
    sample_size: int = 5,
) -> list:
    """One-off investigation: pull a few photos from a known-dense bbox
    and print the raw `result_string` so the maintainer can document its
    actual format (JSON? comma-separated slugs? something else?).

    Run this once before relying on `result_string` for training labels.
    """
    photos = list(iter_photos_in_bbox(bbox=bbox, year=year, user_agent=user_agent))[:sample_size]
    samples = []
    for p in photos:
        samples.append({
            "photo_id": p.photo_id,
            "username": p.username,
            "result_string_type": type(p.result_string).__name__,
            "result_string_value": p.result_string,
            "result_string_preview": (
                p.result_string[:200] if isinstance(p.result_string, str) else repr(p.result_string)
            ),
        })
    return samples
