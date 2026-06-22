"""TACO (CC-BY-4.0) corpus ingestion.

TACO is a litter detection dataset of ~1500 photographs with COCO-format
bounding-box annotations across 60 fine-grained categories. The full
annotations.json (~3 MB) is fetched on first use from the upstream repo
and cached under ``data/taco/`` (gitignored). Image bytes come from
Flickr per-image URLs embedded in the annotations.

Each TACO category maps to an OLM leaf via the crosswalk
(:mod:`litter_detector_baseline.ingest.crosswalk`). Ingest will raise
:class:`KeyError` if any TACO category is missing from the crosswalk —
silent ignores are forbidden per docs/training_quickstart.md Step 2.

Output layout in storage (mirrors openlittermap.py)::

    <key_prefix>/photos/<aa>/<bb>/<sha256>.<ext>          ← image bytes
    <key_prefix>/_metadata/<image_id>.json                ← per-image annotations
    <manifest_path>                                       ← per-run manifest

License attribution
    - Annotations: CC-BY-4.0 (TACO project).
    - Images: per-Flickr-image licenses, captured in each image's
      ``license_id`` field; the TACO ``licenses`` list is preserved in
      the manifest for downstream model-card use.
"""

from __future__ import annotations

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

from .crosswalk import map_label
from .storage import Storage, content_addressed_key

log = logging.getLogger(__name__)

TACO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
TACO_REPO_URL = "https://github.com/pedropro/TACO"
TACO_ANNOTATIONS_LICENSE = "CC-BY-4.0"

DEFAULT_USER_AGENT = (
    "litter-detector-baseline/0.1 (+https://github.com/adawgwats/litter-detector-baseline)"
)

IMAGE_DOWNLOAD_INTERVAL_SEC = 0.3  # match the OLM ingester's politeness
MAX_RETRIES = 6
INITIAL_BACKOFF_SEC = 2.0

# Repo-relative default cache location for the annotations.json payload.
_PKG_ROOT = Path(__file__).resolve().parent.parent  # .../litter_detector_baseline
_REPO_ROOT = _PKG_ROOT.parent.parent
DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "taco"


@dataclass
class TacoAnnotation:
    """A single bbox annotation with its crosswalked OLM leaf label."""

    annotation_id: int
    image_id: int
    category_id: int
    category_name: str  # TACO source label, e.g. "Glass bottle"
    olm_leaf: str       # crosswalk result, e.g. "alcohol.bottle"
    bbox: list[float]   # COCO [x, y, w, h] in pixels
    area: float
    iscrowd: int


@dataclass
class TacoImage:
    """A TACO image with all its annotations + crosswalked labels."""

    image_id: int
    file_name: str
    width: int
    height: int
    flickr_url: Optional[str]      # full-resolution Flickr URL
    flickr_640_url: Optional[str]  # 640px Flickr URL (matches imgsz=640)
    license_id: Optional[int]
    annotations: list[TacoAnnotation]


@dataclass
class TacoIngestManifest:
    """Per-run manifest. Mirrors openlittermap.IngestManifest shape."""

    source: str = "TACO"
    source_license: str = TACO_ANNOTATIONS_LICENSE
    source_url: str = TACO_REPO_URL
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    image_count: int = 0
    annotation_count: int = 0
    images: list = field(default_factory=list)
    olm_leaf_histogram: dict = field(default_factory=dict)  # olm_leaf -> count
    licenses: list = field(default_factory=list)  # per-image Flickr licenses (from annotations.json)
    attribution_note: str = (
        "TACO annotations CC-BY-4.0 by pedropro/TACO. Images carry per-Flickr "
        "licenses captured in this manifest's `licenses` list and each image's "
        "`license_id`. Downstream model cards must preserve attribution per the "
        "TACO and per-image licenses."
    )


# ─── annotations loader (fetch + cache) ──────────────────────────────────


def _ensure_annotations_cached(cache_dir: Path = DEFAULT_CACHE_DIR, *, user_agent: str = DEFAULT_USER_AGENT) -> Path:
    """Download annotations.json on first call; reuse cache on subsequent calls."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "annotations.json"
    if not cached.exists():
        log.info("downloading TACO annotations.json -> %s", cached)
        resp = requests.get(TACO_ANNOTATIONS_URL, headers={"User-Agent": user_agent}, timeout=60)
        resp.raise_for_status()
        cached.write_bytes(resp.content)
    return cached


def load_annotations(
    cache_path: Optional[Path] = None,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    """Return the parsed TACO annotations dict.

    If ``cache_path`` is None, falls back to ``DEFAULT_CACHE_DIR/annotations.json``
    and downloads it from the upstream repo if missing.
    """
    if cache_path is None:
        cache_path = _ensure_annotations_cached(user_agent=user_agent)
    return json.loads(Path(cache_path).read_text(encoding="utf-8"))


# ─── crosswalked enumeration ─────────────────────────────────────────────


def iter_taco_images(annotations: dict) -> Iterator[TacoImage]:
    """Yield :class:`TacoImage` instances with crosswalked OLM leaves.

    Raises :class:`KeyError` if a TACO category is not in the crosswalk
    (no silent ignores). Run :mod:`tests.test_crosswalk` first to guarantee
    coverage.
    """
    cats_by_id = {c["id"]: c for c in annotations["categories"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in annotations["images"]:
        out_anns: list[TacoAnnotation] = []
        for ann in anns_by_image.get(img["id"], []):
            cat = cats_by_id[ann["category_id"]]
            category_name = cat["name"]
            leaf = map_label("taco", category_name)
            if leaf is None:
                raise KeyError(
                    f"TACO category {category_name!r} (id={cat['id']}) is not "
                    "in the crosswalk; refresh configs/label_crosswalk.csv"
                )
            out_anns.append(
                TacoAnnotation(
                    annotation_id=ann["id"],
                    image_id=ann["image_id"],
                    category_id=ann["category_id"],
                    category_name=category_name,
                    olm_leaf=leaf,
                    bbox=list(ann["bbox"]),
                    area=float(ann.get("area", 0.0)),
                    iscrowd=int(ann.get("iscrowd", 0)),
                )
            )
        yield TacoImage(
            image_id=img["id"],
            file_name=img["file_name"],
            width=int(img["width"]),
            height=int(img["height"]),
            flickr_url=img.get("flickr_url"),
            flickr_640_url=img.get("flickr_640_url"),
            license_id=img.get("license"),
            annotations=out_anns,
        )


# ─── HTTP helper for image downloads ─────────────────────────────────────


def _backoff_get_bytes(url: str, *, user_agent: str) -> bytes:
    """GET with exponential backoff on 429/503/5xx, returning body bytes."""
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


# ─── Top-level ingest function ───────────────────────────────────────────


def _process_one_image(
    image: TacoImage,
    *,
    storage: Storage,
    user_agent: str,
    key_prefix: str,
    image_url_field: str,
    skip_existing: bool,
) -> dict:
    """Download + upload one TACO image. Returns a per-image result dict
    (record + status flag) for the caller to aggregate into the manifest.
    Safe to call concurrently from multiple threads (only depends on
    thread-safe `requests` + `Storage` which makes a fresh client per call).
    """
    image_record = {
        "image_id": image.image_id,
        "file_name": image.file_name,
        "width": image.width,
        "height": image.height,
        "flickr_url": image.flickr_url,
        "flickr_640_url": image.flickr_640_url,
        "license_id": image.license_id,
        "annotations": [asdict(a) for a in image.annotations],
    }
    url = getattr(image, image_url_field) or image.flickr_url
    if not url:
        return {"status": "no_url", "image_id": image.image_id, "record": image_record}

    metadata_key = f"{key_prefix.rstrip('/')}/_metadata/{image.image_id}.json"
    if skip_existing and storage.exists(metadata_key):
        return {"status": "skipped", "image_id": image.image_id,
                "record": {**image_record, "skipped": True}}

    try:
        data = _backoff_get_bytes(url, user_agent=user_agent)
    except Exception as exc:
        return {"status": "download_failed", "image_id": image.image_id,
                "error": f"{type(exc).__name__}: {exc}"}

    ext = Path(url.split("?")[0]).suffix.lstrip(".") or "jpg"
    image_key = content_addressed_key(prefix=key_prefix, data=data, ext=ext)

    if not (skip_existing and storage.exists(image_key)):
        storage.put_bytes(image_key, data, content_type=f"image/{ext}")

    full_record = {
        **image_record,
        "image_key": image_key,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    storage.put_bytes(
        metadata_key,
        json.dumps(full_record).encode("utf-8"),
        "application/json",
    )
    return {"status": "ingested", "image_id": image.image_id, "record": full_record}


def ingest_taco(
    *,
    storage: Optional[Storage],
    annotations_path: Optional[Path] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    key_prefix: str = "taco/photos",
    manifest_path: Optional[Path] = None,
    skip_existing: bool = True,
    max_photos: Optional[int] = None,
    dry_run: bool = False,
    image_url_field: str = "flickr_640_url",
    workers: int = 8,
) -> TacoIngestManifest:
    """Crosswalk + (optionally) download TACO into S3-compatible storage.

    Args:
        storage: configured S3-compatible storage. May be ``None`` when
            ``dry_run=True``.
        annotations_path: optional explicit path to a TACO annotations.json.
            If ``None``, downloads + caches under :data:`DEFAULT_CACHE_DIR`.
        key_prefix: object-key prefix in the bucket.
        manifest_path: if set, also write the manifest JSON to disk.
        skip_existing: skip image download when the metadata key already
            exists in storage. Idempotent re-runs are near-free.
        max_photos: cap to first N images (regional-pilot mode).
        dry_run: enumerate + crosswalk without downloading or uploading.
            Used by the Step 3 gate test.
        image_url_field: which Flickr URL field to download from. Default
            ``flickr_640_url`` matches the training imgsz=640.
    """
    annotations = load_annotations(annotations_path, user_agent=user_agent)
    manifest = TacoIngestManifest(
        image_count=0,
        annotation_count=0,
        licenses=list(annotations.get("licenses", [])),
    )

    images = list(iter_taco_images(annotations))
    log.info("TACO has %d images, %d annotations after crosswalk", len(images),
             sum(len(im.annotations) for im in images))
    if max_photos is not None:
        images = images[:max_photos]
        log.info("Capped to max_photos=%d", max_photos)

    if not dry_run and storage is None:
        raise ValueError("storage is required when dry_run=False")

    # Annotation/leaf histogram is pure-CPU and identical regardless of upload
    # outcome; aggregate once up front so the IO loop only tracks per-image
    # status.
    for image in images:
        manifest.annotation_count += len(image.annotations)
        for ann in image.annotations:
            manifest.olm_leaf_histogram[ann.olm_leaf] = (
                manifest.olm_leaf_histogram.get(ann.olm_leaf, 0) + 1
            )

    if dry_run:
        for image in images:
            manifest.image_count += 1
            manifest.images.append({
                "image_id": image.image_id, "file_name": image.file_name,
                "width": image.width, "height": image.height,
                "flickr_url": image.flickr_url, "flickr_640_url": image.flickr_640_url,
                "license_id": image.license_id,
                "annotations": [asdict(a) for a in image.annotations],
                "dry_run": True,
            })
    else:
        log.info("dispatching ingest of %d images to %d workers", len(images), workers)
        counter = {"done": 0, "ingested": 0, "skipped": 0, "failed": 0}
        counter_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_image,
                    image,
                    storage=storage,
                    user_agent=user_agent,
                    key_prefix=key_prefix,
                    image_url_field=image_url_field,
                    skip_existing=skip_existing,
                ): image
                for image in images
            }
            for fut in as_completed(futures):
                result = fut.result()
                with counter_lock:
                    counter["done"] += 1
                    if result["status"] == "ingested":
                        counter["ingested"] += 1
                    elif result["status"] == "skipped":
                        counter["skipped"] += 1
                    else:
                        counter["failed"] += 1
                        log.warning("image_id=%s status=%s err=%s",
                                    result.get("image_id"), result["status"],
                                    result.get("error"))
                    done = counter["done"]
                if result.get("record"):
                    manifest.images.append(result["record"])
                    manifest.image_count += 1
                if done % 50 == 0 or done == len(images):
                    log.info(
                        "[%d/%d] ingested=%d skipped=%d failed=%d",
                        done, len(images),
                        counter["ingested"], counter["skipped"], counter["failed"],
                    )

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
        log.info("wrote manifest to %s", manifest_path)

    return manifest
