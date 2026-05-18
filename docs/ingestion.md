# Corpus ingestion

The `litter_detector_baseline.ingest` subpackage pulls training images +
annotations from public litter datasets into an S3-compatible object
store (Cloudflare R2, AWS S3, Backblaze B2, MinIO, etc.).

## Why one package, multiple sources

Training a robust litter detector requires multiple data sources because
each has structural weaknesses:

| Dataset | Strength | Weakness |
|---|---|---|
| OpenLitterMap | 500K+ images, global, citizen-contributed, brand/material/object tags | Per-image multi-label only (no bounding boxes); contributor-camera close-up perspective |
| TACO | Bounding boxes + segmentation masks, 60 classes, COCO format | Only ~1.5K images |
| pLitter | Per-object bboxes + masks, **streetview perspective** | Smaller; specific geography |
| KABML Street Images | Streetview perspective bboxes | Variable quality, community uploaded |
| ShitSpotter / ScatSpotter | Organic-waste (dog feces) class with **before/after/negative (BAN) protocol** | Single object class, ~9K images |
| Hard-negative sources (TBD) | Visually-confusing non-litter (leaves, shadows, mulch) | Need to curate or generate |

A single trained classifier benefits from harmonizing across all of
these. Each source has its own ingestion module here.

## Implementation status

| Source | Module | Status |
|---|---|---|
| OpenLitterMap | `openlittermap.py` | ✅ Implemented (this commit) |
| TACO | `taco.py` | ⏳ Not yet — see issue #2 |
| pLitter | `plitter.py` | ⏳ Not yet — see issue #2 |
| KABML | `kabml.py` | ⏳ Not yet — see issue #2 |
| ShitSpotter | `shitspotter.py` | ⏳ Not yet — see issue #2 |
| Hard negatives | `hard_negatives.py` | ⏳ Not yet — see issue #2 |

## Setup

```sh
pip install -e ".[ingest]"
```

Set required env vars:

```sh
export S3_ACCESS_KEY_ID=...           # your R2 / S3 / B2 access key
export S3_SECRET_ACCESS_KEY=...       # your R2 / S3 / B2 secret key
export S3_ENDPOINT_URL=https://<acct>.r2.cloudflarestorage.com  # omit for AWS S3 default
export S3_BUCKET=my-litter-corpus
export S3_REGION=auto                 # "auto" works for R2; us-east-1 for AWS
```

## Usage

### Verify OLM's `result_string` tag format (do this once before any large pull)

```sh
python -m litter_detector_baseline.ingest verify-olm-tags
```

This pulls ~5 photos from Cork city center (OLM's home base, dense
coverage) and prints the raw `result_string` so you can document the
actual tag format before depending on it.

### Download the OLM tag taxonomy

```sh
python -m litter_detector_baseline.ingest olm-tags --out olm-tags.json
```

### Ingest OLM photos in a bounding box

```sh
# Small test — Fredericksburg, VA metro area, 2024
python -m litter_detector_baseline.ingest olm-bbox \
    --bbox -77.65,38.13,-77.38,38.40 \
    --year 2024 \
    --max-photos 50 \
    --manifest out/olm-fredericksburg-2024-manifest.json
```

For a full corpus pull, bbox-tile the world or your target area and
loop. The ingester is idempotent (sha256-keyed) so re-running is a
no-op for photos already in storage — re-starting a crashed pull is
safe and cheap.

## Politeness protocol

The OLM ingester self-throttles to:

- ~1 metadata request per second to `/global/points`
- ~3 concurrent image downloads (300ms per-image floor)
- Exponential backoff on HTTP 429 / 503
- Real `User-Agent` identifying the caller (customize via `--user-agent`)

Before doing a several-hundred-thousand-photo bulk pull, please email
the OpenLitterMap team to let them know: they're a small group running
a public-good service, and a friendly heads-up costs nothing while
preserving the option of partnership down the line.

## Attribution (CC-BY-SA-4.0 requirement)

OpenLitterMap data is licensed CC-BY-SA-4.0. The ingester preserves
per-photo `username` and `team` in the manifest. **Downstream model
cards** for any model trained on this corpus must:

1. Credit OpenLitterMap as the upstream source with link
2. Note the per-contributor attribution chain (link to manifest or
   tag the contributor names alongside the model)
3. License the trained model + weights under CC-BY-SA-4.0 (share-alike
   propagates)

The same principle will apply to other CC-BY-SA / CC-BY datasets as
their ingesters land (ShitSpotter, KABML, etc.).

## Storage layout

Within the configured bucket, the OLM ingester writes:

```
openlittermap/photos/<aa>/<bb>/<sha256>.<ext>     ← image bytes, content-addressable
openlittermap/photos/_metadata/<photo_id>.json    ← per-photo metadata
openlittermap/manifests/<run-id>.json             ← per-run manifest (caller writes)
```

The sharded directory structure (sha256 first 2 + next 2 chars) keeps
any single directory listing small even at millions of files.

## Cost estimate (R2 specifically)

For a full OLM pull of ~375 GB:

| Line item | Cost |
|---|---|
| Storage: 10 GB free + 365 × $0.015/GB/mo | ~$66/year |
| Class A ops (PUT during ingest, ~500K under 1M free/mo) | $0 |
| Class B ops (GET during training, ~50M after free tier) | ~$18/year |
| Egress (R2 always free) | $0 |
| **Total** | **~$84/year** |

S3 would cost roughly 2× this; B2 ~half. R2 is the recommended default
because of universal-free egress.

## Reproducibility

The manifest written by each ingest run captures:

- bbox, year, ingestion timestamp
- Per-photo: photo_id, coordinates, datetime, verified-flag, username, team, result_string
- Contributor count rollup for attribution

This is sufficient to reconstruct the training corpus state at any
point, audit which contributor's data influenced which model version,
and comply with CC-BY-SA-4.0 attribution requirements.
