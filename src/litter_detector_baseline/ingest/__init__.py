"""Corpus ingestion for litter-detection training datasets.

Currently implemented:
    - OpenLitterMap (CC-BY-SA-4.0) via the public /global/points endpoint

Planned (see docs/ingestion.md for status):
    - TACO (open via Roboflow / Zenodo)
    - pLitter (gicait/pLitter on GitHub)
    - KABML street images (Roboflow Universe)
    - ShitSpotter / ScatSpotter (CC-BY-4.0 via Erotemic/shitspotter on GitHub)
    - Hard-negative sources (leaves, shadows, mulch — see docs)

All ingesters write to an S3-compatible object store (R2 / S3 / B2) via a
common ``Storage`` interface. The object key naming convention is
content-addressable so a re-ingest of the same image is a no-op.

Public usage::

    from litter_detector_baseline.ingest import openlittermap, Storage
    storage = Storage.from_env()  # reads endpoint + creds from env vars
    openlittermap.ingest_bbox(
        bbox=(-77.65, 38.13, -77.38, 38.40),
        year=2024,
        storage=storage,
        user_agent="my-pipeline/0.1 (contact@example.org)",
    )

Per-source modules preserve upstream attribution + license per the
``manifest.json`` written alongside each batch — required for CC-BY-SA
downstream redistribution.
"""

from .storage import Storage

__all__ = ["Storage"]
