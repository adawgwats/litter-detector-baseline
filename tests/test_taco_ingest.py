"""TACO ingestion gate tests.

Gate per docs/training_quickstart.md Step 3: ``--dry-run`` enumerates all
1500 TACO images with their crosswalked OLM-leaf labels — no missing
crosswalk entries, no silent ignores. Image bytes are not downloaded in
the gate path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litter_detector_baseline.ingest import taco

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHED_ANNOTATIONS = REPO_ROOT / "data" / "taco" / "annotations.json"


@pytest.fixture(scope="module")
def annotations_path() -> Path:
    """Skip if the cached annotations.json is absent.

    The CLI dry-run path will download + cache on first call, but tests
    shouldn't go out to the network by default — keep this test cheap and
    deterministic. Run ``python -m litter_detector_baseline.ingest taco
    --dry-run`` once first to populate the cache, then re-run pytest.
    """
    if not CACHED_ANNOTATIONS.exists():
        pytest.skip(
            f"TACO annotations cache not found at {CACHED_ANNOTATIONS}; "
            "run `python -m litter_detector_baseline.ingest taco --dry-run` "
            "once to populate, then re-run pytest"
        )
    return CACHED_ANNOTATIONS


def test_dry_run_enumerates_all_1500_images_with_crosswalk(annotations_path: Path):
    """Step 3 gate: dry-run lists every TACO image with crosswalked labels."""
    manifest = taco.ingest_taco(
        storage=None,
        annotations_path=annotations_path,
        dry_run=True,
    )
    assert manifest.image_count == 1500, f"expected 1500 images, got {manifest.image_count}"
    assert manifest.annotation_count > 0, "no annotations were enumerated"

    # No silent ignores: every annotation has a non-None olm_leaf.
    for img in manifest.images:
        for ann in img["annotations"]:
            assert ann["olm_leaf"], (
                f"annotation {ann['annotation_id']} on image {img['image_id']} "
                "has empty olm_leaf"
            )


def test_olm_leaf_histogram_is_populated_and_sane(annotations_path: Path):
    """Smoke: at minimum, smoking.butts and softdrinks.bottle should appear."""
    manifest = taco.ingest_taco(
        storage=None,
        annotations_path=annotations_path,
        dry_run=True,
    )
    # TACO has a "Cigarette" category that maps to smoking.butts.
    assert manifest.olm_leaf_histogram.get("smoking.butts", 0) > 0
    # ...and "Clear plastic bottle" / "Other plastic bottle" -> softdrinks.bottle.
    assert manifest.olm_leaf_histogram.get("softdrinks.bottle", 0) > 0
    # All leaves should be valid OLM leaves (loader validates targets at load time).
    from litter_detector_baseline.ingest.crosswalk import valid_olm_leaves

    leaves = valid_olm_leaves()
    for leaf in manifest.olm_leaf_histogram:
        assert leaf in leaves, f"crosswalk emitted {leaf!r} which is not in OLM snapshot"


def test_max_photos_caps_enumeration(annotations_path: Path):
    manifest = taco.ingest_taco(
        storage=None,
        annotations_path=annotations_path,
        dry_run=True,
        max_photos=10,
    )
    assert manifest.image_count == 10
