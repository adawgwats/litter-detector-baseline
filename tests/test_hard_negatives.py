"""Hard-negatives gate tests.

Gate per docs/training_quickstart.md Step 4:
  * hard-negatives corpus is at least 500 images
  * all images have verified CC0 or compatible-open licenses
  * license manifest is written next to the corpus
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litter_detector_baseline.ingest import hard_negatives
from litter_detector_baseline.ingest.crosswalk import (
    SYNTHETIC_LEAVES,
    valid_olm_leaves,
)
from litter_detector_baseline.ingest.hard_negatives import NO_LITTER_LEAF

REPO_ROOT = Path(__file__).resolve().parent.parent

SEED_PATH = REPO_ROOT / "configs" / "hard_negatives_seed.txt"
LICENSES_PATH = REPO_ROOT / "configs" / "hard_negatives_licenses.json"

# Mirrors the script's ACCEPTED_LICENSES_LOWER. Kept lowercased; the gate
# verifies every accepted record's license is in this set.
ACCEPTED_LICENSES_LOWER = {
    "cc0",
    "public domain",
    "pd",
    "no restrictions",
    "cc by 2.0",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa 2.0",
    "cc by-sa 2.5",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
    "attribution",
}


def _skip_if_missing():
    if not SEED_PATH.exists() or not LICENSES_PATH.exists():
        pytest.skip(
            f"hard-negatives seed/licenses not present at {SEED_PATH} / "
            f"{LICENSES_PATH}; run `python scripts/build_hard_negatives_seed.py`"
        )


def test_seed_has_at_least_500_urls():
    _skip_if_missing()
    urls = hard_negatives.load_seed_urls(SEED_PATH)
    assert len(urls) >= 500, f"hard-negatives gate requires >=500 URLs; got {len(urls)}"


def test_license_manifest_covers_every_seed_url():
    _skip_if_missing()
    urls = hard_negatives.load_seed_urls(SEED_PATH)
    license_idx = hard_negatives.load_license_index(LICENSES_PATH)
    missing = [u for u in urls if u not in license_idx]
    assert not missing, f"{len(missing)} seed URLs have no license record (e.g. {missing[:3]})"


def test_every_license_is_in_accepted_set():
    _skip_if_missing()
    license_idx = hard_negatives.load_license_index(LICENSES_PATH)
    bad = []
    for url, rec in license_idx.items():
        license_lower = rec.get("license_short", "").strip().lower()
        if license_lower not in ACCEPTED_LICENSES_LOWER:
            bad.append((rec.get("license_short"), url))
    assert not bad, (
        f"{len(bad)} hard-negative entries have non-open licenses, e.g. {bad[:3]}"
    )


def test_no_litter_synthetic_leaf_is_registered():
    """The 'no_litter' label must be a recognized synthetic crosswalk target."""
    assert NO_LITTER_LEAF in SYNTHETIC_LEAVES
    # And must NOT collide with a real OLM leaf:
    assert NO_LITTER_LEAF not in valid_olm_leaves()


def test_dry_run_enumerates_seed_with_no_litter_label():
    _skip_if_missing()
    manifest = hard_negatives.ingest_hard_negatives(
        storage=None,
        seed_path=SEED_PATH,
        licenses_path=LICENSES_PATH,
        dry_run=True,
    )
    assert manifest.image_count >= 500
    # Every image is labeled no_litter (the gate's reason for existing):
    for img in manifest.images:
        assert img["olm_leaf"] == NO_LITTER_LEAF, img
