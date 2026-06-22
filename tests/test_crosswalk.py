"""Crosswalk coverage + integrity tests.

Gate per docs/training_quickstart.md Step 2: every TACO label round-trips
through the crosswalk and finds a non-None mapping. No silent ignores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litter_detector_baseline.ingest.crosswalk import (
    all_mapped_labels,
    map_label,
    valid_olm_leaves,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TACO_SNAPSHOT = REPO_ROOT / "configs" / "taxonomies" / "taco_categories.json"


def _taco_labels() -> list[str]:
    data = json.loads(TACO_SNAPSHOT.read_text(encoding="utf-8"))
    return [c["name"] for c in data["categories"]]


def test_olm_snapshot_loads_and_is_nonempty():
    leaves = valid_olm_leaves()
    assert len(leaves) > 100  # OLM had 175 (category x object) leaves at snapshot time
    # Spot-check a few canonical leaves we know exist:
    assert "smoking.butts" in leaves
    assert "softdrinks.bottle" in leaves
    assert "alcohol.bottle" in leaves


@pytest.mark.parametrize("taco_label", _taco_labels())
def test_every_taco_label_maps_to_non_none(taco_label: str):
    """Gate: every TACO category has an OLM leaf mapping. No silent ignores."""
    leaf = map_label("taco", taco_label)
    assert leaf is not None, f"TACO label {taco_label!r} has no crosswalk entry"


def test_every_taco_target_is_a_real_olm_leaf():
    """Defensive: ensures crosswalk targets resolve to actual OLM leaves.

    The crosswalk loader already validates this at load-time; this test
    pins the behavior so a regression in the loader is caught explicitly.
    """
    leaves = valid_olm_leaves()
    for taco_label, olm_leaf in all_mapped_labels("taco").items():
        assert olm_leaf in leaves, (
            f"TACO {taco_label!r} maps to {olm_leaf!r} which is not in OLM snapshot"
        )


def test_taco_coverage_is_complete():
    """All 60 TACO categories appear in the crosswalk (no missing rows)."""
    taco_labels = set(_taco_labels())
    crosswalked = set(all_mapped_labels("taco").keys())
    missing = taco_labels - crosswalked
    extra = crosswalked - taco_labels
    assert not missing, f"TACO labels missing from crosswalk: {sorted(missing)}"
    assert not extra, f"crosswalk has TACO labels not in snapshot: {sorted(extra)}"
