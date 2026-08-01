"""Coarse-10 rollup integrity tests.

Gate for training/eval_v2.py's second reporting granularity: the rollup
must cover every one of the 43 canonical leaves (the TACO-mapped OLM
leaves from the crosswalk) exactly once, with exactly the ten groups the
V2 plan names. A leaf added to the crosswalk without a rollup assignment
fails here, not silently at eval time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ROLLUP_YAML = REPO_ROOT / "training" / "rollups" / "coarse10.yaml"
CROSSWALK_CSV = REPO_ROOT / "configs" / "label_crosswalk.csv"

EXPECTED_GROUPS = {
    "bottles", "caps_lids", "cans", "bags_film_wrappers", "cups_straws",
    "food_containers", "paper", "glass_broken", "smoking", "other",
}


def _canonical_leaves() -> set[str]:
    """The 43-leaf canonical space: TACO-mapped crosswalk targets, minus
    the synthetic hard-negative marker."""
    rows = [
        ln for ln in CROSSWALK_CSV.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    reader = csv.DictReader(rows)
    return {
        row["olm_leaf_label"]
        for row in reader
        if row["source_dataset"] == "taco" and row["olm_leaf_label"] != "no_litter"
    }


def _rollup_groups() -> dict[str, list[str]]:
    return yaml.safe_load(ROLLUP_YAML.read_text(encoding="utf-8"))["groups"]


def test_exactly_ten_expected_groups():
    assert set(_rollup_groups().keys()) == EXPECTED_GROUPS


def test_every_canonical_leaf_assigned_exactly_once():
    groups = _rollup_groups()
    assigned = [leaf for leaves in groups.values() for leaf in leaves]
    dupes = {leaf for leaf in assigned if assigned.count(leaf) > 1}
    assert not dupes, f"leaves in multiple groups: {sorted(dupes)}"

    canonical = _canonical_leaves()
    assert len(canonical) == 43
    missing = canonical - set(assigned)
    assert not missing, f"canonical leaves without a group: {sorted(missing)}"
    extra = set(assigned) - canonical
    assert not extra, f"rollup names unknown leaves: {sorted(extra)}"


def test_no_empty_groups():
    for name, leaves in _rollup_groups().items():
        assert leaves, f"group {name!r} is empty"
