"""Label crosswalk: source-dataset labels -> OLM leaf taxonomy.

Each upstream litter dataset (TACO, Drinking Waste, ShitSpotter, ...) uses
its own label vocabulary. Training and inference both speak OLM's leaf
taxonomy (category x object pairs from /api/tags/all). This module loads
the CSV crosswalk that maps the former to the latter and validates that
every target leaf is a real OLM leaf.

OLM leaf labels use dotted form ``<category>.<object>`` (e.g.
``smoking.butts``, ``softdrinks.bottle``). The authoritative leaf list is
the snapshot at ``configs/taxonomies/olm_leaves.json`` — refresh it via
``scripts/refresh_taxonomy_snapshots.py`` when OLM's taxonomy changes.

Typical usage::

    from litter_detector_baseline.ingest.crosswalk import map_label
    olm_leaf = map_label("taco", "Glass bottle")  # -> "alcohol.bottle"

Returns ``None`` if the (source_dataset, source_label) pair is not in the
crosswalk. Callers that require coverage (e.g. ingest pipelines) should
assert non-None and treat None as a manifest bug, not a runtime condition.
"""

from __future__ import annotations

import csv
import functools
import json
from importlib import resources
from pathlib import Path
from typing import Optional

# Package-relative paths to data files. The crosswalk + taxonomy snapshots
# live under ``configs/`` at the repo root; the package itself lives under
# ``src/litter_detector_baseline/``. Walk up from this file to find them.
_PKG_ROOT = Path(__file__).resolve().parent.parent  # .../litter_detector_baseline
_REPO_ROOT = _PKG_ROOT.parent.parent  # .../litter-detector-baseline
_CONFIGS = _REPO_ROOT / "configs"

CROSSWALK_CSV = _CONFIGS / "label_crosswalk.csv"
OLM_LEAVES_JSON = _CONFIGS / "taxonomies" / "olm_leaves.json"

# Synthetic leaves are not part of OLM's taxonomy but are valid crosswalk
# targets. ``no_litter`` is the V1 marker for hard-negatives (leaves, mulch,
# shadows, painted asphalt) — images we want the model to learn produce NO
# detections on. See docs/contributor_assist_goal.md § Success metrics:
# leaves are EXPLICITLY rejected as litter in V1.
SYNTHETIC_LEAVES: frozenset[str] = frozenset({"no_litter"})


@functools.lru_cache(maxsize=1)
def _load_olm_leaves() -> frozenset[str]:
    """Return the set of valid OLM leaf labels (``category.object`` form)."""
    data = json.loads(OLM_LEAVES_JSON.read_text(encoding="utf-8"))
    return frozenset(f"{leaf['category']}.{leaf['object']}" for leaf in data["leaves"])


@functools.lru_cache(maxsize=1)
def _load_crosswalk() -> dict[tuple[str, str], str]:
    """Load the crosswalk CSV. Validates targets against the OLM snapshot.

    Returns a mapping (source_dataset, source_label) -> olm_leaf_label.
    Raises ValueError on:
      - duplicate (source_dataset, source_label) rows
      - olm_leaf_label not present in the OLM leaf snapshot
    """
    valid_leaves = _load_olm_leaves() | SYNTHETIC_LEAVES
    mapping: dict[tuple[str, str], str] = {}
    with CROSSWALK_CSV.open("r", encoding="utf-8", newline="") as fh:
        # Strip comment + blank lines before handing to csv.reader so the
        # header row is the first non-comment line.
        rows = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(rows)
    expected_cols = {"source_dataset", "source_label", "olm_leaf_label"}
    if set(reader.fieldnames or []) != expected_cols:
        raise ValueError(
            f"{CROSSWALK_CSV.name} columns are {reader.fieldnames!r}; "
            f"expected {sorted(expected_cols)!r}"
        )
    for row in reader:
        key = (row["source_dataset"], row["source_label"])
        leaf = row["olm_leaf_label"]
        if key in mapping:
            raise ValueError(f"duplicate crosswalk entry for {key!r}")
        if leaf not in valid_leaves:
            raise ValueError(
                f"crosswalk target {leaf!r} for {key!r} is not in OLM leaf "
                f"snapshot ({OLM_LEAVES_JSON.name}); refresh snapshot or fix CSV"
            )
        mapping[key] = leaf
    return mapping


def map_label(source_dataset: str, source_label: str) -> Optional[str]:
    """Map a source-dataset label to an OLM leaf label.

    Returns ``None`` if the pair is not in the crosswalk. Callers that
    require coverage (ingest pipelines) should assert non-None.
    """
    return _load_crosswalk().get((source_dataset, source_label))


def all_mapped_labels(source_dataset: str) -> dict[str, str]:
    """Return every (source_label -> olm_leaf) entry for a given source.

    Useful for batch validation of an upstream dataset against the crosswalk.
    """
    return {
        src_label: leaf
        for (src, src_label), leaf in _load_crosswalk().items()
        if src == source_dataset
    }


def valid_olm_leaves() -> frozenset[str]:
    """Public accessor for the validated OLM leaf set."""
    return _load_olm_leaves()
