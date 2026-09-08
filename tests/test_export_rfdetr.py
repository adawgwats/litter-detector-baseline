"""Unit tests for export.export_rfdetr's safety mechanisms.

The reason this file exists: `_check_class_alignment` is the strongest
correctness guarantee in the export path — it is the only thing standing
between a re-ordered training run and a served model whose logit column
*i* is not canonical class *i*, which would mislabel every detection
without failing anything. Until this file was written it had never been
observed to raise. An unexercised guard is a comment.

Same for the version regex: `export()` derives the artifact's precision
by splitting the version string, so a version without a precision suffix
does not merely look wrong, it makes `precision` garbage.

No checkpoint or model download is needed — every path tested here is
reached before `RFDETRSmall` is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from export.export_rfdetr import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_RESOLUTION,
    _check_class_alignment,
)
from export.export_yolov8 import _validate_version

CANONICAL = ["alcohol.bottle", "food.can", "smoking.butts"]


def _checkpoint_with_config(tmp_path: Path, class_names: list[str] | None) -> Path:
    """A .pth path whose sibling training_config.json is what we control.

    The checkpoint file itself is never opened by the code under test —
    only its parent directory matters — so a placeholder is enough.
    """
    ckpt = tmp_path / "checkpoint_best_total.pth"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    if class_names is not None:
        (tmp_path / "training_config.json").write_text(
            json.dumps({"class_names": class_names}), encoding="utf-8"
        )
    return ckpt


# --------------------------------------------------------------------------
# _check_class_alignment
# --------------------------------------------------------------------------

def test_class_alignment_passes_on_exact_agreement(tmp_path: Path) -> None:
    ckpt = _checkpoint_with_config(tmp_path, CANONICAL)
    _check_class_alignment(ckpt, CANONICAL)  # must not raise


def test_class_alignment_raises_on_reordering(tmp_path: Path) -> None:
    """The dangerous case: same classes, different order.

    Set-equal but order-different is exactly what a naive check would
    miss, and it is the failure that silently mislabels every detection.
    """
    reordered = [CANONICAL[1], CANONICAL[0], CANONICAL[2]]
    ckpt = _checkpoint_with_config(tmp_path, reordered)
    with pytest.raises(RuntimeError) as exc:
        _check_class_alignment(ckpt, CANONICAL)
    msg = str(exc.value)
    assert "class-order mismatch" in msg
    assert "training_config.json" in msg
    # The message must name the mismatch concretely enough to act on.
    assert str(len(reordered)) in msg
    assert "--skip-class-check" in msg


def test_class_alignment_raises_on_different_class_count(tmp_path: Path) -> None:
    ckpt = _checkpoint_with_config(tmp_path, CANONICAL[:2])
    with pytest.raises(RuntimeError, match="class-order mismatch"):
        _check_class_alignment(ckpt, CANONICAL)


def test_class_alignment_warns_and_proceeds_when_config_absent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No training_config.json is a WARNING, not an error — deliberately.

    Older rfdetr runs did not write one, and refusing them outright would
    make the guard unshippable. The compensating control is that the
    warning says the order could not be verified.
    """
    ckpt = _checkpoint_with_config(tmp_path, None)
    with caplog.at_level("WARNING"):
        _check_class_alignment(ckpt, CANONICAL)
    assert "cannot verify" in caplog.text


def test_class_alignment_warns_and_proceeds_on_unparseable_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ckpt = _checkpoint_with_config(tmp_path, CANONICAL)
    (tmp_path / "training_config.json").write_text("{ not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        _check_class_alignment(ckpt, CANONICAL)
    assert "could not parse" in caplog.text


def test_class_alignment_proceeds_when_config_has_no_class_names(
    tmp_path: Path,
) -> None:
    """An empty/absent ``class_names`` key means nothing to compare against.

    Documents current behaviour rather than endorsing it — see the note
    in reports/CONVERSION-REPORT.md about this being the one silent path
    through the guard.
    """
    ckpt = _checkpoint_with_config(tmp_path, CANONICAL)
    (tmp_path / "training_config.json").write_text(
        json.dumps({"other_key": 1}), encoding="utf-8"
    )
    _check_class_alignment(ckpt, CANONICAL)  # must not raise


def test_skip_class_check_flag_is_wired_to_bypass_the_guard() -> None:
    """`--skip-class-check` must reach `export()` as `skip_class_check`.

    The guard is bypassable on purpose (a canonical-order run whose
    config was lost). This asserts the escape hatch exists and is
    off by default, so it can never be on by accident.
    """
    import inspect

    from export.export_rfdetr import export, main

    sig = inspect.signature(export)
    assert sig.parameters["skip_class_check"].default is False

    src = inspect.getsource(main)
    assert "--skip-class-check" in src
    assert "skip_class_check=args.skip_class_check" in src


# --------------------------------------------------------------------------
# version string -> precision
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "version",
    ["v2.0.0", "v2.0.0-", "v2.0.0-fp8", "2.0.0-fp16", "v2.0-fp16", "v2.0.0-FP16"],
)
def test_version_without_valid_precision_suffix_is_rejected(version: str) -> None:
    """`export()` does `version.rsplit("-", 1)[1]` to get the precision.

    Without the regex that split yields nonsense (or IndexErrors), and
    the nonsense would be written into the sidecar as the artifact's
    declared precision.
    """
    with pytest.raises(ValueError, match="does not match"):
        _validate_version(version)


@pytest.mark.parametrize("version", ["v2.0.0-fp32", "v2.0.0-fp16", "v10.3.11-int8"])
def test_valid_versions_accepted_and_precision_recoverable(version: str) -> None:
    _validate_version(version)
    assert version.rsplit("-", 1)[1] in {"fp32", "fp16", "int8"}


# --------------------------------------------------------------------------
# sidecar defaults the consumer depends on
# --------------------------------------------------------------------------

def test_default_resolution_is_divisible_by_32() -> None:
    """RFDETRSmall's patch_size * num_windows == 32; export() enforces it."""
    assert DEFAULT_RESOLUTION % 32 == 0


def test_recommended_thresholds_are_in_unit_range() -> None:
    assert 0.0 < DEFAULT_CONF < 1.0
    assert 0.0 < DEFAULT_IOU < 1.0
