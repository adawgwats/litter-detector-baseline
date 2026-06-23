"""Unit tests for the pure helpers in export.export_yolov8.

The integration path (loading weights + running ultralytics.export()) is
covered by manual smoke runs against real V1 weights and is not tested
here — would require either committing a checkpoint to git or pulling
~10 MB on every CI run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from export.export_yolov8 import (
    MAX_ONNX_BYTES,
    _camel,
    _precision_to_export_kwargs,
    _read_class_names,
    _read_training_metrics,
    _read_trained_at,
    _validate_version,
)


class TestValidateVersion:
    @pytest.mark.parametrize(
        "version",
        ["v1.0.0-fp16", "v1.2.3-fp32", "v2.0.0-int8", "v10.20.30-fp16"],
    )
    def test_accepts_valid_format(self, version: str) -> None:
        _validate_version(version)  # does not raise

    @pytest.mark.parametrize(
        "version",
        [
            "1.0.0-fp16",                # missing leading 'v'
            "v1.0-fp16",                 # missing patch
            "v1.0.0",                    # missing precision
            "v1.0.0-fp64",               # unknown precision
            "v1.0.0-FP16",               # uppercase precision
            "v1.0.0-fp16-extra",         # trailing junk
        ],
    )
    def test_rejects_malformed(self, version: str) -> None:
        with pytest.raises(ValueError, match="does not match"):
            _validate_version(version)


class TestPrecisionToExportKwargs:
    def test_fp32(self) -> None:
        assert _precision_to_export_kwargs("fp32") == {"half": False}

    def test_fp16(self) -> None:
        assert _precision_to_export_kwargs("fp16") == {"half": True}

    def test_int8_unimplemented(self) -> None:
        # Spec §3 lists INT8 as a non-goal until calibration data is wired.
        with pytest.raises(NotImplementedError, match="calibration"):
            _precision_to_export_kwargs("int8")

    def test_unknown_precision(self) -> None:
        with pytest.raises(ValueError, match="unknown precision"):
            _precision_to_export_kwargs("bf16")


class TestReadClassNames:
    def test_reads_ordered_list(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(
            "names:\n  0: alpha\n  1: bravo\n  2: charlie\n",
            encoding="utf-8",
        )
        assert _read_class_names(yaml_path) == ["alpha", "bravo", "charlie"]

    def test_preserves_order_despite_dict_iteration_order(
        self, tmp_path: Path
    ) -> None:
        # YAML dict order may not be insertion order; we sort by index.
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(
            "names:\n  2: charlie\n  0: alpha\n  1: bravo\n",
            encoding="utf-8",
        )
        assert _read_class_names(yaml_path) == ["alpha", "bravo", "charlie"]

    def test_rejects_list_format(self, tmp_path: Path) -> None:
        # The legacy `names: [alpha, bravo]` format is NOT what our
        # training pipeline emits; flag it loudly.
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text("names:\n  - alpha\n  - bravo\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a dict"):
            _read_class_names(yaml_path)


class TestReadTrainingMetrics:
    def test_extracts_known_keys(self, tmp_path: Path) -> None:
        rcpt = tmp_path / "eval.json"
        rcpt.write_text(
            json.dumps(
                {
                    "map50": 0.048,
                    "map5095": 0.034,
                    "hazards_mean_recall": 0.0,
                    "hard_negative_fpr": 0.045,
                    "run_name": "ignored",
                }
            ),
            encoding="utf-8",
        )
        assert _read_training_metrics(rcpt) == {
            "map50": 0.048,
            "map5095": 0.034,
            "hazardsMeanRecall": 0.0,
            "hardNegativeFpr": 0.045,
        }

    def test_skips_nulls(self, tmp_path: Path) -> None:
        rcpt = tmp_path / "eval.json"
        rcpt.write_text(
            json.dumps({"map50": 0.05, "hard_negative_fpr": None}),
            encoding="utf-8",
        )
        assert _read_training_metrics(rcpt) == {"map50": 0.05}

    def test_returns_empty_when_path_none(self) -> None:
        assert _read_training_metrics(None) == {}

    def test_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        assert _read_training_metrics(tmp_path / "absent.json") == {}

    def test_returns_empty_on_malformed_json(self, tmp_path: Path) -> None:
        rcpt = tmp_path / "broken.json"
        rcpt.write_text("not json {", encoding="utf-8")
        assert _read_training_metrics(rcpt) == {}


class TestReadTrainedAt:
    def test_prefers_finished_at(self, tmp_path: Path) -> None:
        rcpt = tmp_path / "energy.json"
        rcpt.write_text(
            json.dumps(
                {
                    "started_at_utc": "2026-06-01T18:00:00+00:00",
                    "finished_at_utc": "2026-06-01T19:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        assert _read_trained_at(rcpt) == "2026-06-01T19:00:00+00:00"

    def test_falls_back_to_started_at(self, tmp_path: Path) -> None:
        rcpt = tmp_path / "energy.json"
        rcpt.write_text(
            json.dumps({"started_at_utc": "2026-06-01T18:00:00+00:00"}),
            encoding="utf-8",
        )
        assert _read_trained_at(rcpt) == "2026-06-01T18:00:00+00:00"

    def test_returns_none_when_absent(self) -> None:
        assert _read_trained_at(None) is None


class TestCamel:
    @pytest.mark.parametrize(
        "snake,camel",
        [
            ("map50", "map50"),
            ("map5095", "map5095"),
            ("hazards_mean_recall", "hazardsMeanRecall"),
            ("hard_negative_fpr", "hardNegativeFpr"),
            ("a", "a"),
            ("a_b_c_d", "aBCD"),
        ],
    )
    def test_snake_to_camel(self, snake: str, camel: str) -> None:
        assert _camel(snake) == camel


class TestMaxBytesConstant:
    def test_matches_spec(self) -> None:
        # Spec §10 perf budget — if this constant changes, update the
        # spec too. Asserting here so the constant doesn't quietly drift.
        assert MAX_ONNX_BYTES == 6 * 1024 * 1024
