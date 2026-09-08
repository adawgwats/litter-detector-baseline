"""Tests for the shared precision-conversion path and the target descriptor.

Two things are being defended here, and they are different.

**The descriptor is data.** A target names its precision, IO boundary,
opset, input shape, size budget, backend and architecture. If adding a
target ever requires editing a function body, this file should start
failing — hence the sweep over ``TARGETS`` and the assertions that
``derive()`` is how a CLI flag reaches the matrix.

**The refusals fire.** The bug this module exists to prevent is silent:
you ask ultralytics for fp16 expecting an fp32 IO boundary, it gives you
an fp16 boundary, and nothing anywhere says so. Every unsupported
combination below is therefore asserted to *raise*, with the message
checked, because an unexercised guard is a comment
(cf. tests/test_export_rfdetr.py, same argument).

The fp16 graph conversion is exercised against a real ONNX model built
in-process — four by four, one MatMul — rather than mocked. Mocking the
converter would test that we call it, not that ``keep_io_types`` does
what the descriptor claims, which is the only interesting question.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from export.precision import (
    BACKENDS,
    SHIPPED,
    TARGETS,
    Backend,
    ConversionResult,
    ExportTarget,
    IOPrecision,
    Mechanism,
    Precision,
    UnsupportedTargetError,
    convert,
    get_target,
    plan,
    target_for,
)

onnx = pytest.importorskip("onnx", reason="onnx is in the [parity] extra, not required")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _tiny_fp32_model(path: Path) -> Path:
    """A real, minimal fp32 ONNX graph: y = x @ w.

    Small enough to convert in milliseconds, real enough that
    ``convert_float_to_float16`` does to it what it does to a detector:
    retype the initializer, retype the interior, and insert boundary
    casts (or not) according to ``keep_io_types``.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(
        np.arange(16, dtype=np.float32).reshape(4, 4) / 16.0, name="w"
    )
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="matmul")
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.save(model, str(path))
    return path


def _io_elem_types(path: Path) -> tuple[int, int]:
    model = onnx.load(str(path))
    return (
        model.graph.input[0].type.tensor_type.elem_type,
        model.graph.output[0].type.tensor_type.elem_type,
    )


def _initializer_elem_types(path: Path) -> set[int]:
    model = onnx.load(str(path))
    return {init.data_type for init in model.graph.initializer}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


ULTRALYTICS_FP16 = TARGETS["yolo11n-640-fp16"]
RFDETR_FP16 = TARGETS["rfdetr-s-512-fp16"]


# --------------------------------------------------------------------------
# the descriptor is data
# --------------------------------------------------------------------------

class TestTargetMatrix:
    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_every_shipped_target_resolves(self, name: str) -> None:
        """No entry in the matrix is a combination the code cannot run."""
        resolved = plan(TARGETS[name])
        assert resolved.target is TARGETS[name]
        assert resolved.describe().startswith(name)

    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_every_target_names_a_known_backend(self, name: str) -> None:
        assert TARGETS[name].backend in BACKENDS

    def test_target_names_match_their_keys(self) -> None:
        for key, target in TARGETS.items():
            assert target.name == key

    def test_shipped_map_points_at_real_targets(self) -> None:
        for (arch, precision), name in SHIPPED.items():
            assert name in TARGETS, f"{arch}/{precision} -> missing {name}"
            assert TARGETS[name].architecture == arch
            assert TARGETS[name].precision.value == precision

    def test_a_third_target_needed_no_new_code(self) -> None:
        """The acceptance criterion, asserted.

        ``yolo11n-640-fp16-io-fp32`` is a boundary ultralytics cannot
        produce. It exists purely as a dict entry, and it plans.
        """
        target = TARGETS["yolo11n-640-fp16-io-fp32"]
        resolved = plan(target)
        assert target.backend == "ultralytics"
        assert resolved.mechanism is Mechanism.ONNX_GRAPH_FP16
        assert resolved.keep_io_types is True
        # ...and the same backend's native mechanism cannot do it.
        with pytest.raises(UnsupportedTargetError):
            plan(target.derive(mechanism=Mechanism.BACKEND_NATIVE))

    def test_descriptor_is_frozen(self) -> None:
        with pytest.raises(Exception):
            RFDETR_FP16.opset = 18  # type: ignore[misc]

    def test_derive_is_how_a_cli_flag_reaches_the_matrix(self) -> None:
        derived = RFDETR_FP16.derive(input_shape=(1, 3, 640, 640), opset=18)
        assert derived.input_shape == (1, 3, 640, 640)
        assert derived.opset == 18
        # unchanged fields carry through, and the original is untouched
        assert derived.precision is RFDETR_FP16.precision
        assert derived.io_precision is RFDETR_FP16.io_precision
        assert RFDETR_FP16.input_shape == (1, 3, 512, 512)
        assert RFDETR_FP16.opset == 17

    def test_target_for_applies_overrides(self) -> None:
        target = target_for("rfdetr", "fp16", input_shape=(1, 3, 1024, 1024))
        assert target.input_edge == 1024
        assert target.precision is Precision.FP16


class TestSizeBudget:
    def test_budget_is_data_not_a_branch(self) -> None:
        # The browser-served YOLO artifact has a ceiling; the
        # Lambda-served RF-DETR one does not.
        assert TARGETS["yolo11n-640-fp32"].max_artifact_bytes == 6 * 1024 * 1024
        assert TARGETS["rfdetr-s-512-fp32"].max_artifact_bytes is None

    def test_under_budget_returns_size(self, tmp_path: Path) -> None:
        artifact = tmp_path / "a.onnx"
        artifact.write_bytes(b"x" * 1024)
        assert TARGETS["yolo11n-640-fp32"].check_artifact_size(artifact) == 1024

    def test_over_budget_raises(self, tmp_path: Path) -> None:
        artifact = tmp_path / "a.onnx"
        artifact.write_bytes(b"x" * 32)
        target = TARGETS["yolo11n-640-fp32"].derive(max_artifact_bytes=16)
        with pytest.raises(UnsupportedTargetError, match="caps at"):
            target.check_artifact_size(artifact)

    def test_no_budget_never_raises(self, tmp_path: Path) -> None:
        artifact = tmp_path / "a.onnx"
        artifact.write_bytes(b"x" * 4096)
        assert TARGETS["rfdetr-s-512-fp32"].check_artifact_size(artifact) == 4096


class TestInputShapeGate:
    def test_non_divisible_edge_is_refused(self) -> None:
        with pytest.raises(UnsupportedTargetError, match="divisible by 32"):
            plan(RFDETR_FP16.derive(input_shape=(1, 3, 500, 500)))

    def test_non_square_is_refused(self) -> None:
        with pytest.raises(UnsupportedTargetError, match="not.*square"):
            plan(RFDETR_FP16.derive(input_shape=(1, 3, 512, 640)))

    def test_non_rgb_is_refused(self) -> None:
        with pytest.raises(UnsupportedTargetError, match="input channels"):
            plan(RFDETR_FP16.derive(input_shape=(1, 1, 512, 512)))

    def test_unknown_backend_is_refused(self) -> None:
        with pytest.raises(UnsupportedTargetError, match="unknown backend"):
            plan(RFDETR_FP16.derive(backend="tensorrt"))


# --------------------------------------------------------------------------
# the IO boundary is an explicit parameter — the point of the module
# --------------------------------------------------------------------------

class TestIOBoundaryIsExplicit:
    def test_rfdetr_fp16_declares_an_fp32_boundary(self) -> None:
        """What used to be ``keep_io_types=True`` hard-coded in a private
        helper is now a field, and it resolves to the same value."""
        resolved = plan(RFDETR_FP16)
        assert RFDETR_FP16.io_precision is IOPrecision.FP32
        assert resolved.keep_io_types is True

    def test_the_boundary_is_a_parameter_not_a_constant(self) -> None:
        """Flip the descriptor's field, and the plan flips with it.

        This is the assertion that distinguishes "we moved the constant"
        from "the boundary is a decision".
        """
        matched = RFDETR_FP16.derive(io_precision=IOPrecision.MATCH_COMPUTE)
        assert plan(matched).keep_io_types is False

    def test_ultralytics_fp16_declares_the_boundary_it_actually_gets(self) -> None:
        """ultralytics halves the module before tracing, so the graph's
        IO is fp16. Previously nothing recorded that; now the descriptor
        does, and ``keep_io_types`` is not applicable."""
        resolved = plan(ULTRALYTICS_FP16)
        assert ULTRALYTICS_FP16.io_precision is IOPrecision.MATCH_COMPUTE
        assert resolved.mechanism is Mechanism.BACKEND_NATIVE
        assert resolved.keep_io_types is None
        assert resolved.backend_kwargs == {"half": True}
        assert resolved.post_export is False

    def test_the_two_exporters_no_longer_disagree_by_accident(self) -> None:
        """Both boundaries are still different — the artifacts genuinely
        differ — but each is now declared, and the difference is one
        readable field rather than two unrelated code paths."""
        assert RFDETR_FP16.io_precision is not ULTRALYTICS_FP16.io_precision
        assert {RFDETR_FP16.io_precision, ULTRALYTICS_FP16.io_precision} == {
            IOPrecision.FP32,
            IOPrecision.MATCH_COMPUTE,
        }


# --------------------------------------------------------------------------
# refusals — every one of these must fire, loudly
# --------------------------------------------------------------------------

class TestUnsupportedCombinationsAreRejected:
    def test_backend_native_cannot_hold_an_fp32_boundary(self) -> None:
        """The refusal this whole module exists for.

        Asking ultralytics for ``half=True`` and an fp32 IO boundary is
        not something it can do. Before, you got an fp16 boundary and no
        signal. Now it raises, and the message names both the boundary
        you get and the mechanism that would give you the one you asked
        for.
        """
        bad = ULTRALYTICS_FP16.derive(io_precision=IOPrecision.FP32)
        with pytest.raises(UnsupportedTargetError) as exc:
            plan(bad)
        message = str(exc.value)
        assert "ultralytics" in message
        assert "match-compute" in message          # what you actually get
        assert "ONNX_GRAPH_FP16" in message        # how to get what you asked for

    def test_backend_without_a_half_switch_cannot_convert_itself(self) -> None:
        bad = RFDETR_FP16.derive(mechanism=Mechanism.BACKEND_NATIVE)
        with pytest.raises(UnsupportedTargetError, match="no half-precision switch"):
            plan(bad)

    def test_fp32_with_a_conversion_mechanism_is_refused(self) -> None:
        bad = TARGETS["rfdetr-s-512-fp32"].derive(
            mechanism=Mechanism.ONNX_GRAPH_FP16
        )
        with pytest.raises(UnsupportedTargetError, match="needs no conversion"):
            plan(bad)

    def test_fp32_cannot_ask_for_a_non_fp32_boundary(self) -> None:
        bad = TARGETS["rfdetr-s-512-fp32"].derive(
            io_precision=IOPrecision.MATCH_COMPUTE
        )
        with pytest.raises(UnsupportedTargetError, match="boundary is fp32"):
            plan(bad)

    def test_fp16_with_no_mechanism_is_refused(self) -> None:
        bad = RFDETR_FP16.derive(mechanism=Mechanism.NONE)
        with pytest.raises(UnsupportedTargetError, match="does not produce fp16"):
            plan(bad)

    def test_fp16_via_a_quantization_mechanism_is_refused(self) -> None:
        bad = RFDETR_FP16.derive(mechanism=Mechanism.ORT_DYNAMIC_QUANT)
        with pytest.raises(UnsupportedTargetError, match="does not produce fp16"):
            plan(bad)

    def test_int8_without_calibration_is_refused(self) -> None:
        """Preserves both exporters' pre-existing refusal, from one place."""
        bad = RFDETR_FP16.derive(precision=Precision.INT8)
        with pytest.raises(NotImplementedError, match="calibration"):
            plan(bad)

    def test_int8_refusal_names_the_mechanism_that_does_work(self) -> None:
        bad = ULTRALYTICS_FP16.derive(
            precision=Precision.INT8, io_precision=IOPrecision.FP32
        )
        with pytest.raises(NotImplementedError) as exc:
            plan(bad)
        assert "ORT_DYNAMIC_QUANT" in str(exc.value)

    def test_dynamic_quant_boundary_is_float_and_says_so(self) -> None:
        bad = TARGETS["rfdetr-s-512-int8-dynamic"].derive(
            io_precision=IOPrecision.MATCH_COMPUTE
        )
        with pytest.raises(UnsupportedTargetError, match="leaves the graph boundary"):
            plan(bad)

    def test_unknown_precision_string(self) -> None:
        with pytest.raises(ValueError, match="unknown precision"):
            Precision.parse("bf16")

    def test_unknown_target_name(self) -> None:
        with pytest.raises(UnsupportedTargetError, match="unknown target"):
            get_target("rfdetr-s-512-fp8")

    def test_unmapped_architecture_precision_pair(self) -> None:
        with pytest.raises(UnsupportedTargetError, match="no shipped target"):
            target_for("tensorrt-detr", "fp16")

    def test_target_for_int8_is_not_wired_for_either_exporter(self) -> None:
        for architecture in ("rfdetr", "yolov8"):
            with pytest.raises(NotImplementedError, match="calibration"):
                target_for(architecture, "int8")

    def test_refusals_are_value_errors_so_the_clis_still_exit_1(self) -> None:
        """Both exporters' ``main()`` catches
        ``(ValueError, NotImplementedError, RuntimeError)``. A refusal
        that escaped that tuple would traceback at users instead of
        printing the message."""
        assert issubclass(UnsupportedTargetError, ValueError)


# --------------------------------------------------------------------------
# the conversion itself, against a real graph
# --------------------------------------------------------------------------

class TestConvertGraphFp16:
    def test_keep_io_types_true_holds_an_fp32_boundary(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxconverter_common")
        from onnx import TensorProto

        src = _tiny_fp32_model(tmp_path / "src.onnx")
        dst = tmp_path / "dst.onnx"
        target = RFDETR_FP16.derive(
            name="tiny-fp16-io-fp32", input_shape=(1, 3, 32, 32)
        )
        convert(plan(target), src, dst)

        assert _io_elem_types(dst) == (TensorProto.FLOAT, TensorProto.FLOAT)
        # ...while the interior really did convert.
        assert _initializer_elem_types(dst) == {TensorProto.FLOAT16}

    def test_keep_io_types_false_moves_the_boundary(self, tmp_path: Path) -> None:
        """The same code path, the same model, one descriptor field
        different — and the artifact a consumer must feed changes."""
        pytest.importorskip("onnxconverter_common")
        from onnx import TensorProto

        src = _tiny_fp32_model(tmp_path / "src.onnx")
        dst = tmp_path / "dst.onnx"
        target = RFDETR_FP16.derive(
            name="tiny-fp16-io-fp16",
            input_shape=(1, 3, 32, 32),
            io_precision=IOPrecision.MATCH_COMPUTE,
        )
        convert(plan(target), src, dst)

        assert _io_elem_types(dst) == (TensorProto.FLOAT16, TensorProto.FLOAT16)
        assert _initializer_elem_types(dst) == {TensorProto.FLOAT16}

    def test_source_is_never_modified(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxconverter_common")
        src = _tiny_fp32_model(tmp_path / "src.onnx")
        before = _sha256(src)
        target = RFDETR_FP16.derive(input_shape=(1, 3, 32, 32))
        convert(plan(target), src, tmp_path / "dst.onnx")
        assert _sha256(src) == before

    def test_in_place_conversion_when_dst_omitted(self, tmp_path: Path) -> None:
        """``export_rfdetr._to_fp16`` converts in place; that has to keep
        working."""
        pytest.importorskip("onnxconverter_common")
        from onnx import TensorProto

        path = _tiny_fp32_model(tmp_path / "m.onnx")
        target = RFDETR_FP16.derive(input_shape=(1, 3, 32, 32))
        result = convert(plan(target), path)
        assert result.output == path
        assert _initializer_elem_types(path) == {TensorProto.FLOAT16}

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert(plan(RFDETR_FP16), tmp_path / "absent.onnx", tmp_path / "o.onnx")

    def test_fp32_target_is_a_byte_exact_passthrough(self, tmp_path: Path) -> None:
        src = _tiny_fp32_model(tmp_path / "src.onnx")
        dst = tmp_path / "dst.onnx"
        target = TARGETS["rfdetr-s-512-fp32"].derive(input_shape=(1, 3, 32, 32))
        result = convert(plan(target), src, dst)
        assert result.plan.post_export is False
        assert _sha256(dst) == _sha256(src)

    def test_result_is_recordable(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxconverter_common")
        src = _tiny_fp32_model(tmp_path / "src.onnx")
        target = RFDETR_FP16.derive(input_shape=(1, 3, 32, 32))
        result = convert(plan(target), src, tmp_path / "dst.onnx")
        record = result.as_dict()
        assert record["mechanism"] == "onnx-graph-fp16"
        assert record["ioPrecision"] == "fp32"
        assert record["keepIoTypes"] is True
        assert record["outputBytes"] > 0


class TestRepairAndValidateAreDescriptorFields:
    def test_repair_off_leaves_the_converter_output_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("onnxconverter_common")
        called: list[str] = []
        import export.fp16_repair as fp16_repair

        monkeypatch.setattr(
            fp16_repair,
            "find_duplicate_node_names",
            lambda model: called.append("repair") or {},
        )
        src = _tiny_fp32_model(tmp_path / "src.onnx")
        target = RFDETR_FP16.derive(input_shape=(1, 3, 32, 32), repair=False)
        result = convert(plan(target), src, tmp_path / "dst.onnx")
        assert called == []
        assert result.repaired is False

    def test_repair_on_runs_the_repair_pass(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxconverter_common")
        src = _tiny_fp32_model(tmp_path / "src.onnx")
        target = RFDETR_FP16.derive(input_shape=(1, 3, 32, 32), repair=True)
        result = convert(plan(target), src, tmp_path / "dst.onnx")
        # The tiny graph has neither defect, so the pass is a clean no-op
        # — which is itself the assertion: repair does not invent work.
        assert result.repaired is False
        assert result.duplicate_node_names == {}

    def test_explicit_override_beats_the_descriptor(self, tmp_path: Path) -> None:
        """``scripts/make_fp16.py --no-repair`` reproduces the load
        failure on purpose, so the override has to win."""
        pytest.importorskip("onnxconverter_common")
        src = _tiny_fp32_model(tmp_path / "src.onnx")
        target = RFDETR_FP16.derive(input_shape=(1, 3, 32, 32), repair=True)
        result = convert(plan(target), src, tmp_path / "dst.onnx", repair=False)
        assert result.repaired is False

    def test_validate_gate_fires_on_an_unloadable_artifact(
        self, tmp_path: Path
    ) -> None:
        """Prove the gate is a gate, through ``convert()`` itself.

        The target here is fp32, i.e. a pure passthrough with no
        conversion work — which is exactly the case where an early
        return would have skipped the gate and let a broken artifact
        through reporting success.
        """
        src = tmp_path / "src.onnx"
        src.write_bytes(b"this is not a protobuf")
        target = TARGETS["rfdetr-s-512-fp32"].derive(
            input_shape=(1, 3, 32, 32), validate=True
        )
        with pytest.raises(Exception, match="(?i)fail|invalid|protobuf|load"):
            convert(plan(target), src, tmp_path / "dst.onnx")

    def test_passthrough_without_the_gate_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """The control for the test above: same broken bytes, gate off,
        so the failure it reports is the gate and not the copy."""
        src = tmp_path / "src.onnx"
        src.write_bytes(b"this is not a protobuf")
        target = TARGETS["rfdetr-s-512-fp32"].derive(
            input_shape=(1, 3, 32, 32), validate=False
        )
        result = convert(plan(target), src, tmp_path / "dst.onnx")
        assert result.validated is False

    def test_validate_passes_on_a_real_graph(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxconverter_common")
        src = _tiny_fp32_model(tmp_path / "src.onnx")
        target = RFDETR_FP16.derive(
            input_shape=(1, 3, 32, 32), repair=True, validate=True
        )
        result = convert(plan(target), src, tmp_path / "dst.onnx")
        assert result.validated is True


# --------------------------------------------------------------------------
# the shipped recipes are pinned, because their bytes are
# --------------------------------------------------------------------------

class TestShippedRecipesArePinned:
    """The refactor's hard constraint, in testable form.

    ``dist/parity/rfdetr-s-litter.fp16.onnx`` was verified byte-identical
    before and after this refactor
    (sha256 ccdddc43...4ce7faab, 63,068,681 bytes). What produced those
    bytes is the tuple below. A change to any field changes the artifact,
    so a change to any field should have to be deliberate enough to
    update this test.
    """

    def test_exporter_fp16_recipe(self) -> None:
        target = TARGETS["rfdetr-s-512-fp16"]
        resolved = plan(target)
        assert resolved.mechanism is Mechanism.ONNX_GRAPH_FP16
        assert resolved.keep_io_types is True
        assert target.repair is False
        assert target.validate is False
        assert target.opset == 17
        assert target.input_shape == (1, 3, 512, 512)

    def test_make_fp16_recipe(self) -> None:
        target = TARGETS["rfdetr-s-512-fp16-repaired"]
        resolved = plan(target)
        assert resolved.mechanism is Mechanism.ONNX_GRAPH_FP16
        assert resolved.keep_io_types is True
        assert target.repair is True
        assert target.validate is True

    def test_the_two_differ_only_in_repair_and_validate(self) -> None:
        """The one real discrepancy between the exporter and the script,
        left in place rather than silently resolved: fixing it changes an
        artifact's bytes, which is a release decision."""
        import dataclasses

        shipped = dataclasses.asdict(TARGETS["rfdetr-s-512-fp16"])
        repaired = dataclasses.asdict(TARGETS["rfdetr-s-512-fp16-repaired"])
        differing = {
            k for k in shipped if shipped[k] != repaired[k]
        } - {"name", "notes"}
        assert differing == {"repair", "validate"}

    def test_make_int8_recipe(self) -> None:
        target = TARGETS["rfdetr-s-512-int8-dynamic"]
        resolved = plan(target)
        assert resolved.mechanism is Mechanism.ORT_DYNAMIC_QUANT
        assert target.quant_weight_type == "int8"
        assert target.validate is True


# --------------------------------------------------------------------------
# the path is actually shared — not two paths with a shared name
# --------------------------------------------------------------------------

class TestBothExportersUseTheSharedPath:
    def test_only_precision_module_calls_the_fp16_converter(self) -> None:
        """The structural claim: one place converts precision.

        If a second module starts *calling* ``convert_float_to_float16``
        or ``quantize_dynamic``, the one-off pattern is back and this
        fails. Matched on the AST rather than the text, so that prose
        about the converter — which several modules legitimately contain
        — does not count as a second conversion path.
        """
        import ast

        converters = {"convert_float_to_float16", "quantize_dynamic"}
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for path in sorted(
            list((root / "export").rglob("*.py"))
            + list((root / "scripts").rglob("*.py"))
        ):
            if path.name == "precision.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = (
                        func.attr
                        if isinstance(func, ast.Attribute)
                        else func.id
                        if isinstance(func, ast.Name)
                        else None
                    )
                    if name in converters:
                        offenders.append(f"{path.relative_to(root)}: calls {name}")
                elif isinstance(node, ast.ImportFrom) and node.module and (
                    "onnxconverter_common" in node.module
                    or "onnxruntime.quantization" in node.module
                ):
                    imported = ", ".join(a.name for a in node.names)
                    offenders.append(
                        f"{path.relative_to(root)}: imports {imported} "
                        f"from {node.module}"
                    )
        assert offenders == []

    def test_yolo_kwargs_helper_delegates(self) -> None:
        from export.export_yolov8 import _precision_to_export_kwargs

        assert _precision_to_export_kwargs("fp32") == dict(
            plan(target_for("yolov8", "fp32")).backend_kwargs
        )
        assert _precision_to_export_kwargs("fp16") == dict(
            plan(target_for("yolov8", "fp16")).backend_kwargs
        )

    def test_yolo_size_budget_comes_from_the_descriptor(self) -> None:
        from export.export_yolov8 import MAX_ONNX_BYTES

        assert MAX_ONNX_BYTES == TARGETS["yolo11n-640-fp32"].max_artifact_bytes

    def test_rfdetr_to_fp16_shim_delegates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shim must run the *shipped* descriptor, not a fresh set of
        arguments that could drift from it."""
        import export.precision as precision_module
        from export.export_rfdetr import _to_fp16

        seen: dict[str, object] = {}

        def _spy(plan_, src, dst=None, **kwargs):
            seen["target"] = plan_.target.name
            seen["keep_io_types"] = plan_.keep_io_types
            seen["src"] = src
            return ConversionResult(
                plan=plan_, source=Path(src), output=Path(src), output_bytes=0
            )

        monkeypatch.setattr(precision_module, "convert", _spy)
        artifact = tmp_path / "m.onnx"
        artifact.write_bytes(b"placeholder")
        _to_fp16(artifact)

        assert seen["target"] == "rfdetr-s-512-fp16"
        assert seen["keep_io_types"] is True
        assert seen["src"] == artifact


class TestBackendSurfaceStaysThin:
    def test_backends_contribute_facts_not_behaviour(self) -> None:
        """A backend is three facts and one derived kwarg dict. If it
        grows behaviour, precision policy has leaked back out."""
        for backend in BACKENDS.values():
            assert isinstance(backend, Backend)
            assert backend.default_fp16_mechanism in Mechanism
            if backend.half_kwarg is None:
                assert backend.native_io_precision is None
            else:
                assert backend.native_io_precision in IOPrecision

    def test_kwargs_for_a_backend_without_a_switch_is_empty(self) -> None:
        assert BACKENDS["rfdetr"].kwargs_for(half=True) == {}
        assert BACKENDS["rfdetr"].kwargs_for(half=False) == {}

    def test_kwargs_for_ultralytics(self) -> None:
        assert BACKENDS["ultralytics"].kwargs_for(half=True) == {"half": True}
        assert BACKENDS["ultralytics"].kwargs_for(half=False) == {"half": False}

    def test_graph_mechanism_asks_the_backend_for_fp32(self) -> None:
        """When the graph pass owns the conversion, the backend must be
        told *not* to convert — otherwise it happens twice."""
        resolved = plan(TARGETS["yolo11n-640-fp16-io-fp32"])
        assert resolved.backend_kwargs == {"half": False}
        assert resolved.post_export is True


def test_export_target_is_importable_from_the_package() -> None:
    """Cheap guard that the public names stay public."""
    assert ExportTarget.__module__ == "export.precision"
