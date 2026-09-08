"""The one place a precision conversion happens, and the descriptor that names one.

Before this module the repository converted precision in two places that
shared no code and, more importantly, made two different decisions about
the **IO boundary** without either of them being written down as a
decision:

*   ``export_rfdetr._to_fp16`` called ``onnxconverter_common.float16``
    with ``keep_io_types=True``. The graph computes in fp16 and its
    inputs/outputs stay fp32. That *is* a decision, and a deliberate one
    — the backend's ORT session feeds float32 — but it lived as a
    keyword argument in a private helper.
*   ``export_yolov8._precision_to_export_kwargs`` mapped fp16 to
    ultralytics' ``{'half': True}``. Ultralytics halves the torch module
    before tracing, so the exported graph's inputs and outputs are
    **fp16 too**. Nobody chose that. It is what the library does, and
    the boundary a consumer must feed therefore depended on which
    exporter ran.

Those are different artifacts in a way that matters to the consumer, and
nothing in the codebase said so. Here the boundary is a field
(:class:`IOPrecision`) on a descriptor, the mechanism that realises it is
a field (:class:`Mechanism`), and :func:`plan` refuses combinations the
backend cannot actually produce instead of silently giving you a
different boundary than the one you asked for.

The shape of the thing
----------------------

``ExportTarget``  — a target is **data**: precision, IO precision, opset,
input shape, size budget, backend, architecture. Adding a target is a new
entry in :data:`TARGETS`, not a new script or a new branch. ``TARGETS``
already carries a third entry that exists only to demonstrate this: an
ultralytics model converted through the *graph* path to get the fp32 IO
boundary ultralytics' own ``half=True`` cannot give you. It is a
descriptor and nothing else.

``plan(target)``  — resolves a descriptor into a
:class:`PrecisionPlan`: what the exporting backend must be told
(``backend_kwargs``), whether a post-export conversion is owed
(``post_export``), and — for the graph path — the explicit
``keep_io_types`` that realises the requested boundary. This is the only
function that decides how a precision is realised. It validates loudly:
see :class:`UnsupportedTargetError`.

``convert(plan, src, dst)``  — the only implementation of a precision
conversion. Backends stay thin: they export fp32 (or take the one kwarg
``plan`` hands them) and know nothing about precision policy.

What this module deliberately does not do
-----------------------------------------

It does not change what any currently-shipping recipe produces. The
fp16 sequence below (load → convert → save → load → repair → save-if-
changed) reproduces ``scripts/make_fp16.py`` step for step, and the
artifact it writes is byte-identical to the pre-refactor one. Where the
two shipping recipes disagreed — ``export_rfdetr.export()`` does not run
``export.fp16_repair`` and does not validate, ``scripts/make_fp16.py``
does both — that disagreement is preserved and made visible as two
descriptors (``rfdetr-s-512-fp16`` and ``rfdetr-s-512-fp16-repaired``)
rather than quietly resolved. Resolving it changes an artifact's bytes,
which is a release decision, not a refactor.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

LOG = logging.getLogger("export.precision")


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

class Precision(str, Enum):
    """The compute precision of the graph's weights and arithmetic."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"

    @classmethod
    def parse(cls, raw: str) -> "Precision":
        try:
            return cls(raw)
        except ValueError as exc:
            raise ValueError(
                f"unknown precision {raw!r}; expected one of "
                f"{', '.join(p.value for p in cls)}"
            ) from exc


class IOPrecision(str, Enum):
    """The dtype of the graph's *inputs and outputs*, which is a separate
    decision from the compute precision and the one consumers feel.

    ``FP32`` means the conversion holds an fp32 boundary around an
    otherwise-converted graph — the caller feeds and reads float32 and
    does not have to know the interior precision changed. ``MATCH_COMPUTE``
    means the boundary moved with the interior and the caller must now
    feed the new dtype.
    """

    FP32 = "fp32"
    MATCH_COMPUTE = "match-compute"


class Mechanism(str, Enum):
    """*How* a precision is realised. Different mechanisms are not
    interchangeable: they run at different points (before vs. after the
    graph exists) and they do not all support every boundary."""

    #: Nothing to do — the backend's natural output is already the target.
    NONE = "none"
    #: ``onnxconverter_common.float16.convert_float_to_float16`` on the
    #: exported graph. Supports either boundary via ``keep_io_types``.
    ONNX_GRAPH_FP16 = "onnx-graph-fp16"
    #: The exporting library converts during export (ultralytics
    #: ``half=True``). The boundary is whatever that library does.
    BACKEND_NATIVE = "backend-native"
    #: ``onnxruntime.quantization.quantize_dynamic`` on the exported
    #: graph. Weights quantized ahead of time, activation scales chosen
    #: at run time, so no calibration set — and the boundary stays float.
    ORT_DYNAMIC_QUANT = "ort-dynamic-quant"


class UnsupportedTargetError(ValueError):
    """A descriptor asks for something its backend/mechanism cannot do.

    Subclasses ``ValueError`` so the exporters' existing
    ``except (ValueError, NotImplementedError, RuntimeError)`` in
    ``main()`` keeps catching it and exits 1 rather than tracebacking.
    """


# --------------------------------------------------------------------------
# backends — thin, and only describe what the library can be asked for
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Backend:
    """What an exporting library can and cannot be told about precision.

    This is the whole of the "pluggable backend" surface as far as
    precision is concerned. A backend contributes three facts and no
    behaviour: the kwarg that asks it for half precision (if any), the
    IO boundary it produces when it does its own conversion, and the
    mechanism it reaches for by default.
    """

    name: str
    #: Kwarg name that asks the library to export in half precision.
    #: ``None`` means the library has no such switch and a non-fp32
    #: target must be reached by converting the graph afterwards.
    half_kwarg: str | None
    #: The boundary the library's own half-precision export produces.
    #: ``None`` when it has no such export.
    native_io_precision: IOPrecision | None
    #: The mechanism used when a descriptor does not name one.
    default_fp16_mechanism: Mechanism

    def kwargs_for(self, half: bool) -> dict[str, Any]:
        """What to pass the library's ``export()``. Empty when the
        library has no precision switch — the graph path handles it."""
        if self.half_kwarg is None:
            return {}
        return {self.half_kwarg: half}


BACKENDS: dict[str, Backend] = {
    # ultralytics halves the torch module before tracing, so the exported
    # graph's IO is fp16 as well. Verified against the kwarg contract in
    # export_yolov8._precision_to_export_kwargs, which this replaces.
    "ultralytics": Backend(
        name="ultralytics",
        half_kwarg="half",
        native_io_precision=IOPrecision.MATCH_COMPUTE,
        default_fp16_mechanism=Mechanism.BACKEND_NATIVE,
    ),
    # rfdetr's export() takes no precision argument at all; it emits
    # fp32 and any other precision is a post-export graph conversion.
    "rfdetr": Backend(
        name="rfdetr",
        half_kwarg=None,
        native_io_precision=None,
        default_fp16_mechanism=Mechanism.ONNX_GRAPH_FP16,
    ),
}


# --------------------------------------------------------------------------
# the descriptor
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportTarget:
    """One conversion target, as data.

    The point of this object is that adding a target is adding one of
    these. If a change to the target matrix requires editing a function
    body, the abstraction has failed and should be fixed here rather
    than worked around at the call site.
    """

    name: str
    architecture: str            # decode family: "rfdetr" | "yolov8"
    backend: str                 # key into BACKENDS — the exporting library
    precision: Precision
    io_precision: IOPrecision
    opset: int
    input_shape: tuple[int, int, int, int]   # NCHW
    #: ``None`` means no ceiling. The YOLO artifact serves from a
    #: browser and has one; the RF-DETR artifact serves from Lambda and
    #: does not.
    max_artifact_bytes: int | None = None
    #: Input edge must be divisible by this (patch_size * num_windows
    #: for RF-DETR; stride for YOLO).
    input_multiple_of: int = 32
    #: ``None`` selects the backend's default for this precision.
    mechanism: Mechanism | None = None
    #: Weight type for ORT_DYNAMIC_QUANT. Ignored by other mechanisms.
    quant_weight_type: str = "int8"
    #: Run export.fp16_repair on the converted graph.
    repair: bool = False
    #: Build an ORT session on the result and fail if it does not load.
    #: ``onnx.checker`` is not a substitute — it accepts graphs ORT
    #: rejects, which is how two defects reached this repo unnoticed.
    validate: bool = False
    notes: str = ""

    @property
    def backend_spec(self) -> Backend:
        try:
            return BACKENDS[self.backend]
        except KeyError as exc:
            raise UnsupportedTargetError(
                f"target {self.name!r} names unknown backend "
                f"{self.backend!r}; known backends: "
                f"{', '.join(sorted(BACKENDS))}"
            ) from exc

    @property
    def input_edge(self) -> int:
        return self.input_shape[-1]

    def derive(self, **overrides: Any) -> "ExportTarget":
        """A new descriptor differing in the named fields. This is how a
        CLI flag (``--imgsz``, ``--opset``) reaches the target matrix
        without the matrix growing a branch."""
        return replace(self, **overrides)

    def check_input_shape(self) -> None:
        _, channels, height, width = self.input_shape
        if channels != 3:
            raise UnsupportedTargetError(
                f"target {self.name!r} has {channels} input channels; "
                f"the export path assumes 3 (RGB)"
            )
        if height != width:
            raise UnsupportedTargetError(
                f"target {self.name!r} input {height}x{width} is not "
                f"square; the exporters take a single edge length"
            )
        if self.input_multiple_of and height % self.input_multiple_of:
            raise UnsupportedTargetError(
                f"target {self.name!r} input edge {height} is not "
                f"divisible by {self.input_multiple_of}"
            )

    def check_artifact_size(self, path: Path) -> int:
        """Enforce the size budget. Returns the size in bytes."""
        size = path.stat().st_size
        if self.max_artifact_bytes is not None and size > self.max_artifact_bytes:
            raise UnsupportedTargetError(
                f"{path} is {size / 1024 / 1024:.2f} MB; target "
                f"{self.name!r} caps at "
                f"{self.max_artifact_bytes / 1024 / 1024:.0f} MB. "
                f"Use a smaller precision or revisit the input size."
            )
        return size


# --------------------------------------------------------------------------
# the plan — the single place a precision decision is made
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecisionPlan:
    """A validated, executable reading of a descriptor."""

    target: ExportTarget
    mechanism: Mechanism
    #: What to hand the exporting library's ``export()``.
    backend_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: Whether :func:`convert` has work to do after the backend runs.
    post_export: bool = False
    #: The explicit IO boundary for the graph fp16 path. ``None`` when
    #: the mechanism is not ``ONNX_GRAPH_FP16``. This is the value that
    #: used to be a hard-coded keyword argument in a private helper.
    keep_io_types: bool | None = None

    def describe(self) -> str:
        boundary = (
            f"io={self.target.io_precision.value}"
            if self.keep_io_types is None
            else f"io={self.target.io_precision.value} "
                 f"(keep_io_types={self.keep_io_types})"
        )
        return (
            f"{self.target.name}: {self.target.precision.value} via "
            f"{self.mechanism.value}, {boundary}"
        )


_INT8_NOT_WIRED = (
    "INT8 export via {mechanism} requires a static calibration pass to "
    "pick activation scales, and none is wired here. Two things that "
    "are: Mechanism.ORT_DYNAMIC_QUANT (weights quantized ahead of time, "
    "activation scales chosen at run time, so no calibration set is "
    "needed — see scripts/make_int8.py), or fp16."
)


def plan(target: ExportTarget) -> PrecisionPlan:
    """Resolve a descriptor into an executable plan, or refuse.

    This is the only function that decides how a precision is realised.
    Every refusal below is a combination that would otherwise have
    silently produced an artifact with a different IO boundary than the
    one the descriptor asked for.
    """
    backend = target.backend_spec
    target.check_input_shape()
    mechanism = target.mechanism or _default_mechanism(target, backend)

    if target.precision is Precision.FP32:
        if mechanism is not Mechanism.NONE:
            raise UnsupportedTargetError(
                f"target {target.name!r} is fp32 but names mechanism "
                f"{mechanism.value!r}; fp32 is the backends' native "
                f"output and needs no conversion"
            )
        if target.io_precision is not IOPrecision.FP32:
            raise UnsupportedTargetError(
                f"target {target.name!r} is fp32 but asks for io "
                f"{target.io_precision.value!r}; an fp32 graph's "
                f"boundary is fp32"
            )
        return PrecisionPlan(
            target=target,
            mechanism=Mechanism.NONE,
            backend_kwargs=backend.kwargs_for(half=False),
            post_export=False,
        )

    if target.precision is Precision.FP16:
        if mechanism is Mechanism.ONNX_GRAPH_FP16:
            # Both boundaries are reachable here, which is the whole
            # reason this mechanism is worth having: keep_io_types is an
            # argument, so the boundary is a choice.
            return PrecisionPlan(
                target=target,
                mechanism=mechanism,
                # The backend exports fp32; the graph pass does the rest.
                backend_kwargs=backend.kwargs_for(half=False),
                post_export=True,
                keep_io_types=target.io_precision is IOPrecision.FP32,
            )
        if mechanism is Mechanism.BACKEND_NATIVE:
            if backend.half_kwarg is None:
                raise UnsupportedTargetError(
                    f"target {target.name!r} asks backend "
                    f"{backend.name!r} to convert to fp16 itself, but it "
                    f"has no half-precision switch. Use "
                    f"Mechanism.ONNX_GRAPH_FP16 to convert the exported "
                    f"graph instead."
                )
            if target.io_precision is not backend.native_io_precision:
                # The refusal that motivates this module. Asking
                # ultralytics for half=True and expecting an fp32
                # boundary gets you an fp16 boundary and no warning.
                raise UnsupportedTargetError(
                    f"target {target.name!r} asks for io "
                    f"{target.io_precision.value!r} from backend "
                    f"{backend.name!r}'s own fp16 export, which produces "
                    f"io {backend.native_io_precision.value!r} and takes "
                    f"no argument to change that. Either declare io="
                    f"{backend.native_io_precision.value!r}, or switch "
                    f"to Mechanism.ONNX_GRAPH_FP16, which holds the "
                    f"boundary you asked for via keep_io_types."
                )
            return PrecisionPlan(
                target=target,
                mechanism=mechanism,
                backend_kwargs=backend.kwargs_for(half=True),
                post_export=False,
            )
        raise UnsupportedTargetError(
            f"target {target.name!r} is fp16 but names mechanism "
            f"{mechanism.value!r}, which does not produce fp16"
        )

    if target.precision is Precision.INT8:
        if mechanism is Mechanism.ORT_DYNAMIC_QUANT:
            if target.io_precision is not IOPrecision.FP32:
                raise UnsupportedTargetError(
                    f"target {target.name!r} asks for io "
                    f"{target.io_precision.value!r} from dynamic "
                    f"quantization, which leaves the graph boundary "
                    f"float. Declare io='fp32'."
                )
            return PrecisionPlan(
                target=target,
                mechanism=mechanism,
                backend_kwargs=backend.kwargs_for(half=False),
                post_export=True,
            )
        raise NotImplementedError(
            _INT8_NOT_WIRED.format(mechanism=mechanism.value)
        )

    raise UnsupportedTargetError(  # pragma: no cover - Precision is closed
        f"target {target.name!r} has unhandled precision {target.precision!r}"
    )


def _default_mechanism(target: ExportTarget, backend: Backend) -> Mechanism:
    if target.precision is Precision.FP32:
        return Mechanism.NONE
    if target.precision is Precision.FP16:
        return backend.default_fp16_mechanism
    # INT8 has no default: choosing one silently would be choosing a
    # quantization scheme on the caller's behalf.
    return Mechanism.NONE


# --------------------------------------------------------------------------
# the conversion itself
# --------------------------------------------------------------------------

@dataclass
class ConversionResult:
    """What a conversion did, for the caller to log or record."""

    plan: PrecisionPlan
    source: Path
    output: Path
    output_bytes: int
    duplicate_node_names: dict[str, int] = field(default_factory=dict)
    removed_degenerate_casts: list[str] = field(default_factory=list)
    retyped_float_casts: list[str] = field(default_factory=list)
    repaired: bool = False
    validated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.plan.target.name,
            "precision": self.plan.target.precision.value,
            "ioPrecision": self.plan.target.io_precision.value,
            "mechanism": self.plan.mechanism.value,
            "keepIoTypes": self.plan.keep_io_types,
            "source": str(self.source),
            "output": str(self.output),
            "outputBytes": self.output_bytes,
            "duplicateNodeNames": dict(self.duplicate_node_names),
            "removedDegenerateCasts": list(self.removed_degenerate_casts),
            "retypedFloatCasts": list(self.retyped_float_casts),
            "repaired": self.repaired,
            "validated": self.validated,
        }


def convert(
    plan_: PrecisionPlan,
    src: Path,
    dst: Path | None = None,
    *,
    repair: bool | None = None,
    validate: bool | None = None,
) -> ConversionResult:
    """Apply ``plan_``'s post-export conversion to ``src``, writing ``dst``.

    ``dst=None`` (or ``dst == src``) converts in place.

    ``repair``/``validate`` default to the descriptor's fields; pass them
    explicitly only to reproduce a recipe that differs from the target,
    which is what the legacy ``export_rfdetr._to_fp16`` shim does.

    The operation *order* below is load-bearing and not an accident of
    style: convert-and-save, then separately load-repair-and-save-if-
    changed. That is what ``scripts/make_fp16.py`` did before this
    module existed, and collapsing it into a single save would risk a
    different serialization for no benefit. The artifact this produces
    is byte-identical to the pre-refactor one; see
    reports/CONVERSION-REPORT.md.
    """
    target = plan_.target
    do_repair = target.repair if repair is None else repair
    do_validate = target.validate if validate is None else validate

    src = Path(src)
    dst = src if dst is None else Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"source artifact not found: {src}")

    in_place = dst.resolve() == src.resolve()
    if not in_place:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # ORT_DYNAMIC_QUANT reads src and writes dst itself; every other
        # path rewrites dst in place, so it needs the copy first.
        if plan_.mechanism is not Mechanism.ORT_DYNAMIC_QUANT:
            shutil.copyfile(src, dst)

    if not plan_.post_export:
        # Nothing to convert — the backend's output is already the
        # target. Deliberately NOT an early return: repair and validate
        # below are gates, and a gate that silently does not run for
        # some descriptors is the failure mode this whole module exists
        # to stop repeating.
        LOG.debug("%s: nothing to convert after export", plan_.describe())
    elif plan_.mechanism is Mechanism.ONNX_GRAPH_FP16:
        _convert_graph_fp16(dst, keep_io_types=bool(plan_.keep_io_types))
    elif plan_.mechanism is Mechanism.ORT_DYNAMIC_QUANT:
        _quantize_dynamic(src, dst, weight_type=target.quant_weight_type)
    else:  # pragma: no cover - plan() rejects everything else
        raise UnsupportedTargetError(
            f"no conversion implemented for mechanism {plan_.mechanism!r}"
        )

    result = ConversionResult(
        plan=plan_, source=src, output=dst, output_bytes=dst.stat().st_size
    )

    if do_repair:
        _apply_repair(dst, result)
    if do_validate:
        from export.fp16_repair import validate_loadable

        validate_loadable(str(dst))
        result.validated = True

    result.output_bytes = dst.stat().st_size
    return result


def _convert_graph_fp16(path: Path, *, keep_io_types: bool) -> None:
    """The graph fp16 pass. ``keep_io_types`` is the IO boundary, and it
    is a parameter here precisely because it used to be a constant."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            "fp16 conversion needs onnx + onnxconverter-common: "
            'pip install -e ".[parity]"'
        ) from exc
    model = onnx.load(str(path))
    converted = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
    onnx.save(converted, str(path))


def _quantize_dynamic(src: Path, dst: Path, *, weight_type: str = "int8") -> None:
    """ONNX Runtime dynamic quantization. Weights quantized ahead of
    time, activation scales chosen at run time — no calibration set."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    qtype = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}[weight_type]
    quantize_dynamic(str(src), str(dst), weight_type=qtype)


def _apply_repair(path: Path, result: ConversionResult) -> None:
    import onnx

    from export.fp16_repair import find_duplicate_node_names, repair as repair_model

    model = onnx.load(str(path))
    result.duplicate_node_names = find_duplicate_node_names(model)
    report = repair_model(model)
    result.removed_degenerate_casts = list(report.removed_degenerate_casts)
    result.retyped_float_casts = list(report.retyped_float_casts)
    if report.changed:
        onnx.save(model, str(path))
        result.repaired = True


# --------------------------------------------------------------------------
# the target matrix
# --------------------------------------------------------------------------

MB = 1024 * 1024

TARGETS: dict[str, ExportTarget] = {
    # --- shipped: RF-DETR-Small, Lambda-served, no size ceiling --------
    "rfdetr-s-512-fp32": ExportTarget(
        name="rfdetr-s-512-fp32",
        architecture="rfdetr",
        backend="rfdetr",
        precision=Precision.FP32,
        io_precision=IOPrecision.FP32,
        opset=17,
        input_shape=(1, 3, 512, 512),
        notes="v2.0.1-fp32, the artifact currently serving.",
    ),
    # What export_rfdetr.export() produces today. It does NOT repair and
    # does NOT validate, and reports/CONVERSION-REPORT.md §2 records that
    # the result therefore fails to load in ONNX Runtime. Preserved as-is
    # rather than fixed here: changing it changes an artifact's bytes,
    # which is a release decision. The fix is to ship the descriptor
    # below instead.
    "rfdetr-s-512-fp16": ExportTarget(
        name="rfdetr-s-512-fp16",
        architecture="rfdetr",
        backend="rfdetr",
        precision=Precision.FP16,
        io_precision=IOPrecision.FP32,
        opset=17,
        input_shape=(1, 3, 512, 512),
        mechanism=Mechanism.ONNX_GRAPH_FP16,
        repair=False,
        validate=False,
        notes=(
            "As exported by export_rfdetr.export() today. Known not to "
            "load in ORT — see reports/CONVERSION-REPORT.md §2."
        ),
    ),
    # What scripts/make_fp16.py produces: the same conversion, plus the
    # two repairs and the ORT load gate.
    "rfdetr-s-512-fp16-repaired": ExportTarget(
        name="rfdetr-s-512-fp16-repaired",
        architecture="rfdetr",
        backend="rfdetr",
        precision=Precision.FP16,
        io_precision=IOPrecision.FP32,
        opset=17,
        input_shape=(1, 3, 512, 512),
        mechanism=Mechanism.ONNX_GRAPH_FP16,
        repair=True,
        validate=True,
        notes="The measured fp16 artifact behind the parity report.",
    ),
    "rfdetr-s-512-int8-dynamic": ExportTarget(
        name="rfdetr-s-512-int8-dynamic",
        architecture="rfdetr",
        backend="rfdetr",
        precision=Precision.INT8,
        io_precision=IOPrecision.FP32,
        opset=17,
        input_shape=(1, 3, 512, 512),
        mechanism=Mechanism.ORT_DYNAMIC_QUANT,
        repair=False,
        validate=True,
        notes=(
            "Dynamic quantization, so no calibration set. Weight scales "
            "are per-channel where the op supports it; activation scales "
            "are chosen per-run. See scripts/make_int8.py."
        ),
    ),
    # --- shipped: YOLO11n, browser-served, 6 MB ceiling ---------------
    "yolo11n-640-fp32": ExportTarget(
        name="yolo11n-640-fp32",
        architecture="yolov8",
        backend="ultralytics",
        precision=Precision.FP32,
        io_precision=IOPrecision.FP32,
        opset=17,
        input_shape=(1, 3, 640, 640),
        max_artifact_bytes=6 * MB,
    ),
    "yolo11n-640-fp16": ExportTarget(
        name="yolo11n-640-fp16",
        architecture="yolov8",
        backend="ultralytics",
        precision=Precision.FP16,
        io_precision=IOPrecision.MATCH_COMPUTE,
        opset=17,
        input_shape=(1, 3, 640, 640),
        max_artifact_bytes=6 * MB,
        mechanism=Mechanism.BACKEND_NATIVE,
        notes=(
            "ultralytics half=True. io is fp16 because that is what the "
            "library does, not because it was chosen — declaring it here "
            "is the point."
        ),
    ),
    # --- the third target, and the proof of the claim ------------------
    # A YOLO fp16 artifact with an fp32 boundary. Ultralytics cannot
    # produce this; the graph mechanism can. It exists as a descriptor
    # and required no new script, no new function and no new branch.
    "yolo11n-640-fp16-io-fp32": ExportTarget(
        name="yolo11n-640-fp16-io-fp32",
        architecture="yolov8",
        backend="ultralytics",
        precision=Precision.FP16,
        io_precision=IOPrecision.FP32,
        opset=17,
        input_shape=(1, 3, 640, 640),
        max_artifact_bytes=6 * MB,
        mechanism=Mechanism.ONNX_GRAPH_FP16,
        repair=True,
        validate=True,
        notes=(
            "Not shipped, not measured. Demonstrates that a target is "
            "data: the boundary ultralytics cannot give you is one "
            "field on a descriptor."
        ),
    ),
}


#: Which descriptor each exporter reaches for, keyed by (architecture,
#: precision suffix from the version string). This is the exporters'
#: entire knowledge of precision policy: a new architecture or a new
#: precision is an entry here plus a descriptor above, not a branch in
#: an export script.
SHIPPED: dict[tuple[str, str], str] = {
    ("rfdetr", "fp32"): "rfdetr-s-512-fp32",
    ("rfdetr", "fp16"): "rfdetr-s-512-fp16",
    ("yolov8", "fp32"): "yolo11n-640-fp32",
    ("yolov8", "fp16"): "yolo11n-640-fp16",
    # Deliberately no int8 row. rfdetr-s-512-int8-dynamic exists and
    # works, but it is a post-export step run by scripts/make_int8.py,
    # not something either exporter has ever produced. Mapping it here
    # would change what `--version vX.Y.Z-int8` does.
}


def get_target(name: str) -> ExportTarget:
    try:
        return TARGETS[name]
    except KeyError as exc:
        raise UnsupportedTargetError(
            f"unknown target {name!r}; known targets: "
            f"{', '.join(sorted(TARGETS))}"
        ) from exc


def target_for(architecture: str, precision: str, **overrides: Any) -> ExportTarget:
    """The shipped descriptor for an (architecture, precision) pair.

    ``overrides`` are applied via :meth:`ExportTarget.derive`, which is
    how a CLI flag such as ``--imgsz`` or ``--opset`` reaches the target
    without the target matrix growing a special case.
    """
    parsed = Precision.parse(precision)
    name = SHIPPED.get((architecture, parsed.value))
    if name is None:
        if parsed is Precision.INT8:
            raise NotImplementedError(
                _INT8_NOT_WIRED.format(mechanism="the export path")
            )
        raise UnsupportedTargetError(
            f"no shipped target for architecture {architecture!r} at "
            f"precision {parsed.value!r}; known pairs: "
            f"{', '.join(f'{a}/{p}' for a, p in sorted(SHIPPED))}"
        )
    target = get_target(name)
    return target.derive(**overrides) if overrides else target
