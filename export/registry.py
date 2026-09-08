"""Model release registry: one record per (artifact bytes, claims, target).

**The gap this closes.** `dregsbane-web-backend/src/lib/inference/session.ts`
loads an ONNX from an S3 key plus a `.meta.json` sidecar next to it. Its
`ModelMeta` interface carries `version`, `modelName`, `classes`, `classCount`,
`inputShape`, the two recommended thresholds, `architecture` and
`normalization`. It carries **no precision field**, and no code path checks
one. Precision is encoded in exactly two places today: the `-fp32` suffix
inside the version string, and the S3 key. Neither is checked against the
bytes. An int8 artifact written to the fp32 key would load, serve, and report
``modelVersion: "v2.0.1-fp32"``, and nothing in the system would record that
precision had changed. The parity evidence for a conversion, meanwhile, lives
in `reports/` related to the artifact only by filename.

This module is the smallest store that makes both of those loud.

**Identity is the provenance tuple, not the version string.** A record's
identity digest is computed over: model name, architecture, precision, artifact
sha256 and byte count, the conversion (exporter, mechanism, opset, source ONNX
sha256), the IO signature (input and output names, shapes and element types),
the class-list hash and count, the recommended thresholds, and the target
(runtime, runtime version, execution provider, host class). The version string
is deliberately **outside** that digest. It is a label people re-point at new
bytes — which is precisely the defect above — so it is recorded, indexed, and
constrained (one `(model, version)` may name exactly one artifact sha256), but
it never confers identity. Relabelling an artifact cannot make it a different
release, and different bytes cannot inherit an existing label.

Thresholds are inside the tuple on purpose. A release is bytes *plus the claims
made about them*; `confThresholdRecommended = 0.4` was reasoned about against
fp32 behaviour and is stamped into converted artifacts unchanged. The same
bytes shipped with a different recommendation is a different release.

**Precision is checked against the bytes, not just declared.** A `precision`
field that is only asserted is worth exactly what the version suffix is worth.
`precisionEvidence` is derived instead — a byte-weighted histogram of the
graph's initializer element types and the precision that implies — and a record
whose declaration contradicts its own weights is refused. That inference is a
majority rule over precision-bearing element types, not a proof: it answers
"which element type holds most of the weight bytes", and it declines to answer
when nothing holds a majority.

**Parity is a required field.** `ReleaseRecord` cannot be constructed without a
`ParityRef`, and the reference is checked rather than trusted: the referenced
report must record *this artifact's* sha256 on the side the record claims, must
have been run at the operating point the record recommends, and — for a
converted artifact — must have compared against the source ONNX the record
names. A record that passes those checks still licenses nothing beyond what its
report's own `fixtures` and `environment` blocks license.

**What a record licenses.** That these bytes, with this class list, these
recommended thresholds and this IO signature, were produced by the recorded
mechanism from the recorded source, and that a parity measurement naming these
bytes exists and says what `parity.comparison` says. **What it does not
license:** that the artifact is fit to serve. In particular `measuredOnTarget`
is stored precisely because it is usually `false` — parity was measured by
Python ONNX Runtime on a workstation, and the fp32 record's declared target is
`onnxruntime-node` on arm64 Lambda. Evidence gathered off-target is evidence
about the conversion, not about the deployment.

Deliberately small: stdlib only, JSON files under `registry/`, no service, no
lock, no network. The point is the data model. It is additive — nothing here
reads or writes a shipped artifact, the serving path, or an exporter.

Not yet joined up: `export/precision.py` names the *build* side of the same
concepts (`Precision`, `IOPrecision`, `Mechanism`, `ExportTarget`). This module
stores `precision` and `mechanism` as free text because a record has to be able
to describe an artifact built before that vocabulary existed — including the
three registered here. Making `Conversion.mechanism` carry a `Mechanism` value
alongside its free-text detail, and recording the `IOPrecision` decision by
name rather than only as the input tensor's dtype, is the obvious next step and
is not done. Until it is, the two vocabularies can drift.

Usage::

    python -m export.registry register \\
        --artifact ../roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.onnx \\
        --meta     ../roboflow-deliverables/v2.0.1-fp32/rfdetr-s-litter.fp32.meta.json \\
        --exporter "export.export_rfdetr.export" \\
        --mechanism native-export \\
        --target-runtime onnxruntime-node --target-runtime-version "^1.27.0" \\
        --target-ep cpu --target-host-class aws-lambda-arm64-nodejs20 \\
        --parity reports/parity-rfdetr-v2.0.1-fp32-vs-fp16.json \\
        --parity-role reference

    python -m export.registry list
    python -m export.registry show v2.0.1-fp16
    python -m export.registry verify --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1

#: Repo root, used to resolve the relative paths stored on a record.
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REGISTRY_DIR = REPO_ROOT / "registry"

#: Mirrors ``export.export_yolov8.VERSION_PATTERN``. Restated rather than
#: imported so this module stays stdlib-only (a registry has to be readable in
#: a CI job that has no conversion toolchain installed). ``tests/test_registry``
#: asserts the two patterns are identical, so the copy cannot drift silently.
VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+-(fp32|fp16|int8)$")

PRECISIONS = ("fp32", "fp16", "int8")

#: Mechanism sentinel for an artifact that was exported directly at its final
#: precision rather than converted from another ONNX. Anything else is a
#: conversion and must name its source ONNX.
NATIVE_EXPORT = "native-export"

PARITY_ROLES = ("reference", "candidate")

#: Tolerance for comparing a recommended threshold against the one the parity
#: run used. Both sides are JSON floats written by this repo; this exists to
#: absorb decimal round-tripping, not to permit a different operating point.
_THRESHOLD_TOL = 1e-9


# --------------------------------------------------------------------------
# errors — each one corresponds to a test in tests/test_registry.py. A check
# that is not observed to fire is a comment, not a gate.
# --------------------------------------------------------------------------

class RegistryError(Exception):
    """Base class for every refusal this module makes."""


class MissingParityError(RegistryError):
    """A record was built without a parity result. There is no such record."""


class ParityMismatchError(RegistryError):
    """The referenced parity result is not evidence about this record.

    Raised when the report does not name this artifact's sha256 on the side
    the record claims, when it was run at a different operating point than the
    record recommends, or when it compared against something other than the
    record's declared source ONNX.
    """


class PrecisionMismatchError(RegistryError):
    """The version string's precision suffix disagrees with the bytes' precision.

    This is the failure the serving path cannot currently see: precision lives
    only in the version string and the S3 key, and nothing checks either
    against the artifact.
    """


class VersionCollisionError(RegistryError):
    """A different artifact was registered under an existing (model, version).

    The same defect as overwriting an S3 key, caught at registration instead of
    in production.
    """


class ImmutableRecordError(RegistryError):
    """The provenance tuple is already registered with a different body.

    Records are immutable. Same tuple, same body is idempotent; same tuple,
    different body is a contradiction and has to be resolved by a human.
    """


# --------------------------------------------------------------------------
# hashing and canonicalisation
# --------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes, streamed. Artifacts here are 36-120 MB."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding used for every digest in this module.

    Sorted keys, no insignificant whitespace, UTF-8, and ``allow_nan=False``
    so a NaN cannot silently become part of a content address.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def class_list_sha256(classes: Sequence[str]) -> str:
    """Digest of the class list *in order*.

    Order is load-bearing: logit column *i* is class *i*, so a reordered list
    with identical membership is a different model contract. Recomputable as
    ``sha256(canonical_bytes(list(classes)))``.
    """
    return hashlib.sha256(canonical_bytes(list(classes))).hexdigest()


def precision_of_version(version: str) -> str:
    """The precision suffix a version string claims. Raises on a bad format."""
    m = VERSION_PATTERN.match(version)
    if not m:
        raise ValueError(
            f"version {version!r} does not match "
            f"v<MAJOR>.<MINOR>.<PATCH>-<precision> where precision in "
            f"{PRECISIONS}"
        )
    return m.group(1)


# --------------------------------------------------------------------------
# record parts
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorSpec:
    """One graph input or output, as the ONNX declares it.

    ``dtype`` is the declared element type, not an inferred one. It is carried
    because it is where an fp16 conversion's IO boundary becomes visible:
    ``export.export_rfdetr._to_fp16`` uses ``keep_io_types=True``, so an fp16
    artifact still declares FLOAT at the boundary, and a converter that did not
    would be a different contract with the consumer.
    """

    name: str
    shape: tuple[Any, ...]
    dtype: str

    def to_json_obj(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}

    @staticmethod
    def from_json_obj(o: dict[str, Any]) -> "TensorSpec":
        return TensorSpec(name=o["name"], shape=tuple(o["shape"]), dtype=o["dtype"])


@dataclass(frozen=True)
class Conversion:
    """How these bytes came to exist.

    ``mechanism`` is the specific call, not a category: knowing an artifact is
    "fp16" says nothing useful, knowing it came from
    ``onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)``
    followed by ``export.fp16_repair.repair`` names something reproducible.

    ``source_checkpoint_sha256`` is optional and is ``None`` for every record in
    this repo: the training checkpoints were never hashed. That is a real hole
    in provenance and the field exists so the hole is stated rather than absent.
    """

    exporter: str
    mechanism: str
    opset: int
    source_sha256: str | None = None
    source_checkpoint_sha256: str | None = None

    @property
    def is_conversion(self) -> bool:
        return self.mechanism != NATIVE_EXPORT

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "exporter": self.exporter,
            "mechanism": self.mechanism,
            "opset": self.opset,
            "sourceSha256": self.source_sha256,
            "sourceCheckpointSha256": self.source_checkpoint_sha256,
        }

    @staticmethod
    def from_json_obj(o: dict[str, Any]) -> "Conversion":
        return Conversion(
            exporter=o["exporter"],
            mechanism=o["mechanism"],
            opset=o["opset"],
            source_sha256=o.get("sourceSha256"),
            source_checkpoint_sha256=o.get("sourceCheckpointSha256"),
        )


@dataclass(frozen=True)
class PrecisionEvidence:
    """What the graph's own weights say the precision is.

    The declared ``precision`` field is an assertion by whoever registered the
    artifact — the same kind of assertion as the ``-fp32`` suffix in a version
    string, and worth exactly as much. This block is derived from the bytes
    instead: a byte-weighted histogram of the initializers' element types, and
    the precision that implies.

    It is deliberately a *histogram plus an inference*, not a verdict. A
    dynamically-quantized graph carries int8 weights alongside fp32 scales and
    int64 shapes; "which precision is this" is only answerable as "which
    element type holds most of the weight bytes". ``implied_precision`` is
    ``None`` when nothing holds a majority, and a ``None`` there is not a pass.

    Absent evidence is stored as ``null`` rather than omitted, so a record that
    never had its bytes inspected says so.
    """

    weight_bytes_by_dtype: dict[str, int]
    implied_precision: str | None
    #: Set when the artifact's precision CANNOT be derived from its bytes —
    #: a TensorRT plan, or any other opaque build product. This is not the
    #: same state as ``precision_evidence=None``, which means "nobody looked".
    #: Collapsing the two would let "we cannot check this" masquerade as "we
    #: forgot to check this", and the derived-not-declared guarantee would
    #: silently stop applying exactly where it is least verifiable.
    opaque_reason: str | None = None

    @staticmethod
    def opaque(reason: str) -> "PrecisionEvidence":
        """Evidence that there is no evidence, and why.

        For an artifact whose weights are not inspectable — a serialized
        engine plan, for instance. The declared precision then rests on the
        build recipe recorded in ``Conversion``, and the record says so
        instead of implying an inspection happened.
        """
        return PrecisionEvidence({}, None, opaque_reason=reason)

    def to_json_obj(self) -> dict[str, Any]:
        o: dict[str, Any] = {
            "weightBytesByDtype": dict(self.weight_bytes_by_dtype),
            "impliedPrecision": self.implied_precision,
        }
        if self.opaque_reason is not None:
            o["opaqueReason"] = self.opaque_reason
        return o

    @staticmethod
    def from_json_obj(o: dict[str, Any] | None) -> "PrecisionEvidence | None":
        if o is None:
            return None
        return PrecisionEvidence(
            weight_bytes_by_dtype=dict(o["weightBytesByDtype"]),
            implied_precision=o["impliedPrecision"],
            opaque_reason=o.get("opaqueReason"),
        )


#: Element types that carry a precision claim, and the precision they imply.
#: Everything else (INT64 shapes, INT32 indices, BOOL masks) is plumbing and is
#: counted in the histogram but never votes.
_PRECISION_BEARING = {
    "FLOAT": "fp32",
    "DOUBLE": "fp32",
    "FLOAT16": "fp16",
    "BFLOAT16": "fp16",
    "INT8": "int8",
    "UINT8": "int8",
}


def infer_precision_from_weights(hist: dict[str, int]) -> str | None:
    """The precision implied by a weight-byte histogram, or None.

    Majority rule over the precision-bearing element types only: the winner
    must hold more than half of those bytes. Returns ``None`` rather than
    guessing — an artifact whose weights are evenly split is one a human should
    look at, not one the registry should label.
    """
    votes: dict[str, int] = {}
    for dtype, nbytes in hist.items():
        precision = _PRECISION_BEARING.get(dtype)
        if precision:
            votes[precision] = votes.get(precision, 0) + nbytes
    total = sum(votes.values())
    if not total:
        return None
    winner, nbytes = max(votes.items(), key=lambda kv: kv[1])
    return winner if nbytes * 2 > total else None


@dataclass(frozen=True)
class Target:
    """Where this release is aimed.

    ``host_class`` is a *class* of host, not a machine — "aws-lambda-arm64-
    nodejs20", not an instance id. Two machines in one class can still differ
    in ways that move latency, so a latency number attached to a class is a
    number about the machine it was taken on and nothing more.

    The target is part of the identity tuple: the same bytes served under a
    different runtime or execution provider is a different release, because the
    numerics and the failure modes are not the same ones that were measured.
    """

    runtime: str
    runtime_version: str
    execution_provider: str
    host_class: str
    # Accelerator identity. Optional because a CPU target has none, and
    # because OMITTING them when absent keeps the JSON — and therefore every
    # record_id written before these existed — byte-identical. A GPU target
    # that packed these into host_class would put them in the digest by string
    # concatenation; as fields they are addressable and comparable.
    compute_capability: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None

    def to_json_obj(self) -> dict[str, Any]:
        o: dict[str, Any] = {
            "runtime": self.runtime,
            "runtimeVersion": self.runtime_version,
            "executionProvider": self.execution_provider,
            "hostClass": self.host_class,
        }
        if self.compute_capability is not None:
            o["computeCapability"] = self.compute_capability
        if self.cuda_version is not None:
            o["cudaVersion"] = self.cuda_version
        if self.driver_version is not None:
            o["driverVersion"] = self.driver_version
        return o

    @property
    def digest(self) -> str:
        """Content address of the target alone.

        Uniqueness is keyed on (model, version, THIS) rather than on
        (model, version). identity() has always included the target, so
        keying uniqueness on less than that made the module contradict
        itself: two engines built from one ONNX for different targets are
        two releases by identity() and a collision by the uniqueness rule.
        """
        return hashlib.sha256(canonical_bytes(self.to_json_obj())).hexdigest()

    @staticmethod
    def from_json_obj(o: dict[str, Any]) -> "Target":
        return Target(
            runtime=o["runtime"],
            runtime_version=o["runtimeVersion"],
            execution_provider=o["executionProvider"],
            host_class=o["hostClass"],
            compute_capability=o.get("computeCapability"),
            cuda_version=o.get("cudaVersion"),
            driver_version=o.get("driverVersion"),
        )


@dataclass(frozen=True)
class ParityRef:
    """A pinned reference to a parity result, transcribed from the report.

    ``report_sha256`` pins the content, so the copied ``comparison`` numbers can
    always be re-derived from the exact file they came from. ``role`` says which
    side of the comparison this record's artifact was:

    * ``candidate`` — the artifact was measured *against* a baseline. The
      comparison describes it.
    * ``reference`` — the artifact *was* the baseline. The comparison describes
      a conversion of it and licenses nothing about the artifact's own fidelity
      to the checkpoint it came from. A reference-role record is a record whose
      parity evidence is "other things were measured against me".

    ``measured_on`` is the report's own environment block, kept so a reader can
    see at a glance that the measurement host is not usually the target host.
    """

    report: str
    report_sha256: str
    role: str
    artifact_sha256: str
    compared_against_sha256: str
    conf_threshold: float
    measured_on: dict[str, Any]
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "reportSha256": self.report_sha256,
            "role": self.role,
            "artifactSha256": self.artifact_sha256,
            "comparedAgainstSha256": self.compared_against_sha256,
            "confThreshold": self.conf_threshold,
            "measuredOn": dict(self.measured_on),
            "comparison": dict(self.comparison),
        }

    @staticmethod
    def from_json_obj(o: dict[str, Any]) -> "ParityRef":
        return ParityRef(
            report=o["report"],
            report_sha256=o["reportSha256"],
            role=o["role"],
            artifact_sha256=o["artifactSha256"],
            compared_against_sha256=o["comparedAgainstSha256"],
            conf_threshold=o["confThreshold"],
            measured_on=dict(o.get("measuredOn", {})),
            comparison=dict(o.get("comparison", {})),
        )


def parity_ref_from_report(
    report_path: Path,
    role: str,
    *,
    root: Path = REPO_ROOT,
) -> ParityRef:
    """Transcribe an ``export.parity`` JSON report into a pinned reference.

    Reads only; the report is the authority and nothing is recomputed here. The
    caller supplies the role, and `ReleaseRecord` checks that the role's side of
    the report actually names the record's artifact — this function does not,
    because it does not yet know the artifact.
    """
    if role not in PARITY_ROLES:
        raise ValueError(f"role must be one of {PARITY_ROLES}, got {role!r}")
    report_path = Path(report_path)
    doc = json.loads(report_path.read_text())

    side = "a" if role == "reference" else "b"
    other = "b" if side == "a" else "a"
    artifacts = doc["artifacts"]
    decode = doc["decode"]
    env = doc.get("environment", {})

    recommended = next(
        (row for row in decode.get("byThreshold", []) if row.get("isRecommended")),
        None,
    )
    comparison: dict[str, Any] = {
        "schemaVersion": doc.get("schemaVersion"),
        "generatedAt": doc.get("generatedAt"),
        "fixtureCount": doc.get("fixtures", {}).get("count"),
        "fixtureManifestSha256": doc.get("fixtures", {}).get("manifestSha256"),
        "maxAbsDeltaByOutput": {
            o["name"]: o["maxAbsDelta"] for o in doc.get("rawOutputs", [])
        },
        "maxRelDeltaByOutput": {
            o["name"]: o["maxRelDelta"] for o in doc.get("rawOutputs", [])
        },
        "latencyMsThisArtifact": {
            k: v for k, v in doc.get("latencyMs", {}).get(side, {}).items()
        },
    }
    if recommended is not None:
        comparison["atRecommendedThreshold"] = {
            "threshold": recommended["threshold"],
            "thresholdFlips": recommended["thresholdFlips"],
            "classArgmaxFlips": recommended["classArgmaxFlips"],
            "detectionsReference": recommended["detectionsA"],
            "detectionsCandidate": recommended["detectionsB"],
            "imagesWithAnyFlip": recommended["imagesWithAnyFlip"],
        }

    # Do not assume the measurement was made under ONNX Runtime. This helper
    # transcribes an export.parity report, which is ORT-shaped; a report from
    # another runtime (a TensorRT engine comparison, say) carries a different
    # environment block and, often, no byThreshold[]. Stamping
    # runtime="onnxruntime" on one of those would put a false claim inside a
    # record whose entire purpose is provenance. Refuse instead, and let the
    # caller build the ParityRef explicitly.
    if "onnxruntime" not in env:
        raise ValueError(
            f"{report_path}: no environment.onnxruntime — this does not look "
            f"like an export.parity report, and this helper would otherwise "
            f"record runtime='onnxruntime' for a measurement that was not made "
            f"under it. Construct the ParityRef directly and name the runtime "
            f"that actually produced the numbers."
        )
    eps = env.get("executionProviders") or [None]
    measured_on = {
        "runtime": "onnxruntime",
        "runtimeVersion": env.get("onnxruntime"),
        "executionProvider": eps[0],
        "host": env.get("platform"),
    }

    try:
        rel = str(report_path.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        rel = str(report_path)

    return ParityRef(
        report=rel,
        report_sha256=sha256_file(report_path),
        role=role,
        artifact_sha256=artifacts[side]["sha256"],
        compared_against_sha256=artifacts[other]["sha256"],
        conf_threshold=float(decode["confThresholdRecommended"]),
        measured_on=measured_on,
        comparison=comparison,
    )


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseRecord:
    """One release: bytes, the claims made about them, and where they run.

    Validation happens in ``__post_init__``, so an invalid record cannot be
    constructed — not merely cannot be written. Every refusal below has a test
    that observes it fire.
    """

    model_name: str
    architecture: str
    precision: str
    artifact_sha256: str
    artifact_bytes: int
    conversion: Conversion
    input_spec: TensorSpec
    output_specs: tuple[TensorSpec, ...]
    class_list_sha256: str
    class_count: int
    conf_threshold_recommended: float
    iou_threshold_recommended: float
    target: Target
    parity: ParityRef
    version: str
    artifact_path: str
    precision_evidence: PrecisionEvidence | None = None
    notes: tuple[str, ...] = ()
    registered_at: str | None = None

    # -- validation ------------------------------------------------------

    def __post_init__(self) -> None:
        if self.parity is None:
            raise MissingParityError(
                "a release record requires a parity result; a record whose "
                "conversion has never been measured is a claim, not a release"
            )
        if not isinstance(self.parity, ParityRef):
            raise MissingParityError(
                f"parity must be a ParityRef, got {type(self.parity).__name__}"
            )
        if self.parity.role not in PARITY_ROLES:
            raise ValueError(
                f"parity role must be one of {PARITY_ROLES}, "
                f"got {self.parity.role!r}"
            )
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {PRECISIONS}, got {self.precision!r}"
            )

        claimed = precision_of_version(self.version)  # raises on a bad format
        if claimed != self.precision:
            raise PrecisionMismatchError(
                f"version {self.version!r} claims precision {claimed!r} but the "
                f"artifact is {self.precision!r}. The serving path reads "
                f"precision from this string and nothing else; a record that "
                f"let them disagree would reproduce the defect it exists to "
                f"catch"
            )

        evidence = self.precision_evidence
        if (
            evidence is not None
            and evidence.implied_precision is not None
            and evidence.implied_precision != self.precision
        ):
            raise PrecisionMismatchError(
                f"this record declares {self.precision!r} but the graph's own "
                f"weights are {evidence.implied_precision!r} "
                f"({evidence.weight_bytes_by_dtype}). The declaration is an "
                f"assertion; the histogram is the bytes"
            )

        if self.conversion.is_conversion and not self.conversion.source_sha256:
            raise ValueError(
                f"mechanism {self.conversion.mechanism!r} is a conversion, so "
                f"the source ONNX sha256 is required; pass mechanism="
                f"{NATIVE_EXPORT!r} for an artifact exported directly at its "
                f"final precision"
            )
        if self.class_count <= 0:
            raise ValueError("class_count must be positive")
        if not self.output_specs:
            raise ValueError("a record must name at least one graph output")
        if self.artifact_bytes <= 0:
            raise ValueError("artifact_bytes must be positive")

        # -- the parity gates ------------------------------------------
        if self.parity.artifact_sha256 != self.artifact_sha256:
            raise ParityMismatchError(
                f"parity report {self.parity.report} records sha256 "
                f"{self.parity.artifact_sha256[:12]}… on side "
                f"{self.parity.role!r}, but this record's artifact is "
                f"{self.artifact_sha256[:12]}…. The evidence is about a "
                f"different file"
            )
        if abs(self.parity.conf_threshold - self.conf_threshold_recommended) > _THRESHOLD_TOL:
            raise ParityMismatchError(
                f"parity was measured at confidence "
                f"{self.parity.conf_threshold} but this record recommends "
                f"{self.conf_threshold_recommended}. Operating-point flips "
                f"counted at one threshold say nothing about another"
            )
        if (
            self.parity.role == "candidate"
            and self.conversion.source_sha256
            and self.parity.compared_against_sha256 != self.conversion.source_sha256
        ):
            raise ParityMismatchError(
                f"this artifact was converted from "
                f"{self.conversion.source_sha256[:12]}… but the parity report "
                f"compares it against "
                f"{self.parity.compared_against_sha256[:12]}…. A conversion "
                f"measured against something other than its own source is not "
                f"evidence about the conversion"
            )

    # -- identity --------------------------------------------------------

    def identity(self) -> dict[str, Any]:
        """The provenance tuple. Everything here, and only this, is identity.

        Absent by design: the version string, the artifact's path, the notes,
        the registration timestamp, and the parity reference. The first two are
        labels and locations — the things that get re-pointed at new bytes. The
        last is evidence *about* an identity, not part of one.
        """
        return {
            "modelName": self.model_name,
            "architecture": self.architecture,
            "precision": self.precision,
            "artifact": {"sha256": self.artifact_sha256, "bytes": self.artifact_bytes},
            "conversion": self.conversion.to_json_obj(),
            "io": {
                "input": self.input_spec.to_json_obj(),
                "outputs": [o.to_json_obj() for o in self.output_specs],
            },
            "classes": {"sha256": self.class_list_sha256, "count": self.class_count},
            "thresholds": {
                "confRecommended": self.conf_threshold_recommended,
                "iouRecommended": self.iou_threshold_recommended,
            },
            "target": self.target.to_json_obj(),
        }

    @property
    def record_id(self) -> str:
        """Content address: sha256 over the canonical JSON of ``identity()``."""
        return hashlib.sha256(canonical_bytes(self.identity())).hexdigest()

    @property
    def short_id(self) -> str:
        return self.record_id[:12]

    @property
    def measured_on_target(self) -> bool:
        """Was the parity evidence gathered on the target this record declares?

        Normally ``False``, and stored rather than hidden. Divergence measured
        by Python ONNX Runtime on a workstation is evidence about the
        conversion; it is not evidence about `onnxruntime-node` on Lambda.
        """
        m = self.parity.measured_on
        return (
            m.get("runtime") == self.target.runtime
            and m.get("runtimeVersion") == self.target.runtime_version
            and m.get("executionProvider") == self.target.execution_provider
            and m.get("host") == self.target.host_class
        )

    @property
    def filename(self) -> str:
        return f"{self.model_name}.{self.version}.{self.short_id}.json"

    # -- serialisation ---------------------------------------------------

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "recordId": self.record_id,
            "version": self.version,
            "artifactPath": self.artifact_path,
            "identity": self.identity(),
            "parity": self.parity.to_json_obj(),
            "measuredOnTarget": self.measured_on_target,
            "precisionEvidence": (
                self.precision_evidence.to_json_obj()
                if self.precision_evidence is not None
                else None
            ),
            "notes": list(self.notes),
            "registeredAt": self.registered_at,
        }

    @staticmethod
    def from_json_obj(o: dict[str, Any]) -> "ReleaseRecord":
        ident = o["identity"]
        rec = ReleaseRecord(
            model_name=ident["modelName"],
            architecture=ident["architecture"],
            precision=ident["precision"],
            artifact_sha256=ident["artifact"]["sha256"],
            artifact_bytes=ident["artifact"]["bytes"],
            conversion=Conversion.from_json_obj(ident["conversion"]),
            input_spec=TensorSpec.from_json_obj(ident["io"]["input"]),
            output_specs=tuple(
                TensorSpec.from_json_obj(t) for t in ident["io"]["outputs"]
            ),
            class_list_sha256=ident["classes"]["sha256"],
            class_count=ident["classes"]["count"],
            conf_threshold_recommended=ident["thresholds"]["confRecommended"],
            iou_threshold_recommended=ident["thresholds"]["iouRecommended"],
            target=Target.from_json_obj(ident["target"]),
            parity=ParityRef.from_json_obj(o["parity"]),
            version=o["version"],
            artifact_path=o["artifactPath"],
            precision_evidence=PrecisionEvidence.from_json_obj(o.get("precisionEvidence")),
            notes=tuple(o.get("notes", ())),
            registered_at=o.get("registeredAt"),
        )
        stored = o.get("recordId")
        if stored and stored != rec.record_id:
            raise RegistryError(
                f"record {o.get('version')} claims id {stored[:12]}… but its "
                f"identity hashes to {rec.short_id}…; the file has been edited"
            )
        return rec

    def body_without_timestamp(self) -> dict[str, Any]:
        """The comparable body. ``registeredAt`` is when, not what."""
        body = self.to_json_obj()
        body.pop("registeredAt", None)
        return body


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RegisterResult:
    record: ReleaseRecord
    path: Path
    created: bool


@dataclass(frozen=True)
class VerifyItem:
    what: str
    path: str
    status: str  # "ok" | "changed" | "missing"
    expected: str
    actual: str | None


class Registry:
    """A directory of JSON release records. No index, no lock, no service.

    Small enough to read with ``cat`` and to diff in a pull request, which is
    the point: the review that catches a wrong record is a human one.
    """

    def __init__(self, root: Path = DEFAULT_REGISTRY_DIR) -> None:
        self.root = Path(root)

    # -- reads -----------------------------------------------------------

    def paths(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.glob("*.json"))

    def records(self) -> list[ReleaseRecord]:
        return [ReleaseRecord.from_json_obj(json.loads(p.read_text())) for p in self.paths()]

    def find(self, needle: str) -> list[ReleaseRecord]:
        """Match on record-id prefix, exact version, or model name."""
        out = []
        for r in self.records():
            if (
                r.record_id.startswith(needle)
                or r.version == needle
                or r.model_name == needle
            ):
                out.append(r)
        return out

    # -- write -----------------------------------------------------------

    def register(self, record: ReleaseRecord, *, now: str | None = None) -> RegisterResult:
        """Write a record, or return the existing identical one.

        Three outcomes, and only three:

        * the tuple is new and its version label is free — written;
        * the tuple exists with a byte-identical body — no write, no new
          timestamp, returns the stored record (idempotent);
        * anything else — refused, loudly.
        """
        existing = {r.record_id: r for r in self.records()}

        # (model, version, target) may name exactly one artifact. The S3-key
        # overwrite this was built to catch happens WITHIN a target, and is
        # still caught.
        #
        # The target belongs in this key because identity() has always
        # included it. Keying uniqueness on (model, version) alone made the
        # module contradict itself: an ONNX artifact and the TensorRT engine
        # built FROM it are two releases by identity() — different record_ids
        # — and were a collision by this rule. So were two engines built from
        # one ONNX for different targets (TF32 on vs off), which is exactly
        # the one-source-many-targets case a release registry exists to hold.
        # A version string names a MODEL VERSION, not a build of it.
        for other in existing.values():
            if (
                other.model_name == record.model_name
                and other.version == record.version
                and other.target.digest == record.target.digest
                and other.artifact_sha256 != record.artifact_sha256
            ):
                raise VersionCollisionError(
                    f"{record.model_name} {record.version} on target "
                    f"{other.target.runtime}/{other.target.execution_provider}"
                    f"@{other.target.runtime_version} is already registered as "
                    f"artifact {other.artifact_sha256[:12]}… (record "
                    f"{other.short_id}); this artifact is "
                    f"{record.artifact_sha256[:12]}…. One version string cannot "
                    f"name two sets of bytes FOR THE SAME TARGET — rev the "
                    f"version, or register under the target it was actually "
                    f"built for"
                )

        prior = existing.get(record.record_id)
        if prior is not None:
            was, now_body = prior.body_without_timestamp(), record.body_without_timestamp()
            if was == now_body:
                return RegisterResult(prior, self._path_for(prior), created=False)
            differing = sorted(
                k for k in set(was) | set(now_body) if was.get(k) != now_body.get(k)
            )
            raise ImmutableRecordError(
                f"record {record.short_id} ({prior.version}) already exists "
                f"with a different body; differs in: {', '.join(differing)}. "
                f"Records are immutable: the same provenance tuple cannot be "
                f"re-registered with different claims. If the claims changed, "
                f"the release did"
            )

        stamped = replace(
            record,
            registered_at=now or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path_for(stamped)
        path.write_text(json.dumps(stamped.to_json_obj(), indent=2, sort_keys=True) + "\n")
        return RegisterResult(stamped, path, created=True)

    def _path_for(self, record: ReleaseRecord) -> Path:
        return self.root / record.filename

    # -- verify ----------------------------------------------------------

    def verify(self, record: ReleaseRecord, *, base: Path = REPO_ROOT) -> list[VerifyItem]:
        """Re-hash the artifact and the parity report named by a record.

        Answers exactly one question — *are the files on this disk still the
        ones this record describes?* It does not run the model and it does not
        re-run parity, so an ``ok`` here licenses nothing about behaviour.
        """
        items = []
        for what, rel, expected in (
            ("artifact", record.artifact_path, record.artifact_sha256),
            ("parity-report", record.parity.report, record.parity.report_sha256),
        ):
            p = (Path(base) / rel) if not Path(rel).is_absolute() else Path(rel)
            if not p.exists():
                items.append(VerifyItem(what, str(rel), "missing", expected, None))
                continue
            actual = sha256_file(p)
            items.append(
                VerifyItem(
                    what, str(rel), "ok" if actual == expected else "changed",
                    expected, actual,
                )
            )
        return items


# --------------------------------------------------------------------------
# building a record from files on disk (CLI support)
# --------------------------------------------------------------------------

_ONNX_ELEM_TYPES = {
    1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32",
    7: "INT64", 9: "BOOL", 10: "FLOAT16", 11: "DOUBLE", 12: "UINT32",
    13: "UINT64", 16: "BFLOAT16",
}

_ONNX_ELEM_SIZE = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4,
    13: 8, 16: 2,
}


@dataclass(frozen=True)
class OnnxFacts:
    """What one pass over an ONNX file can state without running it."""

    opset: int
    input_spec: TensorSpec
    output_specs: tuple[TensorSpec, ...]
    weight_bytes_by_dtype: dict[str, int]


def read_onnx_facts(path: Path) -> OnnxFacts:
    """Read opset, IO signature and the weight-dtype histogram from a graph.

    Uses ``onnx`` (the ``parity`` optional extra), which is why it lives at the
    CLI edge and not in the record: a registry has to stay readable on a
    machine with no conversion toolchain. Declared shapes and element types are
    reported as the graph states them; only the histogram is derived, and it is
    counted as ``prod(dims) * itemsize`` per initializer, not as serialized
    size.
    """
    import onnx  # optional extra; a metadata store must not require it

    model = onnx.load(str(path), load_external_data=False)
    opsets = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
    opset = opsets[0].version if opsets else -1

    def spec(vi: Any) -> TensorSpec:
        dims: list[Any] = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
        return TensorSpec(
            name=vi.name,
            shape=tuple(dims),
            dtype=_ONNX_ELEM_TYPES.get(
                vi.type.tensor_type.elem_type, str(vi.type.tensor_type.elem_type)
            ),
        )

    inputs = [spec(i) for i in model.graph.input]
    if len(inputs) != 1:
        raise ValueError(f"expected exactly one graph input, found {len(inputs)}")

    hist: dict[str, int] = {}
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        name = _ONNX_ELEM_TYPES.get(init.data_type, str(init.data_type))
        hist[name] = hist.get(name, 0) + n * _ONNX_ELEM_SIZE.get(init.data_type, 1)

    return OnnxFacts(
        opset=opset,
        input_spec=inputs[0],
        output_specs=tuple(spec(o) for o in model.graph.output),
        weight_bytes_by_dtype=dict(sorted(hist.items())),
    )


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        # Outside the repo (the shipped fp32 lives in ../roboflow-deliverables).
        import os

        return os.path.relpath(Path(path).resolve(), Path(root).resolve())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cmd_register(args: argparse.Namespace) -> int:
    root = Path(args.root)
    artifact = Path(args.artifact)
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}

    version = args.version or meta.get("version")
    if not version:
        print("--version is required when --meta carries none", file=sys.stderr)
        return 2
    precision = args.precision or precision_of_version(version)

    classes = meta.get("classes")
    if not classes:
        print("--meta with a `classes` list is required", file=sys.stderr)
        return 2

    facts = read_onnx_facts(artifact)
    evidence: PrecisionEvidence | None = None
    if not args.skip_weight_check:
        evidence = PrecisionEvidence(
            weight_bytes_by_dtype=facts.weight_bytes_by_dtype,
            implied_precision=infer_precision_from_weights(facts.weight_bytes_by_dtype),
        )

    source_sha = args.source_sha256
    if args.source and not source_sha:
        source_sha = sha256_file(Path(args.source))

    parity = parity_ref_from_report(Path(args.parity), args.parity_role, root=root)

    record = ReleaseRecord(
        model_name=args.model or meta.get("modelName", artifact.stem),
        architecture=args.architecture or meta.get("architecture", "unknown"),
        precision=precision,
        artifact_sha256=sha256_file(artifact),
        artifact_bytes=artifact.stat().st_size,
        conversion=Conversion(
            exporter=args.exporter,
            mechanism=args.mechanism,
            opset=args.opset if args.opset is not None else facts.opset,
            source_sha256=source_sha,
            source_checkpoint_sha256=args.source_checkpoint_sha256,
        ),
        input_spec=facts.input_spec,
        output_specs=facts.output_specs,
        class_list_sha256=class_list_sha256(classes),
        class_count=meta.get("classCount", len(classes)),
        conf_threshold_recommended=(
            args.conf if args.conf is not None else meta["confThresholdRecommended"]
        ),
        iou_threshold_recommended=(
            args.iou if args.iou is not None else meta["iouThresholdRecommended"]
        ),
        target=Target(
            runtime=args.target_runtime,
            runtime_version=args.target_runtime_version,
            execution_provider=args.target_ep,
            host_class=args.target_host_class,
        ),
        parity=parity,
        version=version,
        artifact_path=_relpath(artifact, root),
        precision_evidence=evidence,
        notes=tuple(args.note or ()),
    )

    if args.dry_run:
        print(json.dumps(record.to_json_obj(), indent=2, sort_keys=True))
        return 0

    result = Registry(args.registry).register(record)
    verb = "registered" if result.created else "already registered (no change)"
    print(f"{verb}: {result.record.short_id}  {result.record.version}")
    print(f"  {result.path}")
    if evidence is None:
        print("  note: precision was NOT derived from the bytes "
              "(--skip-weight-check); the record's precision field is an "
              "assertion")
    else:
        print(
            f"  weights imply {evidence.implied_precision!r}; record declares "
            f"{result.record.precision!r}"
        )
    if not result.record.measured_on_target:
        print(
            "  note: parity was NOT measured on this record's declared target "
            f"({result.record.target.runtime} "
            f"{result.record.target.runtime_version} / "
            f"{result.record.target.execution_provider} / "
            f"{result.record.target.host_class})"
        )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    records = Registry(args.registry).records()
    if not records:
        print(f"no records under {args.registry}")
        return 0
    print(f"{'id':14}{'version':16}{'prec':6}{'bytes':>13}  {'parity':10} on-target")
    for r in sorted(records, key=lambda r: (r.model_name, r.version)):
        print(
            f"{r.short_id:14}{r.version:16}{r.precision:6}{r.artifact_bytes:>13,}  "
            f"{r.parity.role:10} {str(r.measured_on_target).lower()}"
        )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    matches = Registry(args.registry).find(args.needle)
    if not matches:
        print(f"no record matching {args.needle!r}", file=sys.stderr)
        return 1
    for r in matches:
        print(json.dumps(r.to_json_obj(), indent=2, sort_keys=True))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    reg = Registry(args.registry)
    records = reg.records() if args.all else reg.find(args.needle or "")
    if not records:
        print("nothing to verify", file=sys.stderr)
        return 1
    bad = 0
    for r in records:
        print(f"{r.short_id}  {r.version}")
        for item in reg.verify(r, base=Path(args.root)):
            mark = {"ok": "ok     ", "changed": "CHANGED", "missing": "MISSING"}[item.status]
            print(f"  {mark} {item.what:14} {item.path}")
            if item.status != "ok":
                bad += 1
                print(f"          expected {item.expected}")
                if item.actual:
                    print(f"          actual   {item.actual}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m export.registry",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_DIR)
    p.add_argument(
        "--root", type=Path, default=REPO_ROOT,
        help="Base that stored relative paths resolve against.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="register one release record")
    r.add_argument("--artifact", type=Path, required=True)
    r.add_argument("--meta", type=Path, help="sidecar meta.json (classes, thresholds)")
    r.add_argument("--model")
    r.add_argument("--architecture")
    r.add_argument("--version")
    r.add_argument("--precision", choices=PRECISIONS)
    r.add_argument("--exporter", required=True, help="the code that produced it")
    r.add_argument(
        "--mechanism", required=True,
        help=f"the specific conversion call, or {NATIVE_EXPORT!r}",
    )
    r.add_argument("--source", type=Path, help="source ONNX for a converted artifact")
    r.add_argument("--source-sha256")
    r.add_argument("--source-checkpoint-sha256")
    r.add_argument("--opset", type=int, help="override; default reads the graph")
    r.add_argument(
        "--skip-weight-check", action="store_true",
        help="Do not derive precision from the graph's initializers. The "
             "record then stores precisionEvidence: null, which says plainly "
             "that its precision field is asserted and not measured.",
    )
    r.add_argument("--conf", type=float)
    r.add_argument("--iou", type=float)
    r.add_argument("--target-runtime", required=True)
    r.add_argument("--target-runtime-version", required=True)
    r.add_argument("--target-ep", required=True)
    r.add_argument("--target-host-class", required=True)
    r.add_argument("--parity", type=Path, required=True, help="export.parity JSON report")
    r.add_argument("--parity-role", choices=PARITY_ROLES, required=True)
    r.add_argument("--note", action="append")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=_cmd_register)

    ls = sub.add_parser("list", help="list registered records")
    ls.set_defaults(fn=_cmd_list)

    sh = sub.add_parser("show", help="print one record")
    sh.add_argument("needle", help="record-id prefix, version, or model name")
    sh.set_defaults(fn=_cmd_show)

    v = sub.add_parser("verify", help="re-hash the files a record names")
    v.add_argument("needle", nargs="?")
    v.add_argument("--all", action="store_true")
    v.set_defaults(fn=_cmd_verify)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except RegistryError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
