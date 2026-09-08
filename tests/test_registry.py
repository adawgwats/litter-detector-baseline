"""Tests for export.registry.

The registry's whole value is a set of refusals, so the tests are mostly
refusals. A check that has never been observed to raise is a comment, not a
gate; each `pytest.raises` below is the observation for one of them.

Three of these correspond directly to defects the deployed system cannot
currently see:

* `test_version_precision_suffix_must_match_the_artifact_precision` — precision
  lives only in the version string and the S3 key today; `ModelMeta` in
  `dregsbane-web-backend/src/lib/inference/session.ts` has no precision field.
* `test_registering_different_bytes_under_the_same_version_is_rejected` — the
  same defect from the other side: overwriting an S3 key.
* `test_a_record_cannot_be_created_without_a_parity_result` — a conversion
  whose divergence was never measured is a claim, not a release.

No test here asserts a divergence tolerance. The registry does not own one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from export.registry import (
    NATIVE_EXPORT,
    Conversion,
    ImmutableRecordError,
    MissingParityError,
    ParityMismatchError,
    ParityRef,
    PrecisionEvidence,
    PrecisionMismatchError,
    Registry,
    RegistryError,
    ReleaseRecord,
    Target,
    TensorSpec,
    VERSION_PATTERN,
    VersionCollisionError,
    canonical_bytes,
    class_list_sha256,
    infer_precision_from_weights,
    main,
    parity_ref_from_report,
    precision_of_version,
    read_onnx_facts,
    sha256_file,
)

SHA_A = "a" * 64  # stands in for the fp32 source
SHA_B = "b" * 64  # stands in for the converted candidate
SHA_C = "c" * 64  # some third artifact


# --------------------------------------------------------------------------
# fixtures — a faithful miniature of an export.parity report
# --------------------------------------------------------------------------

def write_report(
    tmp_path: Path,
    *,
    name: str = "parity.json",
    sha_a: str = SHA_A,
    sha_b: str = SHA_B,
    threshold: float = 0.4,
    platform: str = "macOS-26.5.1-arm64-arm-64bit",
    ort: str = "1.29.0",
) -> Path:
    doc = {
        "schemaVersion": 2,
        "generatedAt": "2026-09-01T03:16:47+00:00",
        "artifacts": {
            "a": {"path": "a.onnx", "sha256": sha_a, "bytes": 120039167},
            "b": {"path": "b.onnx", "sha256": sha_b, "bytes": 63068681},
        },
        "environment": {
            "onnxruntime": ort,
            "executionProviders": ["CPUExecutionProvider"],
            "platform": platform,
            "machine": "arm64",
            "python": "3.12.13",
        },
        "fixtures": {"count": 200, "manifestSha256": "f" * 64},
        "rawOutputs": [
            {"name": "dets", "maxAbsDelta": 2.16, "maxRelDelta": 35878.05},
            {"name": "labels", "maxAbsDelta": 12.17, "maxRelDelta": 79.49},
        ],
        "latencyMs": {
            "a": {"p50": 311.6, "p95": 390.9, "n": 200},
            "b": {"p50": 589.7, "p95": 692.3, "n": 200},
        },
        "decode": {
            "confThresholdRecommended": threshold,
            "byThreshold": [
                {
                    "threshold": 0.25, "isRecommended": False, "thresholdFlips": 36,
                    "classArgmaxFlips": 14, "detectionsA": 325, "detectionsB": 321,
                    "imagesWithAnyFlip": 16,
                },
                {
                    "threshold": threshold, "isRecommended": True,
                    "thresholdFlips": 19, "classArgmaxFlips": 6,
                    "detectionsA": 202, "detectionsB": 203, "imagesWithAnyFlip": 10,
                },
            ],
        },
    }
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def make_record(tmp_path: Path, **over) -> ReleaseRecord:
    """A converted (fp16) candidate record, valid unless an override breaks it."""
    report = over.pop("report", None) or write_report(tmp_path)
    parity = over.pop("parity", ...)
    if parity is ...:
        parity = parity_ref_from_report(report, over.pop("role", "candidate"), root=tmp_path)
    kwargs = dict(
        model_name="rfdetr-s-litter",
        architecture="rfdetr",
        precision="fp16",
        artifact_sha256=SHA_B,
        artifact_bytes=63068681,
        conversion=Conversion(
            exporter="export.export_rfdetr._to_fp16",
            mechanism="onnxconverter_common.float16(keep_io_types=True)",
            opset=17,
            source_sha256=SHA_A,
        ),
        input_spec=TensorSpec("input", (1, 3, 512, 512), "FLOAT"),
        output_specs=(
            TensorSpec("dets", (1, 300, 4), "FLOAT"),
            TensorSpec("labels", (1, 300, 44), "FLOAT"),
        ),
        class_list_sha256="d" * 64,
        class_count=43,
        conf_threshold_recommended=0.4,
        iou_threshold_recommended=0.45,
        target=Target("onnxruntime", "1.29.0", "CPUExecutionProvider",
                      "macOS-26.5.1-arm64-arm-64bit"),
        parity=parity,
        version="v2.0.1-fp16",
        artifact_path="dist/parity/b.onnx",
    )
    kwargs.update(over)
    return ReleaseRecord(**kwargs)


# --------------------------------------------------------------------------
# the required-parity gate
# --------------------------------------------------------------------------

def test_a_record_cannot_be_created_without_a_parity_result(tmp_path: Path) -> None:
    """The headline gate, observed firing both ways it can be defeated.

    Omitting the field is a `TypeError` because the record's shape requires it.
    Passing `None` is a `MissingParityError` because "required" fields usually
    die by being satisfied with an empty value.
    """
    full = make_record(tmp_path)
    kwargs = {
        k: getattr(full, k)
        for k in (
            "model_name", "architecture", "precision", "artifact_sha256",
            "artifact_bytes", "conversion", "input_spec", "output_specs",
            "class_list_sha256", "class_count", "conf_threshold_recommended",
            "iou_threshold_recommended", "target", "version", "artifact_path",
        )
    }
    with pytest.raises(TypeError):
        ReleaseRecord(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(MissingParityError, match="requires a parity result"):
        make_record(tmp_path, parity=None)


def test_a_dict_that_looks_like_parity_is_refused(tmp_path: Path) -> None:
    """A plausible-looking blob is not evidence; only a checked ParityRef is."""
    with pytest.raises(MissingParityError, match="must be a ParityRef"):
        make_record(tmp_path, parity={"report": "reports/parity.json"})


# --------------------------------------------------------------------------
# the parity reference must be evidence about THIS artifact
# --------------------------------------------------------------------------

def test_parity_report_must_name_this_artifact(tmp_path: Path) -> None:
    with pytest.raises(ParityMismatchError, match="different file"):
        make_record(tmp_path, artifact_sha256=SHA_C)


def test_parity_must_have_been_measured_at_the_recommended_threshold(tmp_path: Path) -> None:
    """0.4 flips counted at 0.25 are not evidence about 0.4.

    This is the operating-point half of the P0 finding: the recommendation and
    the measurement have to be the same number or the record is comparing a
    claim against evidence for a different claim.
    """
    report = write_report(tmp_path, threshold=0.25)
    with pytest.raises(ParityMismatchError, match="measured at confidence"):
        make_record(tmp_path, report=report)


def test_candidate_parity_must_compare_against_the_recorded_source(tmp_path: Path) -> None:
    report = write_report(tmp_path, sha_a=SHA_C)  # compared against a third file
    with pytest.raises(ParityMismatchError, match="other than its own source"):
        make_record(tmp_path, report=report)


def test_a_reference_role_record_is_not_subject_to_the_source_check(tmp_path: Path) -> None:
    """The fp32 baseline is side `a` of its own comparisons.

    Its parity evidence says "other artifacts were measured against me", which
    licenses nothing about its own fidelity to the checkpoint it came from —
    hence `role`, so a reader can tell the two apart.
    """
    rec = make_record(
        tmp_path,
        role="reference",
        artifact_sha256=SHA_A,
        artifact_bytes=120039167,
        precision="fp32",
        version="v2.0.1-fp32",
        conversion=Conversion("export.export_rfdetr.export", NATIVE_EXPORT, 17),
    )
    assert rec.parity.role == "reference"
    assert rec.parity.compared_against_sha256 == SHA_B


# --------------------------------------------------------------------------
# precision: the field the serving path does not have
# --------------------------------------------------------------------------

def test_version_precision_suffix_must_match_the_artifact_precision(tmp_path: Path) -> None:
    """An int8 artifact labelled `-fp32` is exactly the undetected failure.

    `ModelMeta` (dregsbane-web-backend/src/lib/inference/session.ts) has no
    precision field, so an int8 file at the fp32 S3 key serves silently. Here
    it is a refusal.
    """
    with pytest.raises(PrecisionMismatchError, match="claims precision 'fp32'"):
        make_record(tmp_path, precision="int8", version="v2.0.1-fp32")


def test_a_version_without_a_precision_suffix_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        make_record(tmp_path, version="v2.0.1")


def test_an_unknown_precision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="precision must be one of"):
        make_record(tmp_path, precision="bf16")


def test_a_declared_precision_that_contradicts_the_weights_is_refused(tmp_path: Path) -> None:
    """The strongest form of the gate: checked against bytes, not a label.

    Declaring fp16 over a graph whose weights are overwhelmingly FLOAT is the
    same class of error as the version-suffix mismatch, one layer down, and it
    is the layer the serving path has no way to see.
    """
    with pytest.raises(PrecisionMismatchError, match="weights are 'fp32'"):
        make_record(
            tmp_path,
            precision_evidence=PrecisionEvidence(
                {"FLOAT": 120_000_000, "INT64": 4_000}, "fp32"
            ),
        )


def test_a_declared_precision_that_agrees_with_the_weights_passes(tmp_path: Path) -> None:
    rec = make_record(
        tmp_path,
        precision_evidence=PrecisionEvidence(
            {"FLOAT": 1_000, "FLOAT16": 63_000_000}, "fp16"
        ),
    )
    assert rec.precision_evidence is not None
    assert rec.to_json_obj()["precisionEvidence"]["impliedPrecision"] == "fp16"


def test_absent_precision_evidence_is_recorded_as_null_not_omitted(tmp_path: Path) -> None:
    """A record that never had its bytes inspected has to say so."""
    rec = make_record(tmp_path)
    assert rec.precision_evidence is None
    assert "precisionEvidence" in rec.to_json_obj()
    assert rec.to_json_obj()["precisionEvidence"] is None


def test_precision_evidence_is_not_part_of_identity(tmp_path: Path) -> None:
    """It is derived from the bytes, which the sha256 already pins.

    Keeping it out of the digest means adding the inspection later does not
    fork an existing release's identity.
    """
    bare = make_record(tmp_path)
    inspected = make_record(
        tmp_path,
        precision_evidence=PrecisionEvidence({"FLOAT16": 63_000_000}, "fp16"),
    )
    assert bare.record_id == inspected.record_id


@pytest.mark.parametrize(
    "hist,expected",
    [
        ({"FLOAT": 120_000_000, "INT64": 900}, "fp32"),
        ({"FLOAT": 60_000, "FLOAT16": 63_000_000}, "fp16"),
        ({"FLOAT": 8_000_000, "INT8": 30_000_000, "INT64": 4_000}, "int8"),
        ({"INT64": 4_000}, None),            # nothing precision-bearing at all
        ({"FLOAT": 1_000, "INT8": 1_000}, None),  # dead heat: decline to guess
    ],
    ids=["fp32", "fp16", "int8-dynamic", "no-weights", "tie"],
)
def test_precision_inference_is_a_majority_rule_that_can_decline(hist, expected) -> None:
    assert infer_precision_from_weights(hist) == expected


def test_the_version_pattern_has_not_drifted_from_the_exporters(tmp_path: Path) -> None:
    """registry.VERSION_PATTERN is a copy; this is the anti-drift gate."""
    from export.export_yolov8 import VERSION_PATTERN as EXPORTER_PATTERN

    assert VERSION_PATTERN.pattern == EXPORTER_PATTERN.pattern


def test_precision_of_version_reads_the_suffix() -> None:
    assert precision_of_version("v2.0.1-int8") == "int8"
    with pytest.raises(ValueError):
        precision_of_version("v2.0.1-bf16")


# --------------------------------------------------------------------------
# provenance completeness
# --------------------------------------------------------------------------

def test_a_conversion_must_name_its_source_onnx(tmp_path: Path) -> None:
    """An fp16 file with no recorded source is an artifact of unknown origin."""
    with pytest.raises(ValueError, match="source ONNX sha256 is required"):
        make_record(
            tmp_path,
            conversion=Conversion("somewhere", "some_converter()", 17),
        )


def test_a_native_export_needs_no_source_onnx(tmp_path: Path) -> None:
    rec = make_record(
        tmp_path,
        role="reference",
        artifact_sha256=SHA_A,
        precision="fp32",
        version="v2.0.1-fp32",
        conversion=Conversion("export.export_rfdetr.export", NATIVE_EXPORT, 17),
    )
    assert rec.conversion.source_sha256 is None
    assert rec.conversion.is_conversion is False


def test_a_record_must_name_at_least_one_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one graph output"):
        make_record(tmp_path, output_specs=())


# --------------------------------------------------------------------------
# identity is the tuple, not the version
# --------------------------------------------------------------------------

def test_identity_ignores_the_version_string(tmp_path: Path) -> None:
    """Relabelling an artifact does not make it a different release."""
    a = make_record(tmp_path)
    b = make_record(tmp_path, version="v9.9.9-fp16")
    assert a.record_id == b.record_id


def test_identity_ignores_the_path_and_the_notes(tmp_path: Path) -> None:
    a = make_record(tmp_path)
    b = make_record(tmp_path, artifact_path="somewhere/else.onnx",
                    notes=("moved to a new bucket",))
    assert a.record_id == b.record_id


@pytest.mark.parametrize(
    "over",
    [
        {"class_list_sha256": "e" * 64},
        {"class_count": 42},
        {"iou_threshold_recommended": 0.5},
        {"artifact_bytes": 63068682},
        {"model_name": "rfdetr-m-litter"},
        {"architecture": "yolo"},
    ],
    ids=lambda o: next(iter(o)),
)
def test_identity_changes_when_any_tuple_field_changes(tmp_path: Path, over) -> None:
    base = make_record(tmp_path)
    variant = make_record(tmp_path, **over)
    assert variant.record_id != base.record_id


def test_identity_changes_with_the_recommended_threshold(tmp_path: Path) -> None:
    """A release is bytes *plus the claims made about them*.

    `confThresholdRecommended = 0.4` was reasoned about against fp32 behaviour
    and is stamped unchanged into converted artifacts. Shipping the same bytes
    with a different recommendation is a different release, and the record has
    to say so — with parity re-measured at the new operating point.
    """
    base = make_record(tmp_path)
    other = make_record(
        tmp_path,
        report=write_report(tmp_path, name="at50.json", threshold=0.5),
        conf_threshold_recommended=0.5,
    )
    assert other.record_id != base.record_id


def test_identity_changes_when_the_bytes_change(tmp_path: Path) -> None:
    """Different bytes need their own evidence, and get their own identity."""
    base = make_record(tmp_path)
    other = make_record(
        tmp_path,
        report=write_report(tmp_path, name="other.json", sha_b=SHA_C),
        artifact_sha256=SHA_C,
    )
    assert other.record_id != base.record_id


def test_identity_changes_with_the_target(tmp_path: Path) -> None:
    """Same bytes on a different runtime is a different release.

    Nothing measured on one execution provider carries to another; recording
    them as one release would launder that.
    """
    base = make_record(tmp_path)
    other = make_record(
        tmp_path,
        target=Target("onnxruntime-node", "^1.27.0", "cpu", "aws-lambda-arm64-nodejs20"),
    )
    assert other.record_id != base.record_id


def test_identity_changes_with_the_conversion_mechanism(tmp_path: Path) -> None:
    base = make_record(tmp_path)
    other = make_record(
        tmp_path,
        conversion=Conversion("export.export_yolov8", "ultralytics half=True", 17,
                              source_sha256=SHA_A),
    )
    assert other.record_id != base.record_id


def test_identity_changes_with_an_output_shape(tmp_path: Path) -> None:
    base = make_record(tmp_path)
    other = make_record(
        tmp_path,
        output_specs=(
            TensorSpec("dets", (1, 300, 4), "FLOAT"),
            TensorSpec("labels", (1, 300, 43), "FLOAT"),  # one fewer column
        ),
    )
    assert other.record_id != base.record_id


def test_identity_changes_with_the_io_dtype(tmp_path: Path) -> None:
    """keep_io_types=True is a decision; a record has to be able to show it."""
    base = make_record(tmp_path)
    other = make_record(
        tmp_path, input_spec=TensorSpec("input", (1, 3, 512, 512), "FLOAT16")
    )
    assert other.record_id != base.record_id


def test_record_id_is_the_digest_of_the_identity_block(tmp_path: Path) -> None:
    import hashlib

    rec = make_record(tmp_path)
    assert rec.record_id == hashlib.sha256(canonical_bytes(rec.identity())).hexdigest()


# --------------------------------------------------------------------------
# the store: idempotent, immutable, collision-loud
# --------------------------------------------------------------------------

def test_registering_the_same_tuple_twice_is_idempotent(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    rec = make_record(tmp_path)

    first = reg.register(rec)
    assert first.created is True
    stamped = first.record.registered_at
    assert stamped

    second = reg.register(make_record(tmp_path))
    assert second.created is False
    assert second.path == first.path
    assert second.record.registered_at == stamped  # no timestamp churn
    assert len(reg.paths()) == 1


def test_registering_different_bytes_under_the_same_version_is_rejected(tmp_path: Path) -> None:
    """The S3-key overwrite, caught at registration.

    Same model, same version string, different artifact: this is what happens
    when an int8 build is dropped at the key an fp32 release already owns.
    """
    reg = Registry(tmp_path / "registry")
    reg.register(make_record(tmp_path))

    report = write_report(tmp_path, name="other.json", sha_b=SHA_C)
    impostor = make_record(tmp_path, report=report, artifact_sha256=SHA_C)
    with pytest.raises(VersionCollisionError, match="cannot name two sets of bytes"):
        reg.register(impostor)
    assert len(reg.paths()) == 1


def test_the_same_tuple_with_a_different_body_is_refused(tmp_path: Path) -> None:
    """Records are immutable: one tuple cannot carry two sets of claims."""
    reg = Registry(tmp_path / "registry")
    reg.register(make_record(tmp_path))
    with pytest.raises(ImmutableRecordError, match="differs in: version"):
        reg.register(make_record(tmp_path, version="v2.0.2-fp16"))


def test_the_same_tuple_with_different_notes_is_refused(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    reg.register(make_record(tmp_path))
    with pytest.raises(ImmutableRecordError):
        reg.register(make_record(tmp_path, notes=("second thoughts",)))


def test_a_record_round_trips_through_json(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    written = reg.register(make_record(tmp_path)).record
    (loaded,) = reg.records()
    assert loaded.record_id == written.record_id
    assert loaded.to_json_obj() == written.to_json_obj()


def test_a_hand_edited_record_is_rejected_on_read(tmp_path: Path) -> None:
    """Content addressing is only worth something if the address is checked."""
    reg = Registry(tmp_path / "registry")
    path = reg.register(make_record(tmp_path)).path
    doc = json.loads(path.read_text())
    doc["identity"]["classes"]["count"] = 42  # the edit nobody would notice
    path.write_text(json.dumps(doc))
    with pytest.raises(RegistryError, match="has been edited"):
        reg.records()


def test_editing_the_precision_of_a_stored_record_trips_the_earlier_gate(tmp_path: Path) -> None:
    """Two gates cover the same edit; the inner one fires first.

    Retyping a stored record's precision is caught by the version/precision
    check before the content address is even compared — which is the ordering
    you want, because the message names the actual contradiction.
    """
    reg = Registry(tmp_path / "registry")
    path = reg.register(make_record(tmp_path)).path
    doc = json.loads(path.read_text())
    doc["identity"]["precision"] = "int8"
    path.write_text(json.dumps(doc))
    with pytest.raises(PrecisionMismatchError):
        reg.records()


def test_find_matches_id_prefix_version_and_model(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry")
    rec = reg.register(make_record(tmp_path)).record
    assert reg.find(rec.short_id)[0].record_id == rec.record_id
    assert reg.find("v2.0.1-fp16")[0].record_id == rec.record_id
    assert reg.find("rfdetr-s-litter")[0].record_id == rec.record_id
    assert reg.find("nope") == []


def test_an_empty_registry_reads_as_empty(tmp_path: Path) -> None:
    assert Registry(tmp_path / "nothing-here").records() == []


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def test_verify_notices_ok_changed_and_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.onnx"
    artifact.write_bytes(b"pretend this is a graph")
    report = write_report(tmp_path, sha_b=sha256_file(artifact))

    reg = Registry(tmp_path / "registry")
    rec = reg.register(
        make_record(
            tmp_path, report=report, artifact_sha256=sha256_file(artifact),
            artifact_bytes=artifact.stat().st_size, artifact_path="artifact.onnx",
        )
    ).record

    items = {i.what: i for i in reg.verify(rec, base=tmp_path)}
    assert items["artifact"].status == "ok"
    assert items["parity-report"].status == "ok"

    artifact.write_bytes(b"pretend this is a DIFFERENT graph")
    assert reg.verify(rec, base=tmp_path)[0].status == "changed"

    artifact.unlink()
    assert reg.verify(rec, base=tmp_path)[0].status == "missing"


def test_verify_also_pins_the_parity_report(tmp_path: Path) -> None:
    """A report that changed after registration is no longer the evidence."""
    artifact = tmp_path / "artifact.onnx"
    artifact.write_bytes(b"graph")
    report = write_report(tmp_path, sha_b=sha256_file(artifact))
    reg = Registry(tmp_path / "registry")
    rec = reg.register(
        make_record(
            tmp_path, report=report, artifact_sha256=sha256_file(artifact),
            artifact_bytes=artifact.stat().st_size, artifact_path="artifact.onnx",
        )
    ).record

    report.write_text(report.read_text().replace('"count": 200', '"count": 20'))
    items = {i.what: i for i in reg.verify(rec, base=tmp_path)}
    assert items["parity-report"].status == "changed"


# --------------------------------------------------------------------------
# transcription and target awareness
# --------------------------------------------------------------------------

def test_parity_ref_transcribes_the_recommended_row(tmp_path: Path) -> None:
    ref = parity_ref_from_report(write_report(tmp_path), "candidate", root=tmp_path)
    at = ref.comparison["atRecommendedThreshold"]
    assert at["threshold"] == 0.4
    assert at["thresholdFlips"] == 19
    assert at["classArgmaxFlips"] == 6
    assert ref.comparison["fixtureCount"] == 200
    assert ref.comparison["maxAbsDeltaByOutput"]["labels"] == 12.17
    # side b's latency, because this record is side b
    assert ref.comparison["latencyMsThisArtifact"]["p50"] == 589.7


def test_parity_ref_takes_the_reference_side_latency_for_a_reference_role(tmp_path: Path) -> None:
    ref = parity_ref_from_report(write_report(tmp_path), "reference", root=tmp_path)
    assert ref.artifact_sha256 == SHA_A
    assert ref.compared_against_sha256 == SHA_B
    assert ref.comparison["latencyMsThisArtifact"]["p50"] == 311.6


def test_parity_ref_rejects_an_unknown_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="role must be one of"):
        parity_ref_from_report(write_report(tmp_path), "baseline", root=tmp_path)


def test_measured_on_target_is_false_when_the_hosts_differ(tmp_path: Path) -> None:
    """The fp32 release serves on Lambda; its parity was run on a workstation."""
    rec = make_record(
        tmp_path,
        target=Target("onnxruntime-node", "^1.27.0", "cpu", "aws-lambda-arm64-nodejs20"),
    )
    assert rec.measured_on_target is False


def test_measured_on_target_is_true_only_on_an_exact_match(tmp_path: Path) -> None:
    rec = make_record(tmp_path)  # target == the report's own environment
    assert rec.measured_on_target is True
    off_by_a_version = make_record(
        tmp_path,
        target=Target("onnxruntime", "1.28.0", "CPUExecutionProvider",
                      "macOS-26.5.1-arm64-arm-64bit"),
    )
    assert off_by_a_version.measured_on_target is False


def test_class_list_hash_is_order_sensitive() -> None:
    """Logit column i is class i; a reordered list is a different contract."""
    a = class_list_sha256(["food.can", "food.bag"])
    b = class_list_sha256(["food.bag", "food.can"])
    assert a != b
    assert class_list_sha256(["food.bag", "food.can"]) == b


def test_canonical_bytes_refuses_nan() -> None:
    with pytest.raises(ValueError):
        canonical_bytes({"conf": float("nan")})


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_lists_shows_and_verifies(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "artifact.onnx"
    artifact.write_bytes(b"graph")
    report = write_report(tmp_path, sha_b=sha256_file(artifact))
    reg_dir = tmp_path / "registry"
    reg = Registry(reg_dir)
    rec = reg.register(
        make_record(
            tmp_path, report=report, artifact_sha256=sha256_file(artifact),
            artifact_bytes=artifact.stat().st_size, artifact_path="artifact.onnx",
        )
    ).record

    common = ["--registry", str(reg_dir), "--root", str(tmp_path)]
    assert main(common + ["list"]) == 0
    assert rec.short_id in capsys.readouterr().out

    assert main(common + ["show", "v2.0.1-fp16"]) == 0
    assert json.loads(capsys.readouterr().out)["recordId"] == rec.record_id

    assert main(common + ["verify", "--all"]) == 0
    assert "ok" in capsys.readouterr().out

    artifact.write_bytes(b"different graph")
    assert main(common + ["verify", "--all"]) == 1
    assert "CHANGED" in capsys.readouterr().out


def test_cli_show_exits_nonzero_when_nothing_matches(tmp_path: Path) -> None:
    assert main(["--registry", str(tmp_path / "registry"), "show", "nope"]) == 1


def test_read_onnx_facts_reads_shapes_dtypes_and_weights(tmp_path: Path) -> None:
    """Shapes, element types and the weight histogram come off the graph.

    Built as a real (tiny) fp16-weighted graph rather than mocked, because the
    thing under test is whether the reader agrees with what ONNX actually
    writes.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    import numpy as np

    w = numpy_helper.from_array(np.zeros((4, 4), dtype=np.float16), name="w")
    node = helper.make_node("Add", ["input", "w"], ["dets"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [4, 4])],
        [helper.make_tensor_value_info("dets", TensorProto.FLOAT16, [4, 4])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))

    facts = read_onnx_facts(path)
    assert facts.opset == 17
    assert facts.input_spec == TensorSpec("input", (4, 4), "FLOAT16")
    assert facts.output_specs[0].name == "dets"
    assert facts.weight_bytes_by_dtype == {"FLOAT16": 32}  # 16 elements x 2 bytes
    assert infer_precision_from_weights(facts.weight_bytes_by_dtype) == "fp16"
