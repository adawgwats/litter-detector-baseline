"""Unit tests for export.parity and export.fp16_repair.

Two things are being protected here.

First, the decode in `export.parity` must stay a faithful mirror of the
backend's `postprocess-detr.ts`. If it drifts, the harness reports parity
for a decode nobody runs, which is worse than reporting nothing: it
would be a green light derived from fiction. The tests below pin the
behaviours that mirror is made of — the trailing background column, the
argmax-then-sigmoid ordering, cxcywh -> xyxy with clamping.

Second, `export.fp16_repair` must fire on the pathology it claims to
repair and stay hands-off otherwise. A repair that silently rewrites a
healthy graph is a worse defect than the one it fixes.

Deliberately absent: any test asserting an absolute divergence
tolerance. The harness does not own a pass/fail threshold, so there is
nothing here to assert one against. The regression gate is tested
instead, because "no worse than the last measurement" is a claim that
can be checked without inventing a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from export.parity import (
    check_regression,
    decode_detr_queries,
    read_fixture_manifest,
)


# --------------------------------------------------------------------------
# decode — must mirror dregsbane-web-backend postprocess-detr.ts
# --------------------------------------------------------------------------

def _logits(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)[None, ...]


def _boxes(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)[None, ...]


def test_decode_applies_sigmoid_to_the_winning_logit() -> None:
    q = decode_detr_queries(_boxes([[0.5, 0.5, 0.2, 0.2]]), _logits([[2.0, -1.0]]), 2)
    assert q.best_class[0] == 0
    assert q.confidence[0] == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_decode_ignores_the_trailing_background_column() -> None:
    """rf-detr exports classCount+1 columns; column C is NOT a class.

    If the background column were included in the argmax, a query the
    model considers background would be reported as whichever real class
    happens to sit at that index — a silent mislabel.
    """
    # Background column has the largest logit by far; it must not win.
    q = decode_detr_queries(
        _boxes([[0.5, 0.5, 0.2, 0.2]]),
        _logits([[0.5, 0.1, 99.0]]),
        2,  # two NAMED classes, three columns
    )
    assert q.best_class[0] == 0
    assert q.confidence[0] == pytest.approx(1 / (1 + np.exp(-0.5)))


def test_decode_accepts_exactly_class_count_columns_too() -> None:
    q = decode_detr_queries(_boxes([[0.5, 0.5, 0.2, 0.2]]), _logits([[0.5, 0.1]]), 2)
    assert q.best_class[0] == 0


def test_decode_rejects_a_column_count_it_cannot_interpret() -> None:
    with pytest.raises(ValueError, match="class channels"):
        decode_detr_queries(
            _boxes([[0.5, 0.5, 0.2, 0.2]]), _logits([[0.1, 0.2, 0.3, 0.4]]), 2
        )


def test_decode_converts_cxcywh_to_xyxy() -> None:
    q = decode_detr_queries(_boxes([[0.5, 0.5, 0.4, 0.2]]), _logits([[1.0]]), 1)
    np.testing.assert_allclose(q.boxes_xyxy[0], [0.3, 0.4, 0.7, 0.6], rtol=1e-6)


def test_decode_clamps_boxes_to_the_unit_square() -> None:
    """A box wider than the image must not produce negative coordinates."""
    q = decode_detr_queries(_boxes([[0.5, 0.5, 2.0, 2.0]]), _logits([[1.0]]), 1)
    np.testing.assert_allclose(q.boxes_xyxy[0], [0.0, 0.0, 1.0, 1.0])


def test_decode_is_query_aligned() -> None:
    """Query order is preserved, which is what makes A/B comparison exact."""
    q = decode_detr_queries(
        _boxes([[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]),
        _logits([[5.0, 0.0], [0.0, 5.0]]),
        2,
    )
    assert list(q.best_class) == [0, 1]
    assert q.confidence[0] == pytest.approx(q.confidence[1])


# --------------------------------------------------------------------------
# fixture manifest
# --------------------------------------------------------------------------

def test_manifest_ignores_comments_and_blank_lines(tmp_path) -> None:
    p = tmp_path / "fx.txt"
    p.write_text("# header\n\n/a/b.jpg\n\n# note\n/c/d.jpg\n", encoding="utf-8")
    # as_posix(), not str(): on Windows str(Path("/a/b.jpg")) is "\\a\\b.jpg"
    # and this assertion fails for a reason that has nothing to do with
    # comment or blank-line handling, which is what it is meant to test.
    assert [x.as_posix() for x in read_fixture_manifest(p)] == ["/a/b.jpg", "/c/d.jpg"]


def test_manifest_with_no_images_is_an_error(tmp_path) -> None:
    p = tmp_path / "fx.txt"
    p.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lists no images"):
        read_fixture_manifest(p)


# --------------------------------------------------------------------------
# regression gate — the only gate this harness owns
# --------------------------------------------------------------------------

def _report(max_abs: float, flips: int) -> dict:
    return {
        "rawOutputs": [
            {
                "name": "labels",
                "maxAbsDelta": max_abs,
                "meanAbsDelta": max_abs / 10,
                "p99AbsDelta": max_abs / 2,
            }
        ],
        "decode": {
            "byThreshold": [
                {"threshold": 0.4, "thresholdFlips": flips, "classArgmaxFlips": 0}
            ]
        },
    }


def test_identical_report_is_not_a_regression() -> None:
    r = _report(1.0, 10)
    assert check_regression(r, r) == []


def test_improvement_is_not_a_regression() -> None:
    assert check_regression(_report(0.5, 5), _report(1.0, 10)) == []


def test_small_worsening_inside_the_margin_passes() -> None:
    """5% worse with a 10% margin is noise, not a regression."""
    assert check_regression(_report(1.05, 10), _report(1.0, 10), margin=0.10) == []


def test_divergence_growth_beyond_the_margin_fails() -> None:
    failures = check_regression(_report(1.5, 10), _report(1.0, 10), margin=0.10)
    assert failures
    assert any("maxAbsDelta regressed" in f for f in failures)


def test_more_threshold_flips_fails() -> None:
    failures = check_regression(_report(1.0, 40), _report(1.0, 10), margin=0.10)
    assert any("thresholdFlips regressed" in f for f in failures)


def test_a_missing_output_in_the_baseline_is_reported_not_ignored() -> None:
    """Silently passing on an output the baseline lacks would let a graph
    change sneak a new tensor past the gate."""
    baseline = {"rawOutputs": [], "decode": {"byThreshold": []}}
    failures = check_regression(_report(1.0, 10), baseline)
    assert any("absent from baseline" in f for f in failures)


def test_zero_baseline_does_not_make_every_comparison_fail() -> None:
    """A perfectly-matching baseline (0 divergence) must stay gatable."""
    assert check_regression(_report(0.0, 0), _report(0.0, 0)) == []


# --------------------------------------------------------------------------
# fp16 repair
# --------------------------------------------------------------------------

def _degenerate_cast_model():
    """A graph with the exact pathology onnxconverter-common produces:
    a real Cast, plus a second Cast with the SAME NAME whose output name
    equals its input name."""
    from onnx import TensorProto, helper

    real = helper.make_node(
        "Cast", ["x"], ["t"], name="/m/TopK_input_cast0", to=TensorProto.FLOAT
    )
    degenerate = helper.make_node(
        "Cast", ["t"], ["t"], name="/m/TopK_input_cast0", to=TensorProto.FLOAT
    )
    ident = helper.make_node("Identity", ["t"], ["y"], name="/m/out")
    graph = helper.make_graph(
        [real, degenerate, ident],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        value_info=[helper.make_tensor_value_info("t", TensorProto.FLOAT16, [1])],
    )
    return helper.make_model(graph)


def test_duplicate_node_names_are_detected() -> None:
    from export.fp16_repair import find_duplicate_node_names

    dupes = find_duplicate_node_names(_degenerate_cast_model())
    assert dupes == {"/m/TopK_input_cast0": 2}


def test_repair_removes_the_degenerate_cast_and_clears_the_duplicate() -> None:
    from export.fp16_repair import find_duplicate_node_names, repair

    model = _degenerate_cast_model()
    rep = repair(model)
    assert rep.removed_degenerate_casts == ["/m/TopK_input_cast0"]
    assert find_duplicate_node_names(model) == {}
    # The surviving producer of "t" is the real cast, not the self-loop.
    producers = [n.name for n in model.graph.node if "t" in n.output]
    assert producers == ["/m/TopK_input_cast0"]


def test_repair_retypes_only_non_output_casts() -> None:
    """The keep_io_types fp32 boundary at a graph OUTPUT must survive.

    That boundary is a deliberate export decision (the backend feeds and
    reads float32); a repair that "helpfully" pushed it to fp16 would
    change the artifact's interface.
    """
    from onnx import TensorProto

    from export.fp16_repair import repair

    model = _degenerate_cast_model()
    repair(model)
    by_name = {n.name: n for n in model.graph.node}
    # "/m/TopK_input_cast0" writes "t", declared FLOAT16 in value_info and
    # not a graph output -> retyped.
    to_attr = {a.name: a.i for a in by_name["/m/TopK_input_cast0"].attribute}
    assert to_attr["to"] == TensorProto.FLOAT16


def test_repair_is_a_no_op_on_a_healthy_graph() -> None:
    from onnx import TensorProto, helper

    from export.fp16_repair import repair

    node = helper.make_node("Relu", ["x"], ["y"], name="relu")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1])],
    )
    rep = repair(helper.make_model(graph))
    assert not rep.changed


def test_repair_leaves_a_sole_producer_self_loop_alone(caplog) -> None:
    """A self-cast that is the only producer is a DIFFERENT bug.

    Deleting it would leave the tensor undefined. The repair must warn
    and decline rather than "fix" something it does not understand.
    """
    from onnx import TensorProto, helper

    from export.fp16_repair import repair

    bad = helper.make_node("Cast", ["t"], ["t"], name="lonely", to=TensorProto.FLOAT)
    graph = helper.make_graph(
        [bad],
        "g",
        [helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])],
    )
    with caplog.at_level("WARNING"):
        rep = repair(helper.make_model(graph))
    assert rep.removed_degenerate_casts == []
    assert "sole producer" in caplog.text
