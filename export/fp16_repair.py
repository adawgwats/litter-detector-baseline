"""Repair the two ways ``onnxconverter_common.float16`` breaks this graph.

Measured against onnxconverter-common 1.16.0 / onnx 1.22.0 on the
RF-DETR-Small ONNX this repo exports (see reports/CONVERSION-REPORT.md).
Both defects make the converted model **fail to load in ONNX Runtime**,
so neither is subtle once you try — the point is that nothing in this
repository ever tried.

Defect 1 — colliding, degenerate Cast nodes
    ``convert_float_to_float16`` names inserted casts
    ``<node>_input_cast<i>`` / ``<node>_output_cast<i>``. RF-DETR's own
    torch export *already* emitted casts around its TopK using that
    exact convention, so the converter regenerates names that already
    exist. The nodes it emits are additionally degenerate: their single
    output name equals their single input name, which makes them
    self-loops and gives the tensor two producers.
    ORT: ``two nodes with same node name``.

Defect 2 — orphaned ``Cast(to=FLOAT)``
    The converter rewrites tensor type annotations to FLOAT16 wholesale
    but does not update explicit ``Cast`` nodes whose ``to`` attribute
    is FLOAT. The node then produces float into a graph position
    declared float16.
    ORT: ``Type (tensor(float16)) ... does not match expected type
    (tensor(float))``.

Repairs are conservative:

*   A degenerate cast is removed only when it is a ``Cast`` with one
    input, one output, ``input[0] == output[0]``, and another node
    already produces that tensor. Removing it cannot change any
    consumer's value, because the surviving producer writes the same
    tensor name.
*   A ``Cast(to=FLOAT)`` is retyped only when its output is *not* a
    graph output (so the ``keep_io_types=True`` fp32 boundary is left
    exactly where the exporter put it) and the graph's own value_info
    declares that tensor FLOAT16 — i.e. only where the converter has
    already committed to fp16 and left the cast behind.

Nothing here is a general-purpose fp16 fixer. It repairs the two
failures this graph exhibits and refuses to pretend about the rest:
``validate_loadable`` is the actual gate, and it runs ONNX Runtime,
because a model that ``onnx.checker`` accepts can still fail to load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger("export.fp16_repair")


@dataclass
class RepairReport:
    """What the repair actually did. Empty lists mean it was a no-op."""

    removed_degenerate_casts: list[str] = field(default_factory=list)
    retyped_float_casts: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removed_degenerate_casts or self.retyped_float_casts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "removedDegenerateCasts": list(self.removed_degenerate_casts),
            "retypedFloatCasts": list(self.retyped_float_casts),
            "removedCount": len(self.removed_degenerate_casts),
            "retypedCount": len(self.retyped_float_casts),
        }


def find_duplicate_node_names(model: Any) -> dict[str, int]:
    """Node names appearing more than once, with their counts.

    A duplicate name is illegal in ONNX and ORT rejects the model
    outright, so this is a cheap pre-flight check worth running on any
    converted graph.
    """
    import collections

    counts = collections.Counter(n.name for n in model.graph.node if n.name)
    return {name: c for name, c in counts.items() if c > 1}


def repair(model: Any) -> RepairReport:
    """Apply both repairs in place. Returns what was changed."""
    from onnx import TensorProto

    graph = model.graph
    report = RepairReport()

    # --- defect 1 -----------------------------------------------------
    producer_counts: dict[str, int] = {}
    for node in graph.node:
        for out in node.output:
            producer_counts[out] = producer_counts.get(out, 0) + 1

    doomed: list[int] = []
    for i, node in enumerate(graph.node):
        if node.op_type != "Cast":
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            continue
        if node.input[0] != node.output[0]:
            continue
        if producer_counts.get(node.output[0], 0) < 2:
            # A self-loop that is the ONLY producer would make the tensor
            # undefined; that is a different bug and this function will
            # not silently paper over it.
            LOG.warning(
                "self-referential Cast %r is the sole producer of %r — "
                "leaving it alone, this is not the defect this repairs",
                node.name,
                node.output[0],
            )
            continue
        doomed.append(i)
        report.removed_degenerate_casts.append(node.name)

    for i in reversed(doomed):
        del graph.node[i]

    # --- defect 2 -----------------------------------------------------
    declared = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    graph_outputs = {o.name for o in graph.output}

    for node in graph.node:
        if node.op_type != "Cast":
            continue
        if not node.output or node.output[0] in graph_outputs:
            continue  # keep_io_types boundary — the exporter's decision, not ours
        if declared.get(node.output[0]) != TensorProto.FLOAT16:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16
                report.retyped_float_casts.append(node.name)

    return report


def validate_loadable(model_path: str, providers: list[str] | None = None) -> None:
    """Raise unless ONNX Runtime can actually build a session.

    This is the check that matters. ``onnx.checker.check_model`` passes
    on graphs ORT refuses, so a checker-only gate would have let both
    defects above ship.
    """
    import onnxruntime as ort

    ort.InferenceSession(
        str(model_path), providers=providers or ["CPUExecutionProvider"]
    )
