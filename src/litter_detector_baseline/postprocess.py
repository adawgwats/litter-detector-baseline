"""Post-processing for YOLO-family detector outputs.

Handles:

- YOLOv8-style output tensor parsing (shape ``[N, num_classes + 4, num_anchors]``)
- Confidence filtering
- Non-maximum suppression (NMS)
- Scaling boxes back to the original image dimensions
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from litter_detector_baseline.types import Detection


def yolov8_decode(
    output: np.ndarray,
    class_names: Sequence[str],
    score_threshold: float,
    iou_threshold: float,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list[Detection]:
    """Decode raw YOLOv8 output to a list of Detection objects.

    Args:
        output: model output of shape ``[1, 4+num_classes, num_anchors]``.
            First 4 channels are box coordinates (cx, cy, w, h) in model
            input pixel space. Remaining channels are per-class scores.
        class_names: ordered class names indexed by class_id.
        score_threshold: minimum class score for a box to be kept.
        iou_threshold: IoU threshold for non-maximum suppression.
        input_size: (height, width) of the model input.
        original_size: (height, width) of the original input image.

    Returns:
        Detections in original image pixel space, sorted by score desc.
    """
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(
            f"Expected YOLOv8 output shape [1, 4+C, A], got {output.shape}"
        )

    output = output[0]  # [4+C, A]
    num_classes = output.shape[0] - 4
    if num_classes != len(class_names):
        raise ValueError(
            f"Model outputs {num_classes} classes but config provides "
            f"{len(class_names)} class names"
        )

    boxes_cxcywh = output[:4].T  # [A, 4]
    scores_per_class = output[4:].T  # [A, C]

    # Best class per anchor
    class_ids = np.argmax(scores_per_class, axis=1)
    scores = scores_per_class[np.arange(scores_per_class.shape[0]), class_ids]

    keep_mask = scores >= score_threshold
    if not np.any(keep_mask):
        return []

    boxes_cxcywh = boxes_cxcywh[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)
    boxes_xyxy = _scale_to_original(
        boxes_xyxy, input_size=input_size, original_size=original_size
    )

    # NMS, run per-class to avoid suppressing different-class overlaps.
    kept_indices: list[int] = []
    for cid in np.unique(class_ids):
        class_mask = class_ids == cid
        idx = np.where(class_mask)[0]
        nms_kept = _nms(boxes_xyxy[idx], scores[idx], iou_threshold)
        kept_indices.extend(idx[nms_kept].tolist())

    detections = [
        Detection(
            class_id=int(class_ids[i]),
            class_name=class_names[int(class_ids[i])],
            score=float(scores[i]),
            x1=float(boxes_xyxy[i, 0]),
            y1=float(boxes_xyxy[i, 1]),
            x2=float(boxes_xyxy[i, 2]),
            y2=float(boxes_xyxy[i, 3]),
        )
        for i in kept_indices
    ]
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy


def _scale_to_original(
    boxes: np.ndarray,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> np.ndarray:
    """Scale boxes from model input space back to original image space.

    Assumes the preprocessor letterboxed (preserves aspect ratio) the
    image. If your preprocessor used plain resize without letterbox,
    pass ``original_size = input_size`` to skip rescaling.
    """
    in_h, in_w = input_size
    orig_h, orig_w = original_size

    if (in_h, in_w) == (orig_h, orig_w):
        return boxes

    scale = min(in_w / orig_w, in_h / orig_h)
    pad_x = (in_w - orig_w * scale) / 2.0
    pad_y = (in_h - orig_h * scale) / 2.0

    boxes = boxes.copy()
    boxes[:, 0] = (boxes[:, 0] - pad_x) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_y) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_x) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_y) / scale

    # Clamp to image bounds.
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, orig_h)
    return boxes


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=int)
