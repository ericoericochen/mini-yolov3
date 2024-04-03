import torch
from .utils import box_iou
import torch.nn.functional as F


def flatten_bboxes(
    bboxes: torch.Tensor, num_anchors: int, num_classes: int
) -> torch.Tensor:
    """
    (B, A(5 + C), G, G) -> (B, AGG, 5 + C)
    """
    B = bboxes.shape[0]

    bboxes = (
        bboxes.permute(0, 2, 3, 1)
        .view(B, -1, num_anchors, 5 + num_classes)
        .view(B, -1, 5 + num_classes)
    )  # (B, G, G, A(5 + C)) -> (B, G^2 * A, 5 + C) -> (B, AG^2, 5 + C)

    return bboxes


def non_maximum_suppression_batch(bboxes: torch.Tensor, threshold: float = 0.5):
    results = []

    for bbox in bboxes:
        results.append(non_maximum_suppression(bbox, threshold))

    return results


def non_maximum_suppression(bboxes: torch.Tensor, threshold: float = 0.5):
    """
    Performs non-maximum suppression on the bounding boxes. Bounding boxes must
    be in xywh format with confidence scores and class scores.

    Params:
        - bboxes: (AGG, 5 + C)
    """
    # select bboxes with confidence score >= threshold
    obj_mask = bboxes[..., 4] >= threshold
    obj_bboxes = bboxes[obj_mask]

    selected_bboxes = []
    while obj_bboxes.shape[0] > 0:
        # select the bbox with the highest confidence score
        max_idx = torch.argmax(obj_bboxes[..., 4])
        max_bbox = obj_bboxes[max_idx]

        selected_bboxes.append(max_bbox)

        # calculate ious and remove bboxes that have iou >= threshold
        remaining_mask = ~F.one_hot(max_idx, obj_bboxes.shape[0]).bool()
        remaining_bbox = obj_bboxes[remaining_mask]

        max_bbox = max_bbox.unsqueeze(0).repeat(remaining_bbox.shape[0], 1)
        ious = box_iou(max_bbox, remaining_bbox)

        obj_bboxes = remaining_bbox[ious < threshold]

    nms_bboxes = torch.stack(selected_bboxes, dim=0)

    return nms_bboxes
