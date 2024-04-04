from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


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

        max_bbox = max_bbox.unsqueeze(0)
        ious = box_iou(max_bbox, remaining_bbox)

        obj_bboxes = remaining_bbox[ious < threshold]

    nms_bboxes = torch.stack(selected_bboxes, dim=0)

    return nms_bboxes


def to_bbox(
    pred: torch.Tensor, anchors: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Converts model prediction to bounding box predictions by transforming t_x, t_y to x, y and t_w, t_h to w, h.

    Params:
        - pred: (B, A(C + 5), H, W) in cxcywh format with confidence scores and class probabilities
    """

    W, H = pred.shape[3], pred.shape[2]
    A = anchors.shape[0]

    pred = pred.permute(0, 2, 3, 1).view(
        -1, H, W, A, num_classes + 5
    )  # (B, A(C + 5), H, W) -> (B, H, W, A(C + 5)) -> (B, H, W, A, 5 + C)

    pred_tx = pred[..., 0]  # (B, H, W, A)
    pred_ty = pred[..., 1]  # (B, H, W, A)
    pred_twh = pred[..., 2:4]  # (B, W, W, A, 2)
    pred_rest = pred[..., 4:]  # (B, H, W, A, 1 + C)

    # get c_x, c_y
    X, Y = torch.arange(0, W), torch.arange(H)
    x_indices, y_indices = torch.meshgrid(X, Y, indexing="xy")
    x_offsets = x_indices.unsqueeze(0).unsqueeze(-1) * 1 / W
    y_offsets = y_indices.unsqueeze(0).unsqueeze(-1) * 1 / H

    # apply sigmoid to t_x and t_y and add offset
    pred_x = pred_tx.sigmoid() + x_offsets
    pred_y = pred_ty.sigmoid() + y_offsets

    # apply exp to twh and multiply with anchors
    anchors_batch = anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    pred_wh = pred_twh.exp() * anchors_batch

    # concatenate t_x, t_y, t_w, t_h, conf, and class scores
    pred = torch.cat(
        [pred_x.unsqueeze(-1), pred_y.unsqueeze(-1), pred_wh, pred_rest],
        dim=-1,
    )

    pred = pred.view(-1, H, W, A * (5 + num_classes)).permute(
        0, 3, 1, 2
    )  # (B, H, W, A(C + 5)) -> (B, A(C + 5), H, W)

    return pred


@dataclass
class YoloV3Output:
    pred: torch.Tensor
    anchors: torch.Tensor
    num_classes: int

    @property
    def bbox_pred(self) -> torch.Tensor:
        return to_bbox(self.pred, self.anchors, self.num_classes)

    @property
    def bounding_boxes(self) -> torch.Tensor:
        pass
