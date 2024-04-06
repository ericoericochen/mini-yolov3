from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, box_convert


def flatten_bboxes(
    bboxes: torch.Tensor, num_anchors: int, num_classes: int
) -> torch.Tensor:
    """
    (B, A(5 + C), H, W) -> (B, AHW, 5 + C)
    """
    B = bboxes.shape[0]

    bboxes = bboxes.view(B, -1, num_anchors, 5 + num_classes).view(
        B, -1, 5 + num_classes
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
    be in cxcywh format with confidence scores and class scores.

    Params:
        - bboxes: (AHW, 5 + C)
    """
    # select bboxes with confidence score >= threshold
    obj_mask = bboxes[..., 4].sigmoid() >= threshold
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
        ious = box_iou(
            box_convert(
                max_bbox[:, :4],
                in_fmt="cxcywh",
                out_fmt="xyxy",
            ),
            box_convert(
                remaining_bbox[:, :4],
                in_fmt="cxcywh",
                out_fmt="xyxy",
            ),
        )  # convert to xyxy

        obj_bboxes = remaining_bbox[ious.squeeze(0) < threshold]

    if len(selected_bboxes) == 0:
        return torch.empty(0, bboxes.shape[-1])

    nms_bboxes = torch.stack(selected_bboxes, dim=0)

    return nms_bboxes


def to_bbox(
    pred: torch.Tensor, anchors: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Converts model prediction to bounding box predictions by transforming t_x, t_y to x, y and t_w, t_h to w, h.

    Params:
        - pred: (B, H, W, A(C + 5)) in cxcywh format with confidence scores and class probabilities
    """

    W, H = pred.shape[2], pred.shape[1]
    A = anchors.shape[0]

    pred = pred.view(-1, H, W, A, 5 + num_classes)  # (B, H, W, A, 5 + C)

    pred_tx = pred[..., 0]  # (B, H, W, A)
    pred_ty = pred[..., 1]  # (B, H, W, A)
    pred_twh = pred[..., 2:4]  # (B, W, W, A, 2)
    pred_rest = pred[..., 4:]  # (B, H, W, A, 1 + C)

    # get c_x, c_y
    X, Y = torch.arange(0, W, device=pred.device), torch.arange(H, device=pred.device)
    x_indices, y_indices = torch.meshgrid(X, Y, indexing="xy")
    x_offsets = x_indices.unsqueeze(0).unsqueeze(-1) * 1 / W
    y_offsets = y_indices.unsqueeze(0).unsqueeze(-1) * 1 / H

    # apply sigmoid to t_x and t_y and add offset
    pred_x = pred_tx.sigmoid() * (1 / W) + x_offsets
    pred_y = pred_ty.sigmoid() * (1 / H) + y_offsets

    # apply exp to twh and multiply with anchors
    anchors_batch = anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    pred_wh = pred_twh.exp() * anchors_batch

    # concatenate t_x, t_y, t_w, t_h, conf, and class scores
    pred = torch.cat(
        [pred_x.unsqueeze(-1), pred_y.unsqueeze(-1), pred_wh, pred_rest],
        dim=-1,
    )  # (B, H, W, A(C + 5))

    return pred


@dataclass
class YoloV3Output:
    pred: torch.Tensor
    anchors: torch.Tensor
    num_classes: int

    def __post_init__(self):
        self.pred = self.pred.detach()

    @property
    def bbox_pred(self) -> torch.Tensor:
        return to_bbox(self.pred, self.anchors, self.num_classes)

    def bounding_boxes(self, threshold: float = 0.5) -> torch.Tensor:
        bboxes = to_bbox(self.pred, self.anchors, self.num_classes)

        bboxes = flatten_bboxes(
            bboxes, num_anchors=self.anchors.shape[0], num_classes=self.num_classes
        )

        selected_bboxes = non_maximum_suppression_batch(bboxes, threshold)
        boxes = []  # (cx, cy, w, h, confidence, class_scores)

        for bbox in selected_bboxes:
            cxcywh = bbox[:, :4]
            confidence = bbox[:, 4]
            class_scores = bbox[:, 5:]

            # print("bbox:", bbox)
            xywh = box_convert(cxcywh, in_fmt="cxcywh", out_fmt="xywh")
            # print("bbox: ", xywh)
            labels = torch.argmax(class_scores, dim=-1)

            box_data = {"bboxes": xywh, "confidence": confidence, "labels": labels}
            boxes.append(box_data)

        return boxes
