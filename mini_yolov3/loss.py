import torch
from .utils import coco_to_xywh, box_iou, xywh_to_xyxy, xyxy_to_xywh
import torch.nn as nn
from typing import Literal
import torch.nn.functional as F
from .model_output import to_bbox


class YOLOLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        anchors: torch.Tensor,
        lambda_coord: float = 1.0,
        lambda_noobj: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        pred: torch.Tensor,
        bboxes: list[torch.Tensor],
        labels: list[torch.Tensor],
        bbox_format: Literal["coco", "xywh", "xyxy"] = "coco",
    ):
        A = self.anchors.shape[0]
        G = pred.shape[2]

        assert pred.shape[3] == pred.shape[2] == G

        # make global bbox pred
        bbox_pred = to_bbox(pred, anchors=self.anchors, num_classes=self.num_classes)

        # make target tensor
        target = build_targets(
            bboxes=bboxes,
            labels=labels,
            grid_size=G,
            anchors=self.anchors,
            num_classes=self.num_classes,
            bbox_format=bbox_format,
        )  # (B, A(5 + C), G, G)

        bbox_target = to_bbox(
            target, anchors=self.anchors, num_classes=self.num_classes
        )

        print(
            "pred.shape:",
            pred.shape,
            "bbox_pred.shape:",
            bbox_pred.shape,
            "target.shape:",
            target.shape,
            "bbox_target.shape:",
            bbox_target.shape,
        )

        # print(bbox_pred[0, :, 1, 1])
        # print(bbox_target[0, :, 1, 1])

        # move pred dim to last dim
        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # (B, G, G, A(5 + C))
        bbox_target = bbox_target.permute(0, 2, 3, 1)  # (B, G, G, A(5 + C))

        print(target.shape, "target.shape")

        # == coord loss ==
        # get mask for cells that contain an object
        obj_mask = target[..., 4] == 1  # (B, G, G)

        print("obj_mask.shape:", obj_mask.shape, obj_mask)

        # let O = # of cells that contain an object
        obj_bbox_pred = bbox_pred[obj_mask].view(
            -1, A, 5 + self.num_classes
        )  # (O, A, 5 + C)
        obj_bbox_target = bbox_target[obj_mask].view(
            -1, A, 5 + self.num_classes
        )  # (O, A, 5 + C)

        print(
            "obj_bbox_pred.shape",
            obj_bbox_pred.shape,
            "obj_bbox_target.shape",
            obj_bbox_target.shape,
        )

        # calculate box ious
        obj_pred_xywh = obj_bbox_pred[..., :4].view(-1, 4)  # (OA, 4)
        obj_target_xywh = obj_bbox_target[..., :4].view(-1, 4)  # (OA, 4)

        print(
            "obj_pred_xywh.shape",
            obj_pred_xywh.shape,
            "obj_target_xywh.shape",
            obj_target_xywh.shape,
        )

        ious = box_iou(obj_pred_xywh, obj_target_xywh, format="xywh")  # (OA, )
        ious = ious.view(-1, A)  # (O, A)

        max_iou, max_iou_idx = ious.max(dim=-1, keepdim=True)  # (O, 1)
        max_mask = torch.arange(0, A).repeat(ious.shape[0], 1) == max_iou_idx  # (Z, A)

        # reshape pred and target to select bounding box prediction with highest iou
        # pred = pred.view(-1, G, G, A, 5 + self.num_classes)  # (B, G, G, A, 5 + C)
        # target = target.view(-1, G, G, A, 5 + self.num_classes)  # (B, G, G, A, 5 + C)

        obj_pred = pred[obj_mask].view(-1, A, 5 + self.num_classes)
        obj_target = target[obj_mask].view(-1, A, 5 + self.num_classes)

        print("obj_pred.shape", obj_pred.shape, "obj_target.shape", obj_target.shape)

        obj_pred = obj_pred[max_mask]
        obj_target = obj_target[max_mask]

        obj_pred_txywh = obj_pred[..., :4]  # (O, 4)
        obj_target_txywh = obj_target[..., :4]  # (O, 4)

        coord_loss = self.lambda_coord * self.mse_loss(obj_pred_txywh, obj_target_txywh)
        # == coord loss ==

        # == confidence loss ==
        obj_pred_conf = obj_pred[..., 4]  # (O, )
        obj_target_conf = torch.ones_like(obj_pred_conf, device=pred.device)
        obj_conf_loss = self.bce_loss(
            obj_pred_conf, obj_target_conf
        )  # negative log likelihood

        # == confidence loss ==

        # == class loss ==
        obj_pred_class = obj_pred[..., 5:]  # (O, C)
        obj_target_class = obj_target[..., 5:]  # (O, C)

        print(obj_pred_class, obj_target_class)

        class_loss = self.bce_loss(obj_pred_class, obj_target_class)

        print(class_loss)

        # == class loss ==

        # == noobj conf loss
        # noobj mask
        noobj_mask = ~obj_mask
        noobj_pred_conf = pred[noobj_mask][:, 4]
        noobj_target_conf = torch.zeros_like(noobj_pred_conf, device=pred.device)
        noobj_loss = self.lambda_noobj * self.bce_loss(
            noobj_pred_conf, noobj_target_conf
        )

        # == noobj conf loss

        loss = coord_loss + obj_conf_loss + class_loss + noobj_loss

        return loss


def build_targets(
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    grid_size: int,
    anchors: torch.Tensor,
    num_classes: int,
    bbox_format: Literal["coco", "xyxy", "xywh"] = "coco",
):
    """
    Construct Target for Loss Calculation

    Returns: (B, A * 6, G, G)
    """
    A = anchors.shape[0]
    B = len(bboxes)

    # print("building targets")
    # print("A", A)

    target = torch.zeros(B, A * (5 + num_classes), grid_size, grid_size)
    cell_size = 1 / grid_size

    # print("cell size:", cell_size)

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        N = bbox.shape[0]  # number of bounding boxes

        # convert bbox to xywh
        if bbox_format == "coco":
            bbox = coco_to_xywh(bbox)
        elif bbox_format == "xyxy":
            bbox = xyxy_to_xywh(bbox)

        # get the cell each bbox belongs to
        xy = bbox[:, :2]  # (N, 2)
        wh = bbox[:, 2:]  # (N, 2)
        cell_ij = (xy // cell_size).int()

        # print("xy", xy)

        # c_x, c_y from the paper
        offsets = cell_ij * cell_size

        # print("cell and offsets")
        # print(cell_ij, offsets)

        eps = 1e-8
        txy = -torch.log(1 / (xy - offsets + eps) - 1)  # t_x, t_y (N, 2)

        # calculate t_w, t_h
        twh = torch.log(
            wh.unsqueeze(1) / anchors.unsqueeze(0)
        )  # (N, 1, 2), (1, A, 2) -> (N, A, 2)

        # repeat txy for the number of anchors
        txy = txy.unsqueeze(1).repeat(1, A, 1)  # (N, 1, 2) -> (N, A, 2)

        # construct confidence scores
        confidence = torch.ones(N, A, 1)

        # construct one hot vectors for class labels
        class_labels = F.one_hot(label.long(), num_classes).float()  # (N, C)
        class_labels = class_labels.unsqueeze(1).repeat(
            1, A, 1
        )  # (N, 1, C) -> (N, A, C)

        # print("class_labels.shape:", class_labels.shape)

        # construct target value
        # print(txy.shape, twh.shape, confidence.shape, class_labels.shape)
        target_value = torch.cat(
            [txy, twh, confidence, class_labels], dim=-1
        )  # (N, A, 5 + C)
        target_value = target_value.view(N, -1)  # (N, A(5 + C))

        # print("target_value.shape:", target_value.shape)

        assert target_value.shape == torch.Size([2, A * (5 + num_classes)])

        target[i, :, cell_ij[:, 1], cell_ij[:, 0]] = target_value.T  # (A(5 + C), N)

    return target
