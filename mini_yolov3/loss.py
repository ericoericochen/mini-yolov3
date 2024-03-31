import torch
from .utils import coco_to_xywh, box_iou, xywh_to_xyxy, xyxy_to_xywh
import torch.nn as nn
from typing import Literal


class YOLOLoss_(nn.Module):
    def __init__(
        self, num_anchors: int = 2, lambda_coord: float = 1.0, lambda_noobj: float = 1.0
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self, pred: torch.Tensor, bboxes: list[torch.Tensor], labels: list[torch.Tensor]
    ):
        """
        Params:
            - pred: (B, A * (5 + C), H, W)

        A - number of anchors
        C - number of classes
        """
        pred = pred.clone()

        dim = pred.shape[1]
        grid_size = pred.shape[2]
        num_classes = dim // self.num_anchors - 5

        # construct target tensor from bounding boxes and labels
        target = build_target(bboxes, labels, grid_size)

        # scale relative x, y to global x, y
        cell_size = 1 / grid_size
        x_indices = list(range(0, dim, num_classes + 5))
        x_offsets = (
            torch.arange(0, grid_size).repeat(grid_size, 1) * cell_size
        )  # (H, W)

        y_indices = list(range(1, dim, num_classes + 5))
        y_offsets = x_offsets.T

        pred[:, x_indices, :, :] += x_offsets
        pred[:, y_indices, :, :] += y_offsets

        # get mask for cells that contain an object
        target = target.permute(0, 2, 3, 1).view(-1, 6)  # (B, H, W, 6) -> (BHW, 6)

        obj_mask = target[:, 4] == 1  # (BHW, )
        obj_target = target[obj_mask]  # (Z, 6)

        pred = pred.permute(0, 2, 3, 1).view(
            -1, dim
        )  # (B, H, W, A * (5 + C)) -> (BHW, A * (5 + C))

        # get the predictions for the cells that contain an object
        obj_pred = pred[obj_mask]  # (Z, A * (5 + C))
        obj_pred = obj_pred.view(-1, self.num_anchors, num_classes + 5)  # (Z, A, C + 5)

        bboxes = obj_pred[:, :, :4]  # (Z, A, 4)
        target_bboxes = (
            obj_target[:, :4].unsqueeze(1).repeat(1, self.num_anchors, 1)
        )  # (Z, A, 4)

        # convert xywh to xyxy
        bboxes_xyxy = xywh_to_xyxy(bboxes.view(-1, 4))  # (ZA, 4)
        target_bboxes_xyxy = xywh_to_xyxy(target_bboxes.view(-1, 4))

        # calculate iou
        ious = box_iou(bboxes_xyxy, target_bboxes_xyxy)  # (ZA, 4)
        ious = ious.view(-1, self.num_anchors)  # (Z, A)

        # print("ious")
        # print(ious.shape)
        # print(ious)

        # select prediction with highest iou
        best_iou, best_iou_idx = ious.max(dim=1, keepdim=True)  # (Z, )
        max_mask = (
            torch.arange(0, self.num_anchors).repeat(ious.shape[0], 1) == best_iou_idx
        )  # (Z, A)

        # print("max mask")
        # print(max_mask)

        # print("best iou")
        # print(best_iou.shape)

        # print("obj pred")
        # print(obj_pred.shape)

        obj_pred = obj_pred[max_mask]  # (Z, C + 5)

        # print("time to calculate loss")
        # print(obj_pred)
        # print(obj_target)

        # coord loss
        obj_pred_xy = obj_pred[:, :2]  # (Z, 4)
        obj_target_xy = obj_target[:, :2]  # (Z, 4)

        obj_pred_wh_sqrt = obj_pred[:, 2:4] ** 0.5
        obj_target_wh_sqrt = obj_target[:, 2:4] ** 0.5

        coord_loss = self.lambda_coord * (
            self.mse_loss(obj_pred_xy, obj_target_xy)
            + self.mse_loss(obj_pred_wh_sqrt, obj_target_wh_sqrt)
        )

        # object confidence loss
        obj_pred_conf = obj_pred[:, 4]
        obj_conf_loss = (
            -obj_pred_conf.log().mean()
        )  # negative log likelihood (logistic loss)

        # print(obj_conf_loss)

        # loss for class prediction
        # print("[class loss]")

        class_scores = obj_pred[:, 5:]
        target_class = obj_target[:, 5].to(torch.long)

        class_loss = self.cross_entropy_loss(class_scores, target_class)

        # print(class_scores.shape, target_class.shape)
        # print(class_loss)

        # print("[class loss]")

        # loss for no object
        noobj_mask = ~obj_mask
        # print(noobj_mask.shape)
        noobj_pred = pred[noobj_mask].view(
            -1, self.num_anchors, num_classes + 5
        )  # (K, A, C + 5)
        noobj_pred_conf = noobj_pred[:, :, 4]  # (K, A)

        # print(noobj_pred_conf.shape)
        # print(noobj_pred_conf)

        noobj_loss = (
            self.lambda_noobj * -(1 - noobj_pred_conf).log().mean()
        )  # negative log likelihood

        # print(noobj_loss)

        loss = coord_loss + obj_conf_loss + class_loss + noobj_loss

        return loss


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

        # make target tensor
        target = build_targets(
            bboxes=bboxes,
            labels=labels,
            grid_size=G,
            anchors=self.anchors,
            bbox_format=bbox_format,
        )  # (B, 4 + 2A, G, G)

        target = target.permute(0, 2, 3, 1)  # (B, G, G, 6)

        print("target.shape:", target.shape)
        print("pred.shape", pred.shape)

        # == coord loss ==

        # get mask for cells that contain an object
        # obj_mask = target[:, -2, :, :] == 1  # (B, G, G)
        obj_mask = target[:, :, :, -2] == 1  # (B, G, G)

        # convert tx, ty, tw, th to xywh
        pred = pred.permute(0, 2, 3, 1).view(
            -1, G, G, A, self.num_classes + 5
        )  # (B, G, G, A * (5 + C)) -> (B, G, G, A, 5 + C)

        pred_txy = pred[:, :, :, :, :2]  # (B, G, G, A, 2)
        pred_twh = pred[:, :, :, :, 2:4]  # (B, G, G, A, 2)
        pred_class_pred = pred[:, :, :, :, 5:]  # (B, G, G, A, C)

        # relative xy to cell
        pred_xy = pred_txy.sigmoid()

        # convert tw, th to wh
        pred_wh = pred_twh.exp() * self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(
            0
        )  # (B, G, G, A, 2)

        # let O = # of cells that contain an object
        obj_pred_xy = pred_xy[obj_mask]  # (O, A, 2)
        obj_pred_wh = pred_wh[obj_mask]  # (O, A, 2)

        print(
            "obj_pred_xy.shape",
            obj_pred_xy.shape,
            "obj_pred_wh.shape",
            obj_pred_wh.shape,
        )
        print(obj_pred_xy, obj_pred_wh)

        # get target bounding boxes
        obj_target = target[obj_mask]  # (O, 6)
        obj_txy = obj_target[:, :2]  # (O, 2)
        obj_twh = obj_target[:, 2:-2].view(-1, A, 2)  # (O, A, 2)

        print("obj_target.shape", obj_target.shape, "obj_twh.shape", obj_twh.shape)

        # calculate box ious
        obj_pred_xywh = torch.cat((obj_pred_xy, obj_pred_wh), dim=-1)  # (O, A, 4)
        obj_pred_xywh = obj_pred_xywh.view(-1, 4)  # (O * A, 4)

        print("obj_pred_xywh.shape", obj_pred_xywh.shape)

        # == coord loss ==

        # noobj mask
        noobj_mask = ~obj_mask

        # class prediction loss


def build_targets(
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    grid_size: int,
    anchors: torch.Tensor,
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

    target = torch.zeros(B, A * 2 + 4, grid_size, grid_size)
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
        )  # (N, 2), (A, 2) -> (N, A, 2)

        twh = twh.view(-1, A * 2)  # (Z, A * 2)

        # construct target value (t_x, t_y, (t_w, t_h) * A, 1, label)
        confidence = torch.ones(N, 1)
        label = label.unsqueeze(1)

        target_value = torch.cat((txy, twh, confidence, label), dim=1)

        assert target_value.shape == torch.Size([2, 2 + A * 2 + 2])

        target[i, :, cell_ij[:, 1], cell_ij[:, 0]] = target_value.T  # (A*2+4, N)

    return target
