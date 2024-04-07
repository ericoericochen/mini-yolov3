import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou
import pdb


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
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        bboxes: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        """
        Params:
            - pred: model prediction (B, H, W, A(C + 5)) in cxcywh format with confidence scores and class probabilities
        """
        A = self.anchors.shape[0]

        # make target tensor
        grid_size = (pred.shape[1], pred.shape[2])  # W, H
        target = build_targets(
            bboxes=bboxes,
            labels=labels,
            grid_size=grid_size,
            anchors=self.anchors,
            num_classes=self.num_classes,
        )  # (B, H, W, A(5 + C))

        pred = pred.contiguous().view(-1, 5 + self.num_classes)
        target = target.view(-1, 5 + self.num_classes)

        # get mask for cells that contain an object
        obj_mask = target[..., 4] == 1  # (B, H, W)

        # get obj pred and target
        obj_pred = pred[obj_mask]
        obj_target = target[obj_mask]

        # coord loss
        obj_pred_txywh = obj_pred[..., :4]  # (O, 4)
        obj_target_txywh = obj_target[..., :4]  # (O, 4)

        coord_loss = self.lambda_coord * self.mse_loss(obj_pred_txywh, obj_target_txywh)

        # confidence loss
        obj_pred_conf = obj_pred[..., 4]  # (O, )
        target_pred_conf = torch.ones_like(obj_pred_conf, device=pred.device)

        obj_conf_loss = self.bce_loss(obj_pred_conf, target_pred_conf)

        # class loss
        obj_pred_class = obj_pred[..., 5:]  # (O, C)
        obj_target_class = obj_target[..., 5:]  # (O, C)

        class_loss = self.cross_entropy(obj_pred_class, obj_target_class)

        # noobj conf loss
        noobj_mask = ~obj_mask
        noobj_pred_conf = pred[noobj_mask][:, 4]
        noobj_target_conf = torch.zeros_like(noobj_pred_conf, device=pred.device)

        noobj_loss = self.lambda_noobj * self.bce_loss(
            noobj_pred_conf, noobj_target_conf
        )

        # total loss
        loss = coord_loss + obj_conf_loss + noobj_loss + class_loss

        return (
            loss,
            {
                "coord_loss": coord_loss,
                "obj_conf_loss": obj_conf_loss,
                "class_loss": class_loss,
                "noobj_loss": noobj_loss,
            },
        )


def build_targets(
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    grid_size: tuple[int, int],
    anchors: torch.Tensor,
    num_classes: int,
):
    """
    Construct Target for Loss Calculation

    Params:
        - bboxes: list of bounding boxes in xywh format
        - grid_size: tuple of grid size (W, H)

    Returns: (B, A * 6, G, G)
    """
    A = anchors.shape[0]
    B = len(bboxes)
    W, H = grid_size

    grid_dim = torch.tensor(grid_size, device=bboxes[0].device)
    cell_size = 1 / grid_dim

    target = torch.zeros(B, H, W, A * (5 + num_classes)).to(bboxes[0].device)

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # convert bbox to cxcywh format
        bbox = box_convert(bbox, in_fmt="xywh", out_fmt="cxcywh")

        N = bbox.shape[0]  # number of bounding boxes

        # get the cell each bbox belongs to
        xy = bbox[:, :2]  # (N, 2)
        wh = bbox[:, 2:]  # (N, 2)
        cell_ij = (xy // cell_size).int()

        # c_x, c_y from the paper
        offsets = cell_ij * cell_size

        eps = 1e-8
        txy = -torch.log(1 / ((xy - offsets) / cell_size + eps) - 1)  # t_x, t_y (N, 2)

        # calculate t_w, t_h
        twh = torch.log(
            wh.unsqueeze(1) / anchors.unsqueeze(0)
        )  # (N, 1, 2), (1, A, 2) -> (N, A, 2)

        # repeat txy for the number of anchors
        txy = txy.unsqueeze(1).repeat(1, A, 1)  # (N, 1, 2) -> (N, A, 2)

        # get the best bounding box prior -> the twh closest to 0
        sum_twh = twh.abs().sum(dim=-1)
        min_twh_idx = sum_twh.argmin(dim=-1)

        # construct confidence scores
        confidence = torch.zeros(N, A, 1).to(bbox.device)
        confidence[:, min_twh_idx] = 1

        # construct one hot vectors for class labels
        class_labels = F.one_hot(label.long(), num_classes).float()  # (N, C)
        class_labels = class_labels.unsqueeze(1).repeat(
            1, A, 1
        )  # (N, 1, C) -> (N, A, C)

        # construct target value
        target_value = torch.cat(
            [txy, twh, confidence, class_labels], dim=-1
        )  # (N, A, 5 + C)
        target_value = target_value.view(N, -1)  # (N, A(5 + C))

        assert target_value.shape == torch.Size([N, A * (5 + num_classes)])

        try:
            target[i, cell_ij[:, 1], cell_ij[:, 0], :] = target_value  # (N, A(5 + C))
        except:
            # invalid bounding box DO NOT COUNT
            target[i] = -100  # ignore index

    return target
