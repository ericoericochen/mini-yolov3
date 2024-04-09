import torch
import torch.nn as nn
from torchvision.ops import box_convert, box_iou
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(
        self,
        S: int,
        B: int,
        C: int,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse_loss = nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, bboxes: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Params:
            - pred: (B, 5B + C, S, S)
            - bboxes: list of bounding boxes in xywh format (COCO)
            - labels: list of labels
        """
        target = build_target(bboxes, labels, self.S, self.C)

        # permute pred and target
        pred = pred.permute(0, 2, 3, 1)  # (N, S, S, 5B + C)
        target = target.permute(0, 2, 3, 1)  # (N, S, S, 5 + C)

        # seperate bounding box predictions and class predictions
        pred_bboxes = pred[..., : -self.C]  # (N, S, S, 5B)
        pred_classes = pred[..., -self.C :]  # (N, S, S, C)
        target_bboxes = target[..., : -self.C]  # (N, S, S, 5)
        target_classes = target[..., -self.C :]  # (N, S, S, C)

        # select bounding box responsible for prediction by getting the one with the highest iou
        pred_bboxes = pred_bboxes.contiguous().view(-1, 5)  # (NBS^2, 5)
        target_bboxes = target_bboxes.repeat(1, 1, 1, self.B).view(-1, 5)  # (NBS^2, 5)

        obj_mask = target_bboxes[..., 0] == 1

        obj_pred_bboxes = pred_bboxes[obj_mask]
        obj_target_bboxes = target_bboxes[obj_mask]

        # calculate iou
        obj_pred_xywh = obj_pred_bboxes[..., 1:]
        obj_target_xywh = obj_target_bboxes[..., 1:]

        iou = box_iou(
            box_convert(obj_pred_xywh, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(obj_target_xywh, in_fmt="cxcywh", out_fmt="xyxy"),
        ).diag()  # (NBS^2,)

        # select the bounding box with the highest iou
        iou = iou.view(-1, self.B)  # (NS^2, B)
        max_iou, max_iou_idx = iou.max(dim=-1, keepdim=True)  # (NS^2, 1)

        indices = torch.arange(0, self.B, device=pred.device).repeat(iou.shape[0], 1)

        responsible_mask = (indices == max_iou_idx).view(-1)  # (NS^2,)

        # coord loss
        obj_pred_xywh = obj_pred_xywh[responsible_mask]
        obj_target_xywh = obj_target_xywh[responsible_mask]

        obj_pred_xy = obj_pred_xywh[..., :2]
        obj_pred_wh = obj_pred_xywh[..., 2:]

        obj_target_xy = obj_target_xywh[..., :2]
        obj_target_wh = obj_target_xywh[..., 2:]

        assert (
            obj_pred_xy.shape == obj_target_xy.shape
            and obj_pred_wh.shape == obj_target_wh.shape
        )
        coord_loss = self.lambda_coord * (
            self.mse_loss(obj_pred_xy, obj_target_xy)
            + self.mse_loss(obj_pred_wh.sqrt(), obj_target_wh.sqrt())
        )

        # confidence loss
        obj_pred_conf = obj_pred_bboxes[..., 0][responsible_mask]
        obj_target_conf = obj_target_bboxes[..., 0][responsible_mask]

        assert obj_pred_conf.shape == obj_target_conf.shape
        obj_conf_loss = self.mse_loss(obj_pred_conf * max_iou, obj_target_conf)

        # noobj loss
        noobj_mask = ~obj_mask
        noobj_pred_conf = pred_bboxes[..., 0][noobj_mask]
        noobj_target_conf = target_bboxes[..., 0][noobj_mask]

        assert noobj_pred_conf.shape == noobj_target_conf.shape
        noobj_loss = self.lambda_noobj * self.mse_loss(
            noobj_pred_conf, noobj_target_conf
        )

        # class loss
        pred_classes = pred_classes.contiguous().view(-1, self.C)
        target_classes = target_classes.contiguous().view(-1, self.C)

        obj_mask = (target[..., 0] == 1).view(-1)
        obj_pred_classes = pred_classes[obj_mask]
        obj_target_classes = target_classes[obj_mask]

        assert obj_pred_classes.shape == obj_target_classes.shape
        class_loss = self.mse_loss(obj_pred_classes, obj_target_classes)

        # total loss
        loss = coord_loss + obj_conf_loss + noobj_loss + class_loss

        return loss, {
            "coord_loss": coord_loss.item(),
            "obj_conf_loss": obj_conf_loss.item(),
            "noobj_loss": noobj_loss.item(),
            "class_loss": class_loss.item(),
        }


def build_target(
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    S: int,
    C: int,
) -> torch.Tensor:
    """
    Params:
        - bboxes: list of bounding boxes in xywh format (COCO)
        - labels: list of labels
        - S: size of the yolo detection grid S
        - C: number of classes
    """

    B = len(bboxes)
    device = bboxes[0].device

    target = torch.zeros(
        B,
        5 + C,
        S,
        S,
        device=device,
    )

    cell_size = 1 / S
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox = box_convert(bbox, in_fmt="xywh", out_fmt="cxcywh")
        N = bbox.shape[0]

        # get cell index
        xy = bbox[:, :2]
        wh = bbox[:, 2:]
        cell_ij = (xy // cell_size).int()  # (N, 2)

        # get relative xy
        xy = (xy - cell_ij * cell_size) / cell_size
        bbox = torch.cat([xy, wh], dim=-1)

        # construct target value (c, x, y, w, h, classes)
        confidence = torch.ones(N, 1, device=device)  # (N, 1)
        class_labels = F.one_hot(label.long(), C).to(torch.float32)
        target_value = torch.cat([confidence, bbox, class_labels], dim=-1)  # (N, 5 + C)

        target[i, :, cell_ij[:, 1], cell_ij[:, 0]] = target_value.T  # (5 + C, N)

    return target
