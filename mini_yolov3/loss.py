import torch

# from .utils import box_iou
import torch.nn as nn
import torch.nn.functional as F
from .model_output import to_bbox
from torchvision.ops import box_convert, box_iou


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

    def forward(
        self,
        pred: torch.Tensor,
        bboxes: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        """
        Params:
            - pred: model prediction (B, A(C + 5), G, G) in cxcywh format with confidence scores and class probabilities
        """
        B = pred.shape[0]
        A = self.anchors.shape[0]

        target = torch.zeros_like(pred).to(pred.device)
        # print(pred.shape)
        # print(target.shape)

        return {"loss": self.mse_loss(pred, target)}

        raise RuntimeError

        # make target tensor
        grid_size = (pred.shape[3], pred.shape[2])  # W, H
        target = build_targets(
            bboxes=bboxes,
            labels=labels,
            grid_size=grid_size,
            anchors=self.anchors,
            num_classes=self.num_classes,
        )  # (B, A(5 + C), H, W)

        # print(pred.shape, target.shape)

        # pred = pred.permute(0, 2, 3, 1)
        # target = target.permute(0, 3, 2, 1)
        # pred = pred.permute(0, 3, 2, 1)

        target = torch.zeros_like(pred).to(pred.device)
        # target = target.permute(0, 2, 3, 1)
        # raise RuntimeError

        return {"loss": self.mse_loss(pred, target)}

        # return {
        #     "loss": self.mse_loss(pred.permute(0, 2, 3, 1), target.permute(0, 2, 3, 1))
        # }

        # get mask for obj
        obj_mask = target[:, 4:5, ...] == 1  # (B, H, W)

        print(obj_mask.shape)

        # get obj pred and target
        obj_pred = pred[obj_mask]

        print(obj_pred.shape)

        raise RuntimeError

        # make global bbox pred and target
        # bbox_pred = to_bbox(pred, anchors=self.anchors, num_classes=self.num_classes)
        # bbox_target = to_bbox(
        #     target, anchors=self.anchors, num_classes=self.num_classes
        # )

        # print(bbox_pred[0, :, 0, 0], bbox_pred[0, :, 0, 1])
        # print(bbox_target[0, :, 0, 0], bbox_target[0, :, 0, 1])

        # raise RuntimeError

        return {
            "loss": self.mse_loss(pred, bbox_target),
        }

        # move pred dim to last dim
        pred = pred.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))
        target = target.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))
        bbox_target = bbox_target.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))

        # get mask for cells that contain an object
        obj_mask = target[..., 4] == 1  # (B, G, G)

        # let O = # of cells that contain an object
        obj_bbox_pred = bbox_pred[obj_mask].view(
            -1, A, 5 + self.num_classes
        )  # (O, A, 5 + C)
        obj_bbox_target = bbox_target[obj_mask].view(
            -1, A, 5 + self.num_classes
        )  # (O, A, 5 + C)

        # calculate box ious
        obj_pred_xywh = obj_bbox_pred[..., :4].view(-1, 4)  # (OA, 4)
        obj_target_xywh = obj_bbox_target[..., :4].view(-1, 4)  # (OA, 4)

        # convert xywh to xyxy
        obj_pred_xyxy = box_convert(obj_pred_xywh, in_fmt="cxcywh", out_fmt="xyxy")
        obj_target_xyxy = box_convert(obj_target_xywh, in_fmt="cxcywh", out_fmt="xyxy")

        ious = box_iou(obj_pred_xyxy, obj_target_xyxy).diag().detach()  # (OA, )
        ious = ious.view(-1, A)  # (O, A)

        max_iou, max_iou_idx = ious.max(dim=-1, keepdim=True)  # (O, 1)
        max_mask = (
            torch.arange(0, A, device=pred.device).repeat(ious.shape[0], 1)
            == max_iou_idx
        )  # (Z, A)

        # reshape pred and target to select bounding box prediction with highest iou
        obj_pred = pred[obj_mask].view(-1, A, 5 + self.num_classes)  # (O, A, 5 + C)
        obj_target = target[obj_mask].view(-1, A, 5 + self.num_classes)  # (O, A, 5 + C)

        obj_pred = obj_pred[max_mask]
        obj_target = obj_target[max_mask]

        obj_pred_txywh = obj_pred[..., :4]  # (O, 4)
        obj_target_txywh = obj_target[..., :4]  # (O, 4)

        # == coord loss ==
        obj_pred_txywh[:, :2].sigmoid_()
        obj_target_txywh[:, :2].sigmoid_()
        # print("pred: ", obj_pred_txywh)
        # print("target: ", obj_target_txywh)

        coord_loss = self.lambda_coord * self.mse_loss(obj_pred_txywh, obj_target_txywh)
        # == coord loss ==

        # return {
        #     "loss": coord_loss,
        # }

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

        class_loss = self.bce_loss(obj_pred_class, obj_target_class)
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
        # loss = coord_loss + obj_conf_loss + class_loss + noobj_loss
        loss = coord_loss

        return {
            "loss": loss,
            # "coord_loss": coord_loss,
            # "obj_conf_loss": obj_conf_loss,
            # "class_loss": class_loss,
            # "noobj_loss": noobj_loss,
        }


# class YOLOLoss(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         anchors: torch.Tensor,
#         lambda_coord: float = 1.0,
#         lambda_noobj: float = 1.0,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.anchors = anchors
#         self.lambda_coord = lambda_coord
#         self.lambda_noobj = lambda_noobj
#         self.mse_loss = nn.MSELoss()
#         self.bce_loss = nn.BCEWithLogitsLoss()

#     def forward(
#         self,
#         pred: torch.Tensor,
#         bboxes: list[torch.Tensor],
#         labels: list[torch.Tensor],
#     ):
#         """
#         Params:
#             - pred: model prediction (B, A(C + 5), G, G) in cxcywh format with confidence scores and class probabilities
#         """
#         A = self.anchors.shape[0]

#         # make target tensor
#         grid_size = (pred.shape[3], pred.shape[2])  # W, H
#         target = build_targets(
#             bboxes=bboxes,
#             labels=labels,
#             grid_size=grid_size,
#             anchors=self.anchors,
#             num_classes=self.num_classes,
#         )  # (B, A(5 + C), H, W)

#         # return {"loss": self.mse_loss(pred, target)}

#         # return {
#         #     "loss": self.mse_loss(pred.permute(0, 2, 3, 1), target.permute(0, 2, 3, 1))
#         # }

#         # get mask for obj
#         obj_mask = target[:, 4:5, ...] == 1  # (B, H, W)

#         print(obj_mask.shape)

#         # get obj pred and target
#         obj_pred = pred[obj_mask]

#         print(obj_pred.shape)

#         raise RuntimeError

#         # make global bbox pred and target
#         # bbox_pred = to_bbox(pred, anchors=self.anchors, num_classes=self.num_classes)
#         # bbox_target = to_bbox(
#         #     target, anchors=self.anchors, num_classes=self.num_classes
#         # )

#         # print(bbox_pred[0, :, 0, 0], bbox_pred[0, :, 0, 1])
#         # print(bbox_target[0, :, 0, 0], bbox_target[0, :, 0, 1])

#         # raise RuntimeError

#         return {
#             "loss": self.mse_loss(pred, bbox_target),
#         }

#         # move pred dim to last dim
#         pred = pred.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))
#         target = target.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))
#         bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))
#         bbox_target = bbox_target.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))

#         # get mask for cells that contain an object
#         obj_mask = target[..., 4] == 1  # (B, G, G)

#         # let O = # of cells that contain an object
#         obj_bbox_pred = bbox_pred[obj_mask].view(
#             -1, A, 5 + self.num_classes
#         )  # (O, A, 5 + C)
#         obj_bbox_target = bbox_target[obj_mask].view(
#             -1, A, 5 + self.num_classes
#         )  # (O, A, 5 + C)

#         # calculate box ious
#         obj_pred_xywh = obj_bbox_pred[..., :4].view(-1, 4)  # (OA, 4)
#         obj_target_xywh = obj_bbox_target[..., :4].view(-1, 4)  # (OA, 4)

#         # convert xywh to xyxy
#         obj_pred_xyxy = box_convert(obj_pred_xywh, in_fmt="cxcywh", out_fmt="xyxy")
#         obj_target_xyxy = box_convert(obj_target_xywh, in_fmt="cxcywh", out_fmt="xyxy")

#         ious = box_iou(obj_pred_xyxy, obj_target_xyxy).diag().detach()  # (OA, )
#         ious = ious.view(-1, A)  # (O, A)

#         max_iou, max_iou_idx = ious.max(dim=-1, keepdim=True)  # (O, 1)
#         max_mask = (
#             torch.arange(0, A, device=pred.device).repeat(ious.shape[0], 1)
#             == max_iou_idx
#         )  # (Z, A)

#         # reshape pred and target to select bounding box prediction with highest iou
#         obj_pred = pred[obj_mask].view(-1, A, 5 + self.num_classes)  # (O, A, 5 + C)
#         obj_target = target[obj_mask].view(-1, A, 5 + self.num_classes)  # (O, A, 5 + C)

#         obj_pred = obj_pred[max_mask]
#         obj_target = obj_target[max_mask]

#         obj_pred_txywh = obj_pred[..., :4]  # (O, 4)
#         obj_target_txywh = obj_target[..., :4]  # (O, 4)

#         # == coord loss ==
#         obj_pred_txywh[:, :2].sigmoid_()
#         obj_target_txywh[:, :2].sigmoid_()
#         # print("pred: ", obj_pred_txywh)
#         # print("target: ", obj_target_txywh)

#         coord_loss = self.lambda_coord * self.mse_loss(obj_pred_txywh, obj_target_txywh)
#         # == coord loss ==

#         # return {
#         #     "loss": coord_loss,
#         # }

#         # == confidence loss ==
#         obj_pred_conf = obj_pred[..., 4]  # (O, )
#         obj_target_conf = torch.ones_like(obj_pred_conf, device=pred.device)
#         obj_conf_loss = self.bce_loss(
#             obj_pred_conf, obj_target_conf
#         )  # negative log likelihood

#         # == confidence loss ==

#         # == class loss ==
#         obj_pred_class = obj_pred[..., 5:]  # (O, C)
#         obj_target_class = obj_target[..., 5:]  # (O, C)

#         class_loss = self.bce_loss(obj_pred_class, obj_target_class)
#         # == class loss ==

#         # == noobj conf loss
#         # noobj mask
#         noobj_mask = ~obj_mask
#         noobj_pred_conf = pred[noobj_mask][:, 4]
#         noobj_target_conf = torch.zeros_like(noobj_pred_conf, device=pred.device)
#         noobj_loss = self.lambda_noobj * self.bce_loss(
#             noobj_pred_conf, noobj_target_conf
#         )

#         # == noobj conf loss
#         # loss = coord_loss + obj_conf_loss + class_loss + noobj_loss
#         loss = coord_loss

#         return {
#             "loss": loss,
#             # "coord_loss": coord_loss,
#             # "obj_conf_loss": obj_conf_loss,
#             # "class_loss": class_loss,
#             # "noobj_loss": noobj_loss,
#         }


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

    # target tensor where each cell contains (tx, ty, tw, th, confidence, class scores, ... repeat)
    # target = torch.zeros(B, H, W, A * (5 + num_classes)).to(bboxes[0].device)
    target = torch.zeros(B, A * (5 + num_classes), H, W).to(bboxes[0].device)

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # convert bbox to cxcywh format
        bbox = box_convert(bbox, in_fmt="xywh", out_fmt="cxcywh")

        # print(bbox)

        N = bbox.shape[0]  # number of bounding boxes

        # get the cell each bbox belongs to
        xy = bbox[:, :2]  # (N, 2)
        wh = bbox[:, 2:]  # (N, 2)
        cell_ij = (xy // cell_size).int()

        # print(cell_ij)

        # c_x, c_y from the paper
        offsets = cell_ij * cell_size

        eps = 1e-6
        # eps = 1e-8
        # txy = -torch.log(1 / (xy - offsets + eps) - 1)  # t_x, t_y (N, 2)
        txy = -torch.log(1 / ((xy - offsets) / cell_size + eps) - 1)  # t_x, t_y (N, 2)
        # txy = xy - offsets

        # calculate t_w, t_h
        twh = torch.log(
            wh.unsqueeze(1) / anchors.unsqueeze(0)
        )  # (N, 1, 2), (1, A, 2) -> (N, A, 2)

        # repeat txy for the number of anchors
        txy = txy.unsqueeze(1).repeat(1, A, 1)  # (N, 1, 2) -> (N, A, 2)

        # construct confidence scores
        confidence = torch.ones(N, A, 1).to(bbox.device)

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

        # target[i, cell_ij[:, 1], cell_ij[:, 0], :] = target_value  # (N, A(5 + C))
        target[i, :, cell_ij[:, 1], cell_ij[:, 0]] = target_value.T  # (A(5 + C), N)

    return target
