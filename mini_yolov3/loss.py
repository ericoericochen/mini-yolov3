import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou


class YOLOLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        anchors: torch.Tensor,
        lambda_coord: float = 0.05,
        lambda_conf: float = 1.0,
        lambda_cls: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf
        self.lambda_cls = lambda_cls

        self.mse_loss = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        pred: list[torch.Tensor],
        bboxes: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        """
        Params:
            - pred: model prediction (B, H, W, A(C + 5)) in cxcywh format with confidence scores and class probabilities
        """
        targets = build_targets(pred, bboxes, labels, self.anchors, self.num_classes)
        target = torch.cat(
            [target.view(-1, 5 + self.num_classes) for target in targets], dim=0
        )  # (X, 5 + C)

        # target = build_targets_(
        #     bboxes,
        #     labels,
        #     (pred[0].shape[2], pred[0].shape[1]),
        #     self.anchors,
        #     self.num_classes,
        # ).view(-1, 5 + self.num_classes)

        pred = torch.cat(
            [
                pred_item.contiguous().view(-1, 5 + self.num_classes)
                for pred_item in pred
            ],
            dim=0,
        )

        # get mask for cells that contain an object
        obj_mask = target[..., 4] == 1  # (B, H, W)

        # get obj pred and target
        obj_pred = pred[obj_mask]
        obj_target = target[obj_mask]

        # coord loss
        obj_pred_txywh = obj_pred[..., :4]  # (O, 4)
        obj_target_txywh = obj_target[..., :4]  # (O, 4)

        coord_loss = self.lambda_coord * self.mse_loss(obj_pred_txywh, obj_target_txywh)

        coord_loss = torch.tensor(0.0, device=coord_loss.device)

        # confidence loss
        pred_conf = pred[..., 4]  # (O, )
        target_conf = target[..., 4]

        mask = target_conf == 1
        conf_loss = self.bce_loss(
            pred_conf[mask], target_conf[mask]
        ) + 0.5 * self.bce_loss(pred_conf[~mask], target_conf[~mask])

        # conf_loss = self.lambda_conf * self.bce(pred_conf.sigmoid(), target_conf)
        # conf_loss = self.lambda_conf * self.bce_loss(pred_conf, target_conf)

        # class loss
        obj_pred_class = obj_pred[..., 5:]  # (O, C)
        obj_target_class = obj_target[..., 5:]  # (O, C)

        # class_loss = self.lambda_cls * self.cross_entropy(
        #     obj_pred_class, obj_target_class
        # )
        # class_loss = self.lambda_cls * self.bce(
        #     obj_pred_class.sigmoid(), obj_target_class
        # )
        class_loss = self.lambda_cls * self.bce_loss(obj_pred_class, obj_target_class)

        # total loss
        loss = coord_loss + conf_loss + class_loss

        return (
            loss,
            {
                "coord_loss": coord_loss,
                "conf_loss": conf_loss,
                "class_loss": class_loss,
            },
        )


# NOTE: there may be a more vectorized way to do this, this is the
# most readable way I've been able to do it :)
def build_targets(
    pred: list[torch.Tensor],
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    anchors: torch.Tensor,
    num_classes: int,
):
    device = pred[0].device
    A = anchors.shape[0]
    B = pred[0].shape[0]
    S = len(pred)

    num_anchors_per_scale = A // S

    targets = [
        torch.zeros(
            B,
            pred_item.shape[1],
            pred_item.shape[2],
            num_anchors_per_scale,
            (5 + num_classes),
            device=device,
        )
        for pred_item in pred
    ]

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # convert bbox to cxcywh format
        bbox = box_convert(bbox, in_fmt="xywh", out_fmt="cxcywh")
        N = bbox.shape[0]  # number of bounding boxes

        # find the bounding box prior
        wh = bbox[..., 2:]

        # get the anchor with the closest aspect ratio
        prior_diff = wh.unsqueeze(1) - anchors.unsqueeze(0)  # (N, A, 2)
        prior_idx = prior_diff.abs().sum(dim=-1).argmin(dim=-1)  # (N, A) -> (N, )
        target_idx = prior_idx // num_anchors_per_scale

        # convert each bounding box to target value
        for j, (idx, p_idx) in enumerate(zip(target_idx, prior_idx)):
            target = targets[idx.item()][i]  # get correct target by idx and batch idx

            a_idx = p_idx % num_anchors_per_scale
            anchor = anchors[p_idx]
            grid_size = target.shape[1]
            curr_bbox = bbox[j]

            # convert xywh to txywh
            xy = curr_bbox[:2]
            wh = curr_bbox[2:]

            cell_size = 1 / grid_size
            cell_ij = (xy // cell_size).int()

            # c_x, c_y from the paper
            offsets = cell_ij * cell_size

            # compute txy by inverting equation in paper
            eps = 1e-8
            txy = -torch.log(
                1 / ((xy - offsets) / cell_size + eps) - 1
            )  # t_x, t_y (2, )

            # compute twh by inverting equation in paper
            twh = (wh / anchor).log()  # (2, )

            confidence = torch.tensor([1], device=device)
            curr_label = label[j]
            class_labels = torch.zeros(num_classes, device=device)
            class_labels[curr_label.long()] = 1

            target_value = torch.cat(
                [txy, twh, confidence, class_labels],
                dim=0,
            )

            try:
                target[cell_ij[1], cell_ij[0], a_idx] = target_value
            except:
                # invalid bounding box DO NOT COUNT
                pass

    return targets


def build_targets_(
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

        try:
            target[i, cell_ij[:, 1], cell_ij[:, 0], :] = target_value  # (N, A(5 + C))
        except:
            pass

    return target
