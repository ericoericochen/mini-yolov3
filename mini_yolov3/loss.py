import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert
import pdb


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
        A = self.anchors.shape[0]
        S = len(pred)  # number of scales

        # targets = []
        # anchors = self.anchors.chunk(S)
        # for anchor, pred_item in zip(anchors, pred):
        #     grid_size = pred_item.shape[1]
        #     target = build_targets(
        #         bboxes=bboxes,
        #         labels=labels,
        #         grid_size=grid_size,
        #         anchors=anchor,
        #         num_classes=self.num_classes,
        #     )
        #     targets.append(target)

        targets = build_targets_new(
            pred, bboxes, labels, self.anchors, self.num_classes
        )

        target = torch.cat(
            [target.view(-1, 5 + self.num_classes) for target in targets], dim=0
        )  # (X, 5 + C)

        # target = build_targets_new(pred, bboxes, labels, self.anchors, self.num_classes)

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

        # print("obj_pred", obj_pred.shape)
        # print("obj_target", obj_target.shape)

        # raise RuntimeError

        # coord loss
        obj_pred_txywh = obj_pred[..., :4]  # (O, 4)
        obj_target_txywh = obj_target[..., :4]  # (O, 4)

        coord_loss = self.lambda_coord * self.mse_loss(obj_pred_txywh, obj_target_txywh)
        # raise RuntimeError

        # confidence loss
        pred_conf = pred[..., 4]  # (O, )
        target_conf = target[..., 4]

        conf_loss = self.lambda_conf * self.bce_loss(pred_conf, target_conf)

        # class loss
        obj_pred_class = obj_pred[..., 5:]  # (O, C)
        obj_target_class = obj_target[..., 5:]  # (O, C)

        # class_loss = self.lambda_cls * self.cross_entropy(
        #     obj_pred_class, obj_target_class
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
def build_targets_new(
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

    # print("anchors_per_scale", num_anchors_per_scale)

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

            # print(idx)

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

            # print(xy)
            # print(txy)

            # compute twh by inverting equation in paper
            # print(anchor)
            twh = (wh / anchor).log()  # (2, )

            confidence = torch.tensor([1], device=device)
            curr_label = label[j]
            class_labels = torch.zeros(num_classes, device=device)
            class_labels[curr_label.long()] = 1

            target_value = torch.cat(
                [txy, twh, confidence, class_labels],
                dim=0,
            )

            # print(cell_ij)
            # print(target_value)

            # print(xy)
            # print(cell_ij)
            # print(target.shape)
            try:
                target[cell_ij[1], cell_ij[0], a_idx] = target_value
            except:
                # invalid bounding box DO NOT COUNT
                pass
                # target[cell_ij[1], cell_ij[0], a_idx] = -100

            # target[]

        # targets[]

        # raise RuntimeError
    return targets


def build_targets(
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    grid_size: int,
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

    cell_size = 1 / grid_size
    target = torch.zeros(
        B, grid_size, grid_size, A * (5 + num_classes), device=bboxes[0].device
    )

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
        confidence = torch.zeros(N, A, 1, device=bbox.device)
        confidence[:, min_twh_idx] = 1

        # construct one hot vectors for class labels
        class_labels = torch.zeros(N, A, num_classes, device=bbox.device)  # (N, A, C)
        class_labels[
            torch.arange(0, N, dtype=torch.long, device=bbox.device),
            min_twh_idx,
            label.long(),
        ] = 1  # Assign 1 to the class label of the best bounding box prior

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
