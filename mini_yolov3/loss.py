import torch
from .utils import coco_to_xywh
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, pred: torch.Tensor, bboxes: list[torch.Tensor], labels: list[torch.Tensor]
    ):
        pass


def build_target(
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    grid_size: int,
):
    """
    Assume COCO for now

    bbox should be normalized [0, 1]
    """
    assert len(bboxes) == len(labels), "Mismatch b/w images in `bbox` and `labels`"

    B = len(bboxes)

    # create target)
    target = torch.zeros((B, 6, grid_size, grid_size))
    cell_size = 1 / grid_size

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        L = label.shape[0]

        xywh = coco_to_xywh(bbox)
        xy = xywh[:, [0, 1]]
        cell_ij = (xy // cell_size).to(torch.int)

        # value at a cell (b_x, b_y, b_w, h_h, 1, label)
        value = torch.cat(
            [xywh, torch.ones(L, 1), label.unsqueeze(1)], dim=1
        ).T  # (6, L)

        target[i, :, cell_ij[:, 1], cell_ij[:, 0]] = value

    return target
