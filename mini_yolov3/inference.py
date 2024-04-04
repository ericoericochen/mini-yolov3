import torch
import torch.nn.functional as F
from .model import MiniYOLOV3
from typing import Union
from PIL import Image


def flatten_bboxes(
    bboxes: torch.Tensor, num_anchors: int, num_classes: int
) -> torch.Tensor:
    """
    (B, A(5 + C), G, G) -> (B, AGG, 5 + C)
    """
    B = bboxes.shape[0]

    bboxes = (
        bboxes.permute(0, 2, 3, 1)
        .view(B, -1, num_anchors, 5 + num_classes)
        .view(B, -1, 5 + num_classes)
    )  # (B, G, G, A(5 + C)) -> (B, G^2 * A, 5 + C) -> (B, AG^2, 5 + C)

    return bboxes


class YoloV3Pipeline:
    pass
