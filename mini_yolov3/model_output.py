from dataclasses import dataclass
import torch


def to_bbox_pred(
    pred: torch.Tensor,
    anchors: torch.Tensor,
):
    """
    Converts model prediction to bounding box predictions by transforming t_x, t_y to x, y and t_w, t_h to w, h.
    """

    pass


@dataclass
class YoloV3Output:
    pred: torch.Tensor

    @property
    def bbox_pred(self):
        pass
