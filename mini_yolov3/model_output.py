from dataclasses import dataclass
import torch


def to_bbox_pred(
    pred: torch.Tensor, anchors: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Converts model prediction to bounding box predictions by transforming t_x, t_y to x, y and t_w, t_h to w, h.
    """
    assert pred.shape[-2] == pred.shape[-1]

    G = pred.shape[-1]
    A = anchors.shape[0]

    print("to bbox pred")

    # pred = pred.permute(0, 2, 3, 1).view(
    #     -1, G, G, A, num_classes + 5
    # )  # (B, A(C + 5), G, G) -> (B, G, G, A(C + 5)) -> (B, G, G, A, 5 + C)

    # pred_tx = pred[..., 0]  # (B, G, G, A)
    # pred_ty = pred[..., 1]  # (B, G, G, A)
    # pred_twh = pred[..., 2:4]  # (B, G, G, A, 2)

    # print(
    #     "pred_tx.shape",
    #     pred_tx.shape,
    #     "pred_ty.shape",
    #     pred_ty.shape,
    #     "pred_twh.shape",
    #     pred_twh.shape,
    # )

    x_offsets = torch.arange(G).repeat(G, 1).repeat(A, 1, 1)

    print(x_offsets.shape)
    print(x_offsets[:, 0, 0])


@dataclass
class YoloV3Output:
    pred: torch.Tensor
    anchors: torch.Tensor
    num_classes: int

    @property
    def bbox_pred(self) -> torch.Tensor:
        return to_bbox_pred(self.pred, self.anchors, self.num_classes)
