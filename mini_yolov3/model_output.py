from dataclasses import dataclass
import torch


def to_bbox(
    pred: torch.Tensor, anchors: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Converts model prediction to bounding box predictions by transforming t_x, t_y to x, y and t_w, t_h to w, h.
    """
    assert pred.shape[-2] == pred.shape[-1]

    G = pred.shape[-1]
    A = anchors.shape[0]

    # print("to bbox pred")

    pred = pred.permute(0, 2, 3, 1).view(
        -1, G, G, A, num_classes + 5
    )  # (B, A(C + 5), G, G) -> (B, G, G, A(C + 5)) -> (B, G, G, A, 5 + C)

    pred_tx = pred[..., 0]  # (B, G, G, A)
    pred_ty = pred[..., 1]  # (B, G, G, A)
    pred_twh = pred[..., 2:4]  # (B, G, G, A, 2)
    pred_rest = pred[..., 4:]  # (B, G, G, A, 1 + C)

    # print(
    #     "pred_tx.shape",
    #     pred_tx.shape,
    #     "pred_ty.shape",
    #     pred_ty.shape,
    #     "pred_twh.shape",
    #     pred_twh.shape,
    # )

    cell_size = 1 / G
    x_offsets = (
        torch.arange(G).repeat_interleave(A).view(G, A).repeat(G, 1, 1).unsqueeze(0)
    ) * cell_size
    y_offsets = x_offsets.transpose(1, 2)

    # print(x_offsets)
    # print(y_offsets)

    # print("x_offsets.shape:", x_offsets.shape, "y_offsets.shape:", y_offsets.shape)

    # apply sigmoid to t_x and t_y and add offset
    pred_x = pred_tx.sigmoid() + x_offsets
    pred_y = pred_ty.sigmoid() + y_offsets

    # apply exp to twh and multiply with anchors
    # print("anchors.shape:", anchors.shape, "pred_twh.shape:", pred_twh.shape)
    anchors_batch = anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # print("anchors_batch.shape:", anchors_batch.shape)

    pred_wh = pred_twh.exp() * anchors_batch

    # print(pred_twh)
    # concatenate t_x, t_y, t_w, t_h, conf, and class scores
    pred = torch.cat(
        [
            pred_x.unsqueeze(-1),
            pred_y.unsqueeze(-1),
            pred_wh,
            pred_rest,
        ],
        dim=-1,
    )

    # print("pred.shape:", pred.shape)

    pred = pred.view(-1, G, G, A * (5 + num_classes)).permute(
        0, 3, 1, 2
    )  # (B, G, G, A(C + 5)) -> (B, A(C + 5), G, G)

    # print("pred.shape:", pred.shape)

    return pred


@dataclass
class YoloV3Output:
    pred: torch.Tensor
    anchors: torch.Tensor
    num_classes: int

    @property
    def bbox_pred(self) -> torch.Tensor:
        return to_bbox(self.pred, self.anchors, self.num_classes)
