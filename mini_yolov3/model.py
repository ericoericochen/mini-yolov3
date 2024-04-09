import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import TypedDict
from torchvision.ops import nms, box_convert


class ConvLayer(nn.Module):
    """
    Convolutional Layer: Conv2d -> BatchNorm -> LeakyReLU
    """

    def __init__(self, *args):
        """
        Params:
            - *args: args to nn.Conv2d
        """

        super().__init__()
        out_channels = args[1]

        self.conv = nn.Sequential(
            nn.Conv2d(*args, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BoundingBoxPrediction(TypedDict):
    bboxes: torch.Tensor
    confidence: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


@dataclass
class YOLOInferenceOutput:
    pred: torch.Tensor
    bboxes: list[BoundingBoxPrediction]


class YOLO(nn.Module):
    """
    YOLOv1 Implementation https://arxiv.org/abs/1506.02640
    """

    def __init__(
        self,
        image_size: int,
        S: int,
        B: int,
        C: int,
        backbone: list,
        dense_layers: list[int],
    ):
        super().__init__()
        self.image_size = image_size
        self.S = S
        self.B = B
        self.C = C
        self.backbone = backbone
        self.dense_layers = dense_layers

        assert len(backbone) > 0, "Backbone must have at least one layer"
        assert len(dense_layers) > 0, "Dense layers must have at least one layer"

        self.backbone_layers = self._build_backbone()
        self.feedforward = self._build_feedforward()

    def _build_backbone(self):
        backbone = []

        for layer in self.backbone:
            module_type = layer[0]
            module_args = layer[1]

            if module_type == "Conv":
                backbone.append(ConvLayer(*module_args))
            elif module_type == "MaxPool":
                backbone.append(nn.MaxPool2d(*module_args))

        return nn.Sequential(*backbone)

    def _build_feedforward(self):
        feedforward = []

        for i in range(1, len(self.dense_layers)):
            in_channels = self.dense_layers[i - 1]
            out_channels = self.dense_layers[i]

            feedforward.append(nn.Linear(in_channels, out_channels))
            feedforward.append(nn.LeakyReLU(0.1))

        in_channels = self.dense_layers[-1]
        out_channels = self.S * self.S * (5 * self.B + self.C)

        feedforward.append(nn.Linear(in_channels, out_channels))

        return nn.Sequential(*feedforward)

    @property
    def config(self):
        return {
            "image_size": self.image_size,
            "S": self.S,
            "B": self.B,
            "C": self.C,
            "backbone": self.backbone,
            "dense_layers": self.dense_layers,
        }

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> YOLOInferenceOutput:
        """
        Predict bounding boxes for input images

        Params:
            - images: (B, C, H, W) input images
            - confidence_threshold: minimum confidence score to keep a bounding box
            - iou_threshold: threshold for non-maximum suppression
        """
        pred = self(images)
        cell_size = 1 / self.S
        bounding_boxes = []

        for pred_bbox in pred:
            pred_bbox = pred_bbox.permute(1, 2, 0)
            bbox = pred_bbox[..., : -self.C].view(self.S, self.S, self.B, 5)
            bbox_conf = bbox[..., 0]
            bbox_x = bbox[..., 1]
            bbox_y = bbox[..., 2]
            bbox_wh = bbox[..., 3:]

            # scale bounding box pred to global coordinates
            x_offsets, y_offsets = torch.meshgrid(
                torch.arange(self.S, device=pred_bbox.device),
                torch.arange(self.S, device=pred_bbox.device),
                indexing="xy",
            )

            bbox_x = (bbox_x + x_offsets.unsqueeze(-1)) * cell_size
            bbox_y = (bbox_y + y_offsets.unsqueeze(-1)) * cell_size

            # combine xywh
            bbox_xywh = torch.cat(
                [bbox_x.unsqueeze(-1), bbox_y.unsqueeze(-1), bbox_wh], dim=-1
            )

            bbox_xywh = bbox_xywh.contiguous().view(-1, 4)
            bbox_conf = bbox_conf.contiguous().view(-1)

            # keep boxes above a confidence threshold
            keep_mask = bbox_conf >= confidence_threshold
            bbox_xywh = bbox_xywh[keep_mask]
            bbox_conf = bbox_conf[keep_mask]

            pred_classes = (
                pred_bbox[..., -self.C :].repeat(1, 1, self.B).view(-1, self.C)
            )
            pred_classes = pred_classes[keep_mask]

            # non-maximum suppression
            bbox_xyxy = box_convert(bbox_xywh, in_fmt="cxcywh", out_fmt="xyxy")
            nms_idx = nms(bbox_xyxy, bbox_conf, iou_threshold)

            # process kept boxes
            bbox = bbox_xywh[nms_idx]
            conf = bbox_conf[nms_idx].clip(0, 1)
            classes = pred_classes[nms_idx]
            labels = classes.argmax(dim=-1)
            scores = classes.max(dim=-1).values.clip(0, 1)

            prediction: BoundingBoxPrediction = {
                "bboxes": bbox,
                "confidence": conf,
                "labels": labels,
                "scores": scores,
            }
            bounding_boxes.append(prediction)

        return YOLOInferenceOutput(pred, bounding_boxes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == x.shape[-2] == self.image_size, "Invalid image size"

        x = self.backbone_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.feedforward(x)
        x = x.view(
            -1,
            5 * self.B + self.C,
            self.S,
            self.S,
        ).abs()

        return x
