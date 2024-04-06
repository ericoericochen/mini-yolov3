import torch.nn as nn
import torch
from typing import Union
import torch.nn.functional as F
from .loss import YOLOLoss
from dataclasses import dataclass
from torchvision.ops import nms, box_convert


class Downsample(nn.Module):
    """
    Downsamples spatial resolution by 2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
        )

    def forward(self, x: torch.Tensor):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
        )

    def forward(self, x: torch.Tensor):
        return self.upsample(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        mid_channels = channels // 2

        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        x = x + h  # residual connection

        return x


class DetectionLayer(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()

        dim = num_anchors * (5 + num_classes)
        self.detection = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.detection(x)


@dataclass
class MiniYoloV3Output:
    pred: torch.Tensor
    bboxes: list[torch.Tensor]


class MiniYoloV3(nn.Module):
    @staticmethod
    def from_config(config: dict):
        backbone = []
        backbone_def = config["backbone"]

        for i, module_def in enumerate(backbone_def):
            module_type = module_def["type"]

            if i == 0:
                assert module_type == "Downsample", "First module must be Downsample"

            if i == len(backbone_def) - 1:
                assert module_type == "Downsample", "Last module must be Downsample"

            if module_type == "ResidualBlock":
                assert (
                    backbone_def[i - 1]["type"] == "Downsample"
                ), "ResidualBlock must follow Downsample"

                residual_blocks = []

                for i in range(module_def["num_blocks"]):
                    residual_blocks.append(ResidualBlock(module_def["channels"]))

                residual_blocks = nn.Sequential(*residual_blocks)
                backbone.append(residual_blocks)

            if module_type == "Downsample":
                assert (
                    i == 0 or backbone_def[i - 1]["type"] == "ResidualBlock"
                ), "Downsample must follow ResidualBlock"

                backbone.append(
                    Downsample(
                        in_channels=module_def["in_channels"],
                        out_channels=module_def["out_channels"],
                    )
                )

        return MiniYoloV3(
            image_size=config["image_size"],
            num_classes=config["num_classes"],
            anchors=config["anchors"],
            num_anchors_per_scale=config["num_anchors_per_scale"],
            backbone=backbone,
            num_detection_layers=config["num_detection_layers"],
        )

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        anchors: Union[torch.Tensor, list[list[int]]],
        num_anchors_per_scale: int,
        backbone: list[nn.Module],
        num_detection_layers: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_anchors_per_scale = num_anchors_per_scale
        self.num_detection_layers = num_detection_layers

        if isinstance(anchors, list):
            anchors = torch.tensor(anchors, dtype=torch.float32)

        assert (
            anchors.dim() == 2 and anchors.shape[-1] == 2
        ), "Anchors must be N x 2 tensor"

        assert (
            anchors.shape[0] == num_detection_layers * num_anchors_per_scale
        ), "Number of anchors MUST match num_detection_layers x num_anchors_per_scale"

        assert (
            num_detection_layers <= len(backbone) // 2 + 1
        ), "num_detection_layers has to be less than the number of downsample layers in the backbone"

        self.register_buffer("anchors", anchors)

        self.backbone = nn.ModuleList(backbone)

        self.upsample_layers = self._build_upsample_layers()
        self.detection_layers = self._build_detection_layers()

    def _build_upsample_layers(self):
        upsample_layers = nn.ModuleList([])

        # out channels of the last downsample layer in backbone
        down_idx = -1
        channels = self.backbone[down_idx].out_channels

        for i in range(self.num_detection_layers - 1):
            down_idx -= 2
            out_channels = channels - self.backbone[down_idx].out_channels
            upsample_layers.append(Upsample(channels, out_channels))

        return upsample_layers

    def _build_detection_layers(self):
        in_channels = self.backbone[-1].out_channels

        detection_layers = nn.ModuleList(
            [
                DetectionLayer(
                    in_channels=in_channels,
                    num_anchors=self.num_anchors_per_scale,
                    num_classes=self.num_classes,
                )
                for i in range(self.num_detection_layers)
            ]
        )

        return detection_layers

    def get_yolo_loss(
        self,
        images: torch.Tensor,
        bboxes: list[torch.Tensor],
        labels: list[torch.Tensor],
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        criterion = YOLOLoss(
            num_classes=self.num_classes,
            anchors=self.anchors,
            lambda_coord=lambda_coord,
            lambda_noobj=lambda_noobj,
        )
        pred = self(images)
        loss = criterion(pred, bboxes, labels)

        return loss

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        confidence_threshold: float = 0.9,
        iou_threshold: float = 0.5,
    ) -> MiniYoloV3Output:
        """
        Predict bounding boxes for input images

        Params:
            - images: (B, C, H, W) input images
            - confidence_threshold: minimum confidence score to keep a bounding box
            - iou_threshold: threshold for non-maximum suppression
        """
        pred = self(images)

        # pred bbox are in cxcywh format
        bbox_pred = convert_yolo_pred_to_bbox(pred, self.anchors, self.num_classes)
        A = self.anchors.shape[0]

        bounding_boxes = []
        for bbox_data in bbox_pred:
            bbox_data = bbox_data.view(-1, 5 + self.num_classes)
            bbox = bbox_data[..., :4]
            scores = bbox_data[..., 4]

            # keep bboxes with confidence score >= confidence_threshold
            keep_mask = scores >= confidence_threshold
            bbox_data = bbox_data[keep_mask]
            bbox = bbox[keep_mask]
            scores = scores[keep_mask]

            # convert bbox from cxcywh to xyxy
            bbox_xyxy = box_convert(bbox, in_fmt="cxcywh", out_fmt="xyxy")

            # non max suppression
            nms_idx = nms(bbox_xyxy, scores, iou_threshold)

            # bbox + confidence + labels + scores
            selected_bbox_pred = bbox_data[nms_idx]
            selected_bbox = selected_bbox_pred[:, :4]
            confidence = selected_bbox_pred[:, 4]
            class_probs = selected_bbox_pred[:, 5:]
            labels = class_probs.argmax(dim=-1)
            scores = class_probs.max(dim=-1).values

            result = {
                "bboxes": selected_bbox,
                "confidence": confidence,
                "labels": labels,
                "scores": scores,
            }
            bounding_boxes.append(result)

        return MiniYoloV3Output(pred=pred, bboxes=bounding_boxes)

    def forward(self, x: torch.Tensor):
        assert (
            x.shape[-1] == x.shape[-2] == self.image_size
        ), f"Input image size mismatch, expected: {self.image_size}x{self.image_size}"

        # backbone layer
        downsample_results = []  # save downsample results for skip connections
        for module in self.backbone:
            x = module(x)

            if isinstance(module, Downsample):
                downsample_results.append(x)

        # detection layers
        detect_results = []
        for i in range(self.num_detection_layers):
            detection_layer = self.detection_layers[i]

            if i == 0:
                detect = detection_layer(downsample_results.pop())
                detect_results.append(detect)
            else:
                # upsample
                upsample = self.upsample_layers[i - 1]
                x = upsample(x)

                # concat with result from skip connection
                skip = downsample_results.pop()
                x = torch.cat([x, skip], dim=1)

                detect = detection_layer(x)
                detect_results.append(detect)

        # upscale detection results to same shape and concat
        results = []

        for i, detect in enumerate(detect_results):
            exp_factor = len(detect_results) - 1 - i
            scale_factor = 2**exp_factor
            detect = F.interpolate(detect, scale_factor=scale_factor, mode="bilinear")
            results.append(detect)

        x = torch.cat(results, dim=1)  # (B, A(5 + C), H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, A(5 + C))

        return x


def convert_yolo_pred_to_bbox(
    pred: torch.Tensor, anchors: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Converts model prediction to bounding box predictions by transforming t_x, t_y to x, y and t_w, t_h to w, h
    and converting confidence scores and class scores to probabilities.

    Params:
        - pred: (B, H, W, A(C + 5)) in cxcywh format with confidence scores and class probabilities
    """

    B = pred.shape[0]
    W, H = pred.shape[2], pred.shape[1]
    A = anchors.shape[0]

    pred = pred.view(-1, H, W, A, 5 + num_classes)  # (B, H, W, A, 5 + C)

    pred_tx = pred[..., 0]  # (B, H, W, A)
    pred_ty = pred[..., 1]  # (B, H, W, A)
    pred_twh = pred[..., 2:4]  # (B, W, W, A, 2)
    pred_confidence = pred[..., 4:5].sigmoid()  # (B, H, W, A, 1)
    pred_class_scores = pred[..., 5:].softmax(dim=-1)  # (B, H, W, A, C)

    # get c_x, c_y
    X, Y = torch.arange(0, W, device=pred.device), torch.arange(H, device=pred.device)
    x_indices, y_indices = torch.meshgrid(X, Y, indexing="xy")
    x_offsets = x_indices.unsqueeze(0).unsqueeze(-1) * 1 / W
    y_offsets = y_indices.unsqueeze(0).unsqueeze(-1) * 1 / H

    # apply sigmoid to t_x and t_y and add offset
    pred_x = pred_tx.sigmoid() * (1 / W) + x_offsets
    pred_y = pred_ty.sigmoid() * (1 / H) + y_offsets

    # apply exp to twh and multiply with anchors
    anchors_batch = anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    pred_wh = pred_twh.exp() * anchors_batch

    # concatenate t_x, t_y, t_w, t_h, conf, and class scores
    bbox_pred = torch.cat(
        [
            pred_x.unsqueeze(-1),
            pred_y.unsqueeze(-1),
            pred_wh,
            pred_confidence,
            pred_class_scores,
        ],
        dim=-1,
    ).view(
        B, H, W, -1
    )  # (B, H, W, A(C + 5))

    return bbox_pred
