import torch.nn as nn
import torch
from typing import Union
import torch.nn.functional as F


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


class MiniYOLOV3(nn.Module):
    @staticmethod
    def from_config(config: dict):
        print("[loading model from config...]")

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

        return MiniYOLOV3(
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

    def forward(self, x: torch.Tensor):
        assert (
            x.shape[-1] == x.shape[-2] == self.image_size
        ), f"Input image size mismatch, expected: {self.image_size}x{self.image_size}"

        print("forward")

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

        x = torch.cat(results, dim=1)

        return x
