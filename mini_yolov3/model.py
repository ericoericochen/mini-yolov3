import torch.nn as nn
import torch
from typing import Union


class Downsample(nn.Module):
    """
    Downsamples spatial resolution by 2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor):
        return self.downsample(x)


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


class Decoder(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass


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
            num_decoders=config["num_decoders"],
        )

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        anchors: Union[torch.Tensor, list[list[int]]],
        num_anchors_per_scale: int,
        backbone: list[nn.Module],
        num_decoders: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        if isinstance(anchors, list):
            anchors = torch.tensor(anchors, dtype=torch.float32)

        self.register_buffer("anchors", anchors)

        self.backbone = nn.ModuleList(backbone)
        # self.upsample = pass
        self.decoders = nn.ModuleList()

        for i in range(num_decoders):
            pass

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)

        return x


if __name__ == "__main__":
    residual = ResidualBlock(64)

    x = torch.randn(1, 64, 32, 32)

    x = residual(x)

    print(x.shape)
