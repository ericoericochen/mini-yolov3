import torch
import torch.nn as nn


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
    ):
        pass

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
