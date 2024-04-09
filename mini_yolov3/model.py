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
    def __init__(
        self,
        image_size: int,
        grid_size: int,
        predicted_bounding_boxes: int,
        num_classes: int,
        backbone: list,
        dense_layers: list[int],
    ):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.predicted_bounding_boxes = predicted_bounding_boxes
        self.num_classes = num_classes
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
        out_channels = self.grid_size * self.grid_size(
            5 * self.predicted_bounding_boxes + self.num_classes
        )

        feedforward.append(nn.Linear(in_channels, out_channels))

        return nn.Sequential(*feedforward)

    @property
    def config(self):
        return {
            "image_size": self.image_size,
            "grid_size": self.grid_size,
            "predicted_bounding_boxes": self.predicted_bounding_boxes,
            "num_classes": self.num_classes,
            "backbone": self.backbone,
            "dense_layers": self.dense_layers,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == x.shape[2] == self.image_size, "Invalid image size"

        x = self.backbone_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.feedforward(x)
        x = x.view(
            -1,
            self.grid_size,
            self.grid_size,
            5 * self.predicted_bounding_boxes + self.num_classes,
        )

        return x
