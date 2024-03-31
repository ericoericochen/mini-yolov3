import torch.nn as nn
import torch


class MiniYOLOV3(nn.Module):
    def __init__(self, image_size: int, num_classes: int, anchors: torch.Tensor):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.anchors = anchors

        A = anchors.shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Conv2d(32, A * (5 + num_classes), 3, 2, 1),
        )

    def forward(self, x: torch.Tensor):
        assert (
            x.shape[2] == x.shape[3] == self.image_size
        ), f"Invalid image size, height and width should equal {self.image_size}"

        return self.conv(x)
