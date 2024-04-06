import torch.nn as nn
import torch


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


class MiniYOLOV3(nn.Module):
    def from_config(config: dict):
        pass

    def __init__(self, image_size: int, num_classes: int, anchors: torch.Tensor):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        self.register_buffer("anchors", anchors)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 15, 16, 16),
            # nn.Conv2d(3, 32, 3, 2),
            # nn.Conv2d(3, 32, 2, 2),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)

        return x


if __name__ == "__main__":
    residual = ResidualBlock(64)

    x = torch.randn(1, 64, 32, 32)

    x = residual(x)

    print(x.shape)


# class MiniYOLOV3(nn.Module):
#     def __init__(self, image_size: int, num_classes: int, anchors: torch.Tensor):
#         super().__init__()
#         self.image_size = image_size
#         self.num_classes = num_classes
#         # self.anchors = anchors

#         self.register_buffer("anchors", anchors)

#         A = anchors.shape[0]
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, self.image_size // 2, self.image_size // 2, 0),
#             # nn.ReLU(),
#             # nn.LayerNorm(16),
#             # nn.Conv2d(32, 32, 3, 2, 1),
#             # nn.ReLU(),
#             # nn.Conv2d(32, A * (5 + num_classes), 1, 1),
#             # nn.ReLU(),
#             # nn.Conv2d(A * (5 + num_classes), A * (5 + num_classes), 1, 1),
#         )

#     def forward(self, x: torch.Tensor):
#         # assert (
#         #     x.shape[2] == x.shape[3] == self.image_size
#         # ), f"Invalid image size, height and width should equal {self.image_size}"

#         x = self.conv(x)
#         x = x.permute(0, 2, 3, 1)

#         return x
