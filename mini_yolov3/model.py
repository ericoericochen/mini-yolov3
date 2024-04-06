import torch.nn as nn
import torch


class MiniYOLOV3(nn.Module):
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
