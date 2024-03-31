import torch.nn as nn
import torch


class MiniYOLOV3(nn.Module):
    def __init__(self, image_size: int, num_classes: int):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass
