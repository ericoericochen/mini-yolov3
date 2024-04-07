import torch
import torch.nn.functional as F
from .model import MiniYoloV3
from typing import Union
from PIL import Image


class YoloV3Pipeline:
    def __init__(self, model: MiniYoloV3):
        self.model = model

    def __call__(self, image: Image.Image, iou_threshold: float = 0.5):
        pass
