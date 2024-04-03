from torch.utils.data import Dataset
from typing import TypedDict
from PIL import Image
import torch


class ObjectDetectionData(TypedDict):
    image: Image.Image
    bbox: torch.Tensor
    labels: torch.Tensor


class ObjectDetectionDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: int) -> ObjectDetectionData:
        pass
