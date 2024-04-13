from torch.utils.data import Dataset
from typing import TypedDict
from PIL import Image
import torch
from torchvision import transforms as v2


class ObjectDetectionData(TypedDict):
    image: Image.Image
    bbox: torch.Tensor
    label: torch.Tensor


class ObjectDetectionDataset(Dataset):
    """
    Dataset for object detection. Each item consists of an image, its bounding boxes in COCO
    format (x, y, w, h), and corresponding labels for each box.
    """

    LABELS = []

    def __getitem__(self, idx: int) -> ObjectDetectionData:
        pass

    def get_original_image(self, idx: int) -> Image.Image:
        pass


class RandomColorJitter:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.prob:
            return v2.ColorJitter()(image)
        return image


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    bboxes = [item["bbox"] for item in batch]
    labels = [item["label"] for item in batch]

    return {
        "images": images,
        "bboxes": bboxes,
        "labels": labels,
    }
