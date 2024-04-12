from torch.utils.data import Dataset
from typing import TypedDict
from PIL import Image
import torch


class ObjectDetectionData(TypedDict):
    image: Image.Image
    bbox: torch.Tensor
    label: torch.Tensor


class ObjectDetectionDataset(Dataset):
    """
    Dataset for object detection. Each item consists of an image, its bounding boxes in COCO
    format (x, y, w, h), and corresponding labels for each box.
    """

    def __getitem__(self, idx: int) -> ObjectDetectionData:
        pass


def collate_fn(batch):
    # import time

    # a = time.time()

    images = torch.stack([item["image"] for item in batch], dim=0)
    bboxes = [item["bbox"] for item in batch]
    labels = [item["label"] for item in batch]

    # b = time.time()

    # print(f"Collate time: {b - a}")

    # raise RuntimeError

    return {
        "images": images,
        "bboxes": bboxes,
        "labels": labels,
    }
