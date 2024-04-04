from .base import ObjectDetectionDataset, ObjectDetectionData
from datasets import load_dataset

import torch
from typing import Literal
from PIL import Image


class SVHNDataset(ObjectDetectionDataset):
    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        image_transform=None,
        normalize_bbox: bool = True,
    ):
        super().__init__()
        self.split = split
        self.image_transform = image_transform
        self.normalize_bbox = normalize_bbox

        self.dataset = load_dataset("svhn", "full_numbers", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ObjectDetectionData:
        item = self.dataset[idx]
        image, bbox, labels = (
            item["image"],
            item["digits"]["bbox"],
            item["digits"]["label"],
        )

        # store original width and height to scale bbox later
        w, h = image.width, image.height

        # apply image transform to PIL image
        if self.image_transform:
            image = self.image_transform(image)

        if isinstance(image, Image.Image):
            nw, nh = image.width, image.height
        elif isinstance(image, torch.Tensor):
            nw, nh = image.shape[2], image.shape[1]

        w_factor = nw / w
        h_factor = nh / h

        # scale bounding boxes
        bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox[:, [0, 2]] *= w_factor
        bbox[:, [1, 3]] *= h_factor

        if self.normalize_bbox:
            bbox[:, [0, 2]] /= nw
            bbox[:, [1, 3]] /= nh

        labels = torch.tensor(labels, dtype=torch.int8)

        return {"image": image, "bbox": bbox, "labels": labels}
