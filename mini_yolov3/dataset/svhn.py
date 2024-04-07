from .base import ObjectDetectionDataset, ObjectDetectionData
from datasets import load_dataset
from torchvision.transforms import v2
from ..utils import to_tensor
import torch
from typing import Literal
from PIL import Image
import random


class SVHNDataset(ObjectDetectionDataset):
    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        image_size: int = 32,
        normalize_bbox: bool = True,
        data_augment: bool = False,
        augment_prob: float = 0.25,
    ):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.normalize_bbox = normalize_bbox
        self.data_augment = data_augment
        self.augment_prob = augment_prob

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
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        nw, nh = image.width, image.height

        # convert image to tensor
        image = to_tensor(image)

        if self.data_augment:
            prob = torch.rand(1).item()
            if prob <= self.augment_prob:
                image = v2.ColorJitter()(image)

        # scale bounding boxes
        w_factor = nw / w
        h_factor = nh / h

        bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox[:, [0, 2]] *= w_factor  # scale x and w
        bbox[:, [1, 3]] *= h_factor  # scale y and h

        if self.normalize_bbox:
            bbox[:, [0, 2]] /= nw
            bbox[:, [1, 3]] /= nh

        labels = torch.tensor(labels, dtype=torch.int8)

        return {"image": image, "bbox": bbox, "labels": labels}
