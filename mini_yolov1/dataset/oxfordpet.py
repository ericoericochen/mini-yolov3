from .base import ObjectDetectionDataset, ObjectDetectionData
from typing import Literal
import os
import xmltodict
import torch
from torchvision.ops import box_convert
from PIL import Image


class OxfordIIITPetDataset(ObjectDetectionDataset):
    LABELS = ["cat", "dog"]

    def __init__(
        self, root: str, split: Literal["trainval", "test"] = "trainval", transform=None
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.transform = transform

        split_txt_path = os.path.join(root, f"annotations/trainval.txt")
        self.index = []

        with open(split_txt_path, "r") as f:
            for line in f.readlines():
                data = line.strip().split(" ")
                image_name = data[0]
                image_path = os.path.join(root, f"images/{image_name}.jpg")
                label = int(data[2]) - 1
                label = torch.tensor([label])

                # parse bounding box
                bounding_box_xml_path = os.path.join(
                    root, f"annotations/xmls/{image_name}.xml"
                )

                try:
                    with open(bounding_box_xml_path) as xml_file:
                        bbox_data = xmltodict.parse(xml_file.read())
                        bbox = bbox_data["annotation"]["object"][
                            "bndbox"
                        ]  # xyxy format

                        bbox = torch.tensor(
                            [
                                [
                                    float(bbox["xmin"]),
                                    float(bbox["ymin"]),
                                    float(bbox["xmax"]),
                                    float(bbox["ymax"]),
                                ]
                            ]
                        )

                        # normalize bbox to be in [0, 1]
                        image = Image.open(image_path)
                        width, height = image.size
                        bbox[:, [0, 2]] /= width
                        bbox[:, [1, 3]] /= height

                        # convert bbox to xywh format
                        bbox = box_convert(bbox, in_fmt="xyxy", out_fmt="xywh")

                        index_data = {
                            "image": image_path,
                            "bbox": bbox,
                            "label": label,
                        }
                        self.index.append(index_data)
                except Exception as e:
                    pass

        torch.manual_seed(0)
        indices = torch.randperm(len(self.index))
        split = int(0.9 * len(indices))

        self.trainval_indices = indices[:split]
        self.test_indices = indices[split:]

    def __len__(self):
        if self.split == "trainval":
            return len(self.trainval_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx: int) -> ObjectDetectionData:
        if self.split == "trainval":
            i = self.trainval_indices[idx]
        else:
            i = self.test_indices[idx]

        i = i.item()
        data = self.index[i]
        image = Image.open(data["image"])
        if self.transform:
            image = self.transform(image)

        bbox = data["bbox"]
        label = data["label"]

        return {
            "image": image,
            "bbox": bbox,
            "label": label,
        }
