from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import torchvision
import utils


class SVNHDataset(Dataset):
    def __init__(
        self,
        split="train",
        image_size: int = 64,
        type="pil",
        normalize: bool = False,
        bbox_format="coco",
    ):
        assert split in ["train", "test", "extra"]
        assert type in ["pil", "tensor"]

        self.split = split
        self.image_size = image_size
        self.type = type
        self.normalize = normalize
        self.bbox_format = bbox_format
        self.dataset = load_dataset("svhn", "full_numbers", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns: {"image": PIL.Image.Image | torch.Tensor, "bbox": torch.Tensor, "labels": torch.Tensor}

        bbox is in COCO format (x_center, y_center, width, height)
        """
        item = self.dataset[idx]

        image, bbox, labels = (
            item["image"],
            item["digits"]["bbox"],
            item["digits"]["label"],
        )

        w_factor = self.image_size / image.width
        h_factor = self.image_size / image.height

        image = image.resize((self.image_size, self.image_size))

        if self.type == "tensor":
            image = torchvision.transforms.ToTensor()(image)

        bbox = torch.tensor(
            bbox, dtype=torch.float32
        )  # (x_center, y_center, width, height)

        if self.bbox_format == "xyxy":
            bbox = utils.coco_to_xyxy_format(bbox)

        # scale bbox with w_factor and f_factor
        bbox[:, [0, 2]] *= w_factor
        bbox[:, [1, 3]] *= h_factor

        if self.normalize:
            bbox = bbox / self.image_size

        labels = torch.tensor(labels, dtype=torch.int8)

        return {"image": image, "bbox": bbox, "labels": labels}


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    bbox = [item["bbox"] for item in batch]
    labels = [item["labels"] for item in batch]

    return {
        "images": images,
        "bbox": bbox,
        "labels": labels,
    }
