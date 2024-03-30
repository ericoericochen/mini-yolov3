from torch.utils.data import Dataset
from datasets import load_dataset
import torch


class SVNHDataset(Dataset):
    def __init__(self, split="train", image_size: int = 64, type="pil"):
        assert split in ["train", "test", "extra"]
        assert type in ["pil", "tensor"]

        self.split = split
        self.image_size = image_size
        self.type = type
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

        bbox = torch.tensor(
            bbox, dtype=torch.float32
        )  # (x_center, y_center, width, height)

        # scale bbox with w_factor and f_factor
        bbox[:, [0, 2]] *= w_factor
        bbox[:, [1, 3]] *= h_factor

        labels = torch.tensor(labels, dtype=torch.long)

        return {"image": image, "bbox": bbox, "labels": labels}
