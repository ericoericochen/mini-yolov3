import torch
from .model import MiniYOLOV3
from .dataset import ObjectDetectionDataset
from torch.utils.data import DataLoader


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


class Trainer:
    def __init__(
        self,
        model: MiniYOLOV3,
        train_dataset: ObjectDetectionDataset,
        val_dataset: ObjectDetectionDataset = None,
        lr: float = 3e-4,
        batch_size: int = 32,
        num_epochs: int = 10,
        save_dir: str = None,
        device: str = get_device(),
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

        self.lr = lr
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.device = device

    def train(self):
        for epoch in range(self.num_epochs):
            print(epoch)
