from .model import MiniYOLOV3
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision


def calculate_mAP(model: MiniYOLOV3, dataloader: DataLoader):

    for batch in dataloader:
        print(batch)
