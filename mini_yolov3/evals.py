from .model import MiniYoloV3
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision


def calculate_mAP(model: MiniYoloV3, dataloader: DataLoader):

    mAP = None

    for batch in dataloader:
        print(batch)
