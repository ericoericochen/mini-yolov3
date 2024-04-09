from .model import YOLO
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert
import torch
from tqdm import tqdm
from .loss import YOLOLoss


def serialize_mAP(mAP: dict):
    def serialize_metric(value: torch.Tensor):
        try:
            return value.item()
        except:
            return value.tolist()

    return {key: serialize_metric(value) for key, value in mAP.items()}


def calculate_loss(
    model: YOLO, dataloader: DataLoader, criterion: YOLOLoss, device: str = "cpu"
):
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader, leave=False):
        images, bboxes, labels = (
            batch["images"],
            batch["bboxes"],
            batch["labels"],
        )

        images = images.to(device)
        bboxes = [bbox.to(device) for bbox in bboxes]
        labels = [label.to(device) for label in labels]

        # make pred and compute loss
        pred = model(images)
        loss, loss_breakdown = criterion(pred, bboxes, labels)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def calculate_mAP(model: YOLO, dataloader: DataLoader, device: str = "cpu"):
    metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")

    model.eval()
    for batch in tqdm(dataloader, leave=False):
        images, bboxes, labels = (
            batch["images"],
            batch["bboxes"],
            batch["labels"],
        )

        images = images.to(device)
        bboxes = [bbox.to(device) for bbox in bboxes]
        labels = [label.to(device) for label in labels]

        output = model.inference(images)
        bounding_boxes = output.bboxes

        preds_data = [
            {
                "boxes": box_data["bboxes"],
                "scores": box_data["confidence"],
                "labels": box_data["labels"],
            }
            for box_data in bounding_boxes
        ]

        target_data = [
            {
                "boxes": box_convert(bbox, in_fmt="xywh", out_fmt="cxcywh"),
                "labels": label,
            }
            for bbox, label in zip(bboxes, labels)
        ]

        batch_mAP = metric(preds_data, target_data)

    mAP = metric.compute()
    mAP = serialize_mAP(mAP)

    return mAP
