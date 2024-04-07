from .model import MiniYoloV3
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert


def calculate_mAP(model: MiniYoloV3, dataloader: DataLoader, device: str = "cpu"):
    metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")
    preds = []
    targets = []

    model.eval()
    for batch in dataloader:
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

        preds += preds_data
        targets += target_data

    metric.update(preds, targets)
    mAP = metric.compute()

    return mAP
