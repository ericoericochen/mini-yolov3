import torch
from .model import MiniYOLOV3
from .dataset import ObjectDetectionDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import YOLOLoss


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
        weight_decay: float = 0.0,
        batch_size: int = 32,
        num_epochs: int = 10,
        lambda_coord: float = 1.0,
        lambda_noobj: float = 1.0,
        save_dir: str = None,
        device: str = get_device(),
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.save_dir = save_dir
        self.device = device

    def train(self):
        self.model.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        criterion = YOLOLoss(
            num_classes=self.model.num_classes,
            anchors=self.model.anchors,
            lambda_coord=self.lambda_coord,
            lambda_noobj=self.lambda_noobj,
        )

        losses = []
        num_iters = self.num_epochs * len(self.train_loader)
        with tqdm(total=num_iters) as pbar:
            for epoch in range(self.num_epochs):
                for data in self.train_loader:
                    images, bboxes, labels = (
                        data["images"],
                        data["bboxes"],
                        data["labels"],
                    )

                    images = images.to(self.device)
                    bboxes = [bbox.to(self.device) for bbox in bboxes]
                    labels = [label.to(self.device) for label in labels]

                    # make predictions
                    pred = self.model(images)

                    # compute loss
                    loss = criterion(pred, bboxes, labels)

                    # backprop
                    optimizer.zero_grad()
                    loss["loss"].backward()
                    optimizer.step()

                    losses.append(loss["loss"].item())
                    pbar.set_postfix(
                        loss=loss["loss"].item(),
                        # coord_loss=loss["coord_loss"].item(),
                        # obj_conf_loss=loss["obj_conf_loss"].item(),
                        # class_loss=loss["class_loss"].item(),
                        # noobj_loss=loss["noobj_loss"].item(),
                    )
                    pbar.update(1)

        return losses
